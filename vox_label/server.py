"""Local labeling server: serves blind spectrogram crops, collects binary labels.

Two design constraints come straight from the report and are enforced structurally
rather than by discipline.

**Blinding.** No endpoint ever returns a candidate's detection pattern z while it is
unlabeled. The estimand is what the annotator says about the spectrogram; showing which
detectors fired would let the thing being measured contaminate the measurement, and
every rate would move in the same direction.

**The grid.** `/api/report` can only evaluate a stream at a committed checkpoint that
stream has actually crossed. There is no code path that produces a *grid* interval at
an uncommitted length, which is the n = 4,500 failure of section 6.

Watching progress is nevertheless allowed, because `/api/live` answers with anytime-valid
confidence sequences instead (vox_label/anytime.py). Those are valid simultaneously at
every t, so recomputing them after each annotation costs nothing; they are about 1.4x
wider at equal n, which is the price. And reporting a grid interval afterwards is still
sound: eq. 9 covers at every checkpoint simultaneously, so whichever checkpoint gets
reported covers regardless of how it was chosen -- including chosen after seeing all of
them. What the grid never licenses is reading it *between* checkpoints.

Labels append to `labels.jsonl` and are fsynced per write, so a crash costs at most the
candidate on screen, and the log stays replayable and auditable.
"""
import json
import os
import threading
import traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from vox_label import setup as setup_mod
from vox_label.anytime import cs_interval
from vox_label.exact_grid import GammaCache, exact_grid_interval, f1_from_jaccard
from vox_label.ranking import corpus_stats, rank
from vox_label.render import band_rows, clip_channels, to_jpeg
from vox_label.streams import STREAM_KINDS, counts_at

STATIC = Path(__file__).parent / "static"


class LabelService:
    """All campaign state and logic; the HTTP layer below is a thin shell over it."""

    def __init__(self, campaign, source, annotator="anon", pad=0.35, disp_w=900,
                 jpeg_q=72, reveal_every=500, f_lo_khz=5.0, f_hi_khz=60.0,
                 nyquist_khz=62.5):
        self.c = campaign
        self.source = source
        self.annotator = annotator
        self.pad, self.disp_w, self.jpeg_q = pad, disp_w, jpeg_q
        self.reveal_every = reveal_every
        self.f_lo_hz, self.f_hi_hz = f_lo_khz * 1000, f_hi_khz * 1000
        self.nyquist_hz = nyquist_khz * 1000
        self.lock = threading.Lock()
        self.labels = campaign.load_labels()
        self.log = open(campaign.labels_path, "a")
        self._gamma = {}
        self._report = {"status": "idle", "at": None, "rows": []}
        self._report_thread = None
        self._ranking = {"status": "idle", "at": None}
        self._ranking_thread = None
        self._corpus = None

    # ── labeling ────────────────────────────────────────────────────────────────

    def next_candidate(self):
        """The next unlabeled candidate in the committed permutation.

        Nothing about z is included: the response carries only what the annotator is
        allowed to see.
        """
        with self.lock:
            row = self.c.next_unlabeled(self.labels)
            n_done = len(self.labels)
        if row is None:
            return {"done": True, "n_labeled": n_done}
        t0, t1 = row["t_start"], row["t_end"]
        lo, hi = t0 - self.pad, t1 + self.pad
        # Every channel the clip has, not the ones that fired -- see render.clip_channels.
        channels = clip_channels(self.source, row["clip"]) or row["channels"]
        return {
            "done": False,
            "cand_id": row["cand_id"],
            "clip": row["clip"],
            "channels": channels,
            "t_start": t0,
            "t_end": t1,
            "duration_ms": 1000 * (t1 - t0),
            "view_lo": lo,
            "view_hi": hi,
            "n_labeled": n_done,
            **self.progress(),
        }

    def record(self, cand_id, label, combined=False, ms_elapsed=None):
        """Record one annotation.

        `combined` marks a candidate containing overlapping / simultaneous calls from
        more than one animal -- the same distinction the GT CSVs draw with their
        `name` column ("vox" vs "combined", `vox_tracer/scoring.py:292`). It is a
        *second, independent* annotation, not a modifier of the label: a combined call
        is still a genuine vocalization, so it is labeled real and marked combined. It
        therefore has no effect on any precision, recall or F1 stream, and exists so
        the campaign measures how common overlap is while the corpus is being read
        anyway.
        """
        rec = {
            "cand_id": int(cand_id),
            "label": int(label),
            "combined": bool(combined),
            "annotator": self.annotator,
            "ts": datetime.now(timezone.utc).isoformat(),
            "ms_elapsed": ms_elapsed,
            "clip": self.c.pool[int(cand_id)]["clip"],
            "campaign": self.c.spec["pool_sha256"][:12],
        }
        with self.lock:
            self._append(rec)
            self.labels[int(cand_id)] = rec
        return rec

    def retract_last(self):
        """Undo the most recent label only.

        Recorded as a retraction line rather than by editing the log. Correcting a
        misclick is legitimate; the append-only record is what keeps it distinguishable
        from quietly reshaping the data after the fact.
        """
        with self.lock:
            if not self.labels:
                return None
            last_id = max(self.labels, key=lambda k: self.labels[k]["ts"])
            rec = {"cand_id": last_id, "retracted": True, "annotator": self.annotator,
                   "ts": datetime.now(timezone.utc).isoformat()}
            self._append(rec)
            self.labels.pop(last_id, None)
        return last_id

    def _append(self, rec):
        self.log.write(json.dumps(rec) + "\n")
        self.log.flush()
        os.fsync(self.log.fileno())

    def close(self):
        """Release the label log and any open spectrogram handles."""
        try:
            self.log.close()
        except OSError:
            pass
        closer = getattr(self.source, "close", None)
        if closer:
            closer()

    # ── progress: counts only, never an interval ────────────────────────────────

    def progress(self):
        n = len(self.labels)
        n_max = self.c.spec["n_max"]
        nxt = min(((n // self.reveal_every) + 1) * self.reveal_every, n_max)
        return {
            "n_labeled": n,
            "n_max": n_max,
            "next_reveal": nxt,
            "to_next_reveal": max(0, nxt - n),
            # Due exactly when a reveal boundary has just been reached, so the panel
            # fires on the 500th label rather than the 501st.
            "reveal_due": (n > 0 and n % self.reveal_every == 0) or n >= n_max,
        }

    # ── the report, only at committed checkpoints ───────────────────────────────

    def gamma_cache(self, grid):
        key = tuple(grid)
        if key not in self._gamma:
            self._gamma[key] = GammaCache(grid, self.c.alpha)
        return self._gamma[key]

    def compute_report(self):
        """One row per (rule, target), each at the last checkpoint its stream crossed."""
        annotations = self.c.annotations(self.labels)
        rows = []
        for rule in self.c.rules:
            fn = self.c.rule_fn(rule)
            for kind, extract in STREAM_KINDS.items():
                stream = extract(annotations, fn)
                grid = self.c.grid(rule, kind)
                counts, crossed = counts_at(stream, grid)
                if crossed == 0:
                    rows.append({
                        "rule": rule, "target": kind, "available": False,
                        "stream_len": len(stream), "next_checkpoint": grid[0],
                    })
                    continue
                j = crossed - 1
                lo, hi = exact_grid_interval(counts, grid, j, self.c.alpha,
                                             gamma_of_p=self.gamma_cache(grid))
                est = counts[j] / grid[j]
                if kind == "f1":
                    lo, hi, est = (f1_from_jaccard(lo), f1_from_jaccard(hi),
                                   f1_from_jaccard(est))
                rows.append({
                    "rule": rule, "target": kind, "available": True,
                    "checkpoint": grid[j], "checkpoint_index": j + 1,
                    "n_checkpoints": len(grid), "successes": counts[j],
                    "estimate": est, "lo": lo, "hi": hi,
                    "half_width": (hi - lo) / 2, "stream_len": len(stream),
                    "next_checkpoint": grid[j + 1] if j + 1 < len(grid) else None,
                })
        return rows

    def report(self, start=False):
        """Cached report; kicks off a background computation when asked.

        Inverting the grid takes seconds per interval at realistic checkpoint sizes and
        there are ~20 of them, so this runs off the request thread and the page polls.
        """
        with self.lock:
            state = dict(self._report)
            busy = self._report_thread is not None and self._report_thread.is_alive()
        if start and not busy:
            self._report = {"status": "computing", "at": len(self.labels), "rows": []}
            self._report_thread = threading.Thread(target=self._run_report, daemon=True)
            self._report_thread.start()
            return {"status": "computing", "rows": []}
        return state

    def _run_report(self):
        try:
            rows = self.compute_report()
            self._report = {"status": "ready", "at": len(self.labels), "rows": rows}
        except Exception:
            self._report = {"status": "error", "at": None, "rows": [],
                            "error": traceback.format_exc(limit=3)}

    # ── the partial order of main.pdf: anytime-valid, recomputed as labels arrive ──

    def compute_ranking(self):
        params = self.c.ranking_params()
        if params is None:
            return None
        if self._corpus is None:
            self._corpus = corpus_stats(list(self.c.pool.values()), self.c.rules,
                                        {r: self.c.rule_fn(r) for r in self.c.rules})
        return rank(self.c.annotations(self.labels), self.c.rules,
                    {r: self.c.rule_fn(r) for r in self.c.rules}, self._corpus,
                    params["omega"], params["delta"], self.c.alpha,
                    params.get("a_star_guess", 0.5))

    def ranking(self, start=False):
        """Latest ranking; with `start`, recompute in the background if it is stale.

        A pass over every pair takes about a second, too long for the label request.
        """
        with self.lock:
            state = dict(self._ranking)
            busy = self._ranking_thread is not None and self._ranking_thread.is_alive()
            n = len(self.labels)
        if start and not busy and state.get("at") != n:
            self._ranking = {**state, "status": "computing"}
            self._ranking_thread = threading.Thread(target=self._run_ranking, args=(n,),
                                                    daemon=True)
            self._ranking_thread.start()
            state["status"] = "computing"
        return state

    def _run_ranking(self, n):
        try:
            self._ranking = {"status": "ready", "at": n, "result": self.compute_ranking()}
        except Exception:
            self._ranking = {"status": "error", "at": None,
                             "error": traceback.format_exc(limit=3)}

    # ── crops ───────────────────────────────────────────────────────────────────

    def crop_jpeg(self, cand_id, ch):
        row = self.c.pool[int(cand_id)]
        lo, hi = row["t_start"] - self.pad, row["t_end"] + self.pad
        gray, a, b = self.source.crop(row["clip"], int(ch), lo, hi)
        top, bot = band_rows(gray.shape[0], self.nyquist_hz, self.f_lo_hz, self.f_hi_hz)
        return to_jpeg(gray[top:bot], self.disp_w, self.jpeg_q), a, b

    # ── the live panel: anytime-valid, so looking costs nothing ──────────────────

    def live(self):
        """Confidence sequences for every rule and target at the current stream lengths.

        Distinct from `compute_report` in both method and status. These are anytime-valid
        (vox_label/anytime.py), so they may be recomputed after every single annotation
        with no grid and no penalty -- roughly 1.4x wider at equal n, which is the price
        of unrestricted looking. The committed-grid intervals remain the headline numbers
        and still appear only at checkpoints.
        """
        annotations = self.c.annotations(self.labels)
        out = []
        for rule in self.c.rules:
            fn = self.c.rule_fn(rule)
            for kind, extract in STREAM_KINDS.items():
                stream = extract(annotations, fn)
                n, x = len(stream), sum(stream)
                lo, hi = cs_interval(x, n, self.c.alpha)
                est = x / n if n else None
                if kind == "f1" and est is not None:
                    lo, hi, est = (f1_from_jaccard(lo), f1_from_jaccard(hi),
                                   f1_from_jaccard(est))
                grid = self.c.grid(rule, kind)
                _, crossed = counts_at(stream, grid)
                out.append({
                    "rule": rule, "target": kind, "n": n, "x": x, "estimate": est,
                    "lo": lo, "hi": hi, "half_width": (hi - lo) / 2,
                    "next_checkpoint": grid[crossed] if crossed < len(grid) else None,
                })
        return out



class Session:
    """The campaign currently bound to the server, if one is yet.

    The server can start with nothing bound: `/setup` then chooses the dataset and
    either resumes a campaign part way through or freezes a new one, and every
    labeling endpoint returns 409 until it does. Binding is the only mutation, and
    it never touches a frozen spec.
    """

    VIEW_KEYS = ("pad", "disp_w", "jpeg_q", "reveal_every", "f_lo_khz", "f_hi_khz",
                 "nyquist_khz", "prefix", "backend")

    def __init__(self, svc=None, annotator="anon", view_overrides=None):
        self.svc = svc
        self.annotator = annotator
        self.view_overrides = dict(view_overrides or {})
        self.lock = threading.Lock()

    def bind(self, campaign_dir, dataset=None, view=None, annotator=None, verify=True,
             ranking=None):
        """Open a campaign and make it the one being labeled.

        `ranking` (omega, delta) is frozen into ranking.json only if the campaign has
        none yet; an existing choice always wins.
        """
        merged = {**self.view_overrides, **(view or {})}
        campaign, source, dataset, settings = setup_mod.open_campaign(
            campaign_dir, dataset, merged, verify=verify)
        campaign.freeze_ranking(**(ranking or {}))
        svc = LabelService(
            campaign, source, annotator=annotator or self.annotator,
            pad=float(settings["pad"]), disp_w=int(settings["disp_w"]),
            jpeg_q=int(settings["jpeg_q"]), reveal_every=int(settings["reveal_every"]),
            f_lo_khz=float(settings["f_lo_khz"]), f_hi_khz=float(settings["f_hi_khz"]),
            nyquist_khz=float(settings["nyquist_khz"]))
        # The spec may predate the dataset field, so the bound dataset is what counts.
        svc.dataset = dataset
        # Beside the spec, not in it: resuming reuses the view without re-entering it.
        campaign.save_view({k: settings[k] for k in self.VIEW_KEYS if k in settings})
        with self.lock:
            old, self.svc, self.annotator = self.svc, svc, svc.annotator
        if old is not None:
            old.close()
        return svc

    def status(self):
        svc = self.svc
        if svc is None:
            return {"bound": False, "annotator": self.annotator}
        return {
            "bound": True,
            "name": svc.c.spec["name"],
            "dir": svc.c.dir.as_posix(),
            "dataset": getattr(svc, "dataset", None) or svc.c.dataset,
            "annotator": svc.annotator,
            **svc.progress(),
        }


def make_handler(session):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass  # the labeling session should not scroll a request log past the user

        def _send(self, code, body, ctype="application/json", extra=None):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _page(self, name):
            return self._send(200, (STATIC / name).read_text(),
                              "text/html; charset=utf-8")

        def _svc(self):
            """The bound service, or None after answering 409.

            Labeling has no meaning until a campaign is chosen, and the page turns the
            409 into a redirect to /setup rather than showing an empty panel.
            """
            svc = session.svc
            if svc is None:
                self._send(409, {"error": "no campaign bound", "setup": True})
            return svc

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path in ("/", "/index.html"):
                    return self._page("app.html" if session.svc else "setup.html")
                if u.path == "/setup":
                    return self._page("setup.html")
                if u.path == "/api/setup":
                    return self._send(200, {**setup_mod.options(session.view_overrides),
                                            "current": session.status()})
                svc = self._svc()
                if svc is None:
                    return
                if u.path == "/api/next":
                    return self._send(200, svc.next_candidate())
                if u.path == "/api/progress":
                    return self._send(200, svc.progress())
                if u.path == "/api/live":
                    return self._send(200, {"rows": svc.live(),
                                            "n_labeled": len(svc.labels)})
                if u.path == "/api/report":
                    start = q.get("start", ["0"])[0] == "1"
                    return self._send(200, svc.report(start=start))
                if u.path == "/api/ranking":
                    start = q.get("start", ["0"])[0] == "1"
                    return self._send(200, svc.ranking(start=start))
                if u.path == "/api/crop":
                    jpg, a, b = svc.crop_jpeg(q["cand_id"][0], q["ch"][0])
                    return self._send(200, jpg, "image/jpeg",
                                      {"X-Span-Lo": f"{a:.6f}", "X-Span-Hi": f"{b:.6f}"})
                if u.path == "/api/meta":
                    return self._send(200, {
                        "name": svc.c.spec["name"],
                        "detectors": svc.c.detectors,
                        "rules": svc.c.rules,
                        "n_candidates": svc.c.spec["n_candidates"],
                        "alpha": svc.c.alpha,
                        "ranking": svc.c.ranking_params(),
                        "annotator": svc.annotator,
                    })
                return self._send(404, {"error": "not found"})
            except (FileExistsError, FileNotFoundError, KeyError, ValueError) as e:
                # A bad entry on the setup form is the user's, not the server's.
                return self._send(400, {"error": f"{type(e).__name__}: {e}"})
            except Exception as e:
                return self._send(500, {"error": str(e),
                                        "trace": traceback.format_exc(limit=3)})

        def do_POST(self):
            u = urlparse(self.path)
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            try:
                if u.path == "/api/setup/plan":
                    keys = ("rules", "n_max", "J", "alpha", "alpha_per_rule",
                            "recall_stream_max", "p_real", "omega", "delta")
                    return self._send(200, setup_mod.plan(
                        payload["pool_csv"], payload["detectors"],
                        **{k: payload[k] for k in keys if payload.get(k) is not None}))
                if u.path == "/api/setup/start":
                    return self._send(200, _start(session, payload))
                svc = self._svc()
                if svc is None:
                    return
                if u.path == "/api/label":
                    svc.record(payload["cand_id"], payload["label"],
                               payload.get("combined", False), payload.get("ms_elapsed"))
                    return self._send(200, svc.next_candidate())
                if u.path == "/api/undo":
                    svc.retract_last()
                    return self._send(200, svc.next_candidate())
                return self._send(404, {"error": "not found"})
            except (FileExistsError, FileNotFoundError, KeyError, ValueError) as e:
                # A bad entry on the setup form is the user's, not the server's.
                return self._send(400, {"error": f"{type(e).__name__}: {e}"})
            except Exception as e:
                return self._send(500, {"error": str(e),
                                        "trace": traceback.format_exc(limit=3)})

    return Handler


def _start(session, payload):
    """Handle /api/setup/start: freeze a new campaign if asked, then bind one."""
    view = payload.get("view") or {}
    dataset = payload.get("dataset")
    ranking = {k: payload[k] for k in ("omega", "delta") if payload.get(k) is not None}
    if payload.get("mode") == "create":
        path, _ = setup_mod.freeze(
            payload["out_dir"], payload["pool_csv"], payload["detectors"], dataset,
            rules=payload.get("rules"),
            **{k: payload[k] for k in setup_mod.GRID_DEFAULTS if k in payload})
        campaign_dir = str(Path(path).parent)
    else:
        campaign_dir = payload["campaign_dir"]
    session.bind(campaign_dir, dataset=dataset, view=view,
                 annotator=payload.get("annotator"), ranking=ranking)
    return session.status()


def serve(session, host="127.0.0.1", port=8765):
    if isinstance(session, LabelService):  # a service may still be passed directly
        session = Session(session, annotator=session.annotator)
    httpd = ThreadingHTTPServer((host, port), make_handler(session))
    svc = session.svc
    if svc is None:
        print(f"vox label setup  ->  http://{host}:{port}")
        print("  no campaign bound yet; pick a dataset and campaign on that page")
    else:
        print(f"labeling {svc.c.spec['name']}  ->  http://{host}:{port}")
        print(f"  {len(svc.labels)} labeled / {svc.c.spec['n_max']} budget "
              f"/ {svc.c.spec['n_candidates']} in pool")
    httpd.serve_forever()

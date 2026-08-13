"""Render concrete spectrogram chunks with sam3_best SAM masks overlaid, for the
three evaluation outcomes:

  combined DETECTED  — a name=='combined' GT window overlapped by >=1 prediction
  combined MISSED    — a combined window with no overlapping prediction (FN)
  vox FALSE POSITIVE — a merged predicted event overlapping NO vox GT (spurious)

Chunk images are located from the SPECTROGRAM directory (which holds every 1-s
chunk), NOT the prediction COCO — a missed window has no prediction, so its chunk
is often absent from the pred COCO and would otherwise be dropped. Windows whose
time is past the last spectrogram chunk (event beyond the processed audio) are
emitted as explicit out-of-coverage placeholders instead of being silently lost.

Masks come from the COCO 'segmentation' polygons; GT time-spans are drawn as
translucent bands. Scored with vox_tracer.scoring, exactly like scripts/evaluate.py.
"""
import base64, json, re, sys
from pathlib import Path
import cv2, numpy as np

ROOT = Path("/mnt/home/the10/ssl")
sys.path.insert(0, str(ROOT))
from vox_tracer.scoring import iou_1d, bbox_to_time, load_gt_from_csv, score_combined

PRED = ROOT / "outputs/sam3_best"
SPEC = ROOT / "outputs/spectrograms"
DATA = ROOT / "data"
CH = [118, 35]

GREEN = (60, 200, 60); RED = (40, 40, 220); CYAN = (255, 210, 0)
ORANGE = (0, 150, 255); GRAY = (150, 150, 150)
CHUNK_RE = re.compile(r"chunk_\d+_t([0-9.]+)-([0-9.]+)\.png$")


def load(pred_dir, ch):
    """Prediction COCO -> (img_by_fname, ann_by_fname). ({}, {}) if absent."""
    p = pred_dir / f"coco_ch_{ch}.json"
    if not p.exists():
        return {}, {}
    c = json.load(open(p))
    img_by_id = {im["id"]: im for im in c["images"]}
    img_by_fname = {im["file_name"]: im for im in c["images"]}
    ann_by_fname = {}
    for a in c["annotations"]:
        ann_by_fname.setdefault(img_by_id[a["image_id"]]["file_name"], []).append(a)
    return img_by_fname, ann_by_fname


def spec_index(session, ch):
    """Every spectrogram chunk for a channel: sorted [(start, end, fname)], max_end."""
    d = SPEC / session
    out = []
    if d.exists():
        for f in d.iterdir():
            if f"headmic_{ch}_" not in f.name:
                continue
            m = CHUNK_RE.search(f.name)
            if m:
                out.append((float(m.group(1)), float(m.group(2)), f.name))
    out.sort()
    return out, (out[-1][1] if out else 0.0)


def find_chunk(sidx, t):
    for s, e, fn in sidx:
        if s <= t < e:
            return fn, {"file_name": fn, "window_start_sec": s, "window_end_sec": e}
    return None, None


def band(img, t0, t1, ws, we, color, alpha=0.30):
    h, w = img.shape[:2]; dur = we - ws
    x0 = max(0, int((t0 - ws) / dur * w)); x1 = min(w, int((t1 - ws) / dur * w))
    x1 = max(x1, x0 + 2)
    ov = img.copy(); cv2.rectangle(ov, (x0, 0), (x1, h), color, -1)
    out = cv2.addWeighted(ov, alpha, img, 1 - alpha, 0)
    cv2.rectangle(out, (x0, 0), (x1, h - 1), color, 2)
    return out


def draw_polys(img, polys, color, thick=2):
    cnts = [np.array(p, np.float32).reshape(-1, 1, 2).astype(np.int32) for p in polys]
    if cnts:
        cv2.drawContours(img, cnts, -1, color, thick)
    return img


def encode(session, im, gt_span, gt_color, pred_anns, hi_span, out_w=520, q=88):
    path = SPEC / session / im["file_name"]
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    ws, we = im["window_start_sec"], im["window_end_sec"]
    if gt_span is not None:
        bgr = band(bgr, gt_span[0], gt_span[1], ws, we, gt_color)
    for ann, is_hi in pred_anns:
        polys = ann.get("segmentation", [])
        col = hi_span if is_hi else GRAY
        if polys:
            bgr = draw_polys(bgr, polys, col)
        else:
            x, y, w, h = ann["bbox"]
            cv2.rectangle(bgr, (int(x), int(y)), (int(x + w), int(y + h)), col, 2)
    bgr = cv2.resize(bgr, (out_w, round(out_w * 275 / 520)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else None


detected, missed, fps, oob = [], [], [], []

for exp in sorted(PRED.glob("experiment_*")):
    for idx in sorted(exp.glob("idx_*")):
        session = f"{exp.name}/{idx.name}"
        try:
            vox, combined = load_gt_from_csv(DATA / exp.name / idx.name)
        except FileNotFoundError:
            continue
        if not combined:
            continue
        info = {}      # ch -> (ibf, abf, sidx, max_end)
        for ch in CH:
            ibf, abf = load(idx, ch)
            sidx, mend = spec_index(session, ch)
            if ibf or sidx:
                info[ch] = (ibf, abf, sidx, mend)
        if not info:
            continue

        pooled = []
        for ch, (ibf, abf, sidx, mend) in info.items():
            for fn, anns in abf.items():
                im = ibf[fn]
                for a in anns:
                    t0, t1 = bbox_to_time(a["bbox"], im["window_start_sec"], im["window_end_sec"], im["width"])
                    pooled.append((t0, t1))

        for (c0, c1) in combined:
            mid = (c0 + c1) / 2
            hit = any(iou_1d(c0, c1, p0, p1) > 0 for p0, p1 in pooled)
            cands = []                       # (ch, im, marked, n_over)
            max_end = 0.0
            for ch, (ibf, abf, sidx, mend) in info.items():
                max_end = max(max_end, mend)
                fn, im = find_chunk(sidx, mid)
                if fn is None:
                    continue
                marked = []
                for a in abf.get(fn, []):
                    gim = ibf[fn]
                    t0, t1 = bbox_to_time(a["bbox"], gim["window_start_sec"], gim["window_end_sec"], gim["width"])
                    marked.append((a, iou_1d(c0, c1, t0, t1) > 0))
                cands.append((ch, im, marked, sum(m for _, m in marked)))

            if hit and cands:
                ch, im, marked, n_over = max(cands, key=lambda c: c[3])
                b64 = encode(session, im, (c0, c1), GREEN, marked, CYAN, out_w=360, q=80)
                if b64:
                    detected.append({"session": session, "ch": ch, "t": f"{c0:.3f}-{c1:.3f}s",
                                     "n_over": n_over, "img": b64})
            elif cands:
                for ch, im, marked, n_over in cands:
                    b64 = encode(session, im, (c0, c1), RED, marked, CYAN)
                    if b64:
                        missed.append({"session": session, "ch": ch, "t": f"{c0:.3f}-{c1:.3f}s",
                                       "n_over": n_over, "img": b64})
            # else: window is past the last processed chunk — nothing to render; tallied below
            if not hit and not cands:
                oob.append((session, f"{c0:.3f}-{c1:.3f}s"))

# ---- vox false positives (precision side), sampled across recordings ----
import random
random.seed(0)
fp_sessions = list(sorted(PRED.glob("experiment_*"))); random.shuffle(fp_sessions)
for exp in fp_sessions:
    if len(fps) >= 30:
        break
    for idx in sorted(exp.glob("idx_*")):
        session = f"{exp.name}/{idx.name}"
        try:
            vox, combined = load_gt_from_csv(DATA / exp.name / idx.name)
        except FileNotFoundError:
            continue
        boxes = []
        for ch in CH:
            ibf, abf = load(idx, ch)
            for fn, anns in abf.items():
                im = ibf[fn]
                for a in anns:
                    t0, t1 = bbox_to_time(a["bbox"], im["window_start_sec"], im["window_end_sec"], im["width"])
                    boxes.append((t0, t1, a.get("score"), ch, a, im))
        if not boxes:
            continue
        res = score_combined(vox, [(b[0], b[1], b[2]) for b in boxes], 0.0)
        fp_events = [e for e in res["pred_events"] if e not in res["tp_events"]]
        if not fp_events:
            continue
        es, ee = fp_events[0]
        for (t0, t1, sc, ch, a, im) in boxes:
            if iou_1d(t0, t1, es, ee) > 0:
                b64 = encode(session, im, (es, ee), ORANGE, [(a, True)], ORANGE)
                if b64:
                    fps.append({"session": session, "ch": ch, "t": f"{t0:.3f}-{t1:.3f}s",
                                "score": None if sc is None else round(sc, 2), "img": b64})
                break
        break

detected.sort(key=lambda c: (c["session"], c["t"]))
missed.sort(key=lambda c: (c["session"], c["t"]))

manifest = {"detected": detected, "missed": missed, "fps": fps,
            "missed_windows": len({(c["session"], c["t"]) for c in missed}),
            "missed_oob": len(oob), "oob": sorted(oob)}
out = ROOT / "outputs/eval/sam3_best/mask_cells.json"
json.dump(manifest, open(out, "w"))
print(f"detected={len(detected)} missed_cells={len(missed)} "
      f"missed_windows={manifest['missed_windows']} out_of_coverage(not shown)={manifest['missed_oob']} "
      f"fps={len(fps)}  {out.stat().st_size/1e3:.0f} KB")

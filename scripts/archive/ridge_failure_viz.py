"""
Build a self-contained interactive HTML comparing the ridge detector against
squeakout and sam3, focused on what ridge SYSTEMATICALLY MISSES (false negatives)
or OVER-DETECTS (false positives).

Each ground-truth call ridge missed, and each region ridge over-detected, becomes
a card showing the actual 1-second spectrogram for both channels with every
method's COCO segmentation mask overlaid as a toggleable colored polygon. A dashed
band marks the exact missed / over-detected region in focus. Cards are grouped so
you can isolate ridge-specific failures (missed but caught by another method;
over-detected where no other method fired) from generally hard regions.

Scoring mirrors scripts/evaluate.py --gt-csv: recording-level, event-level time
overlap against the sampled contiguous GT, within each recording's labeled coverage
span (predictions outside coverage are dropped so FP is only counted where GT is
complete). All scoring goes through vox_tracer.scoring, the shared source of truth.

Usage
-----
    python scripts/ridge_failure_viz.py \
        --gt-csv data/sampled_contiguous_detections.csv \
        --out outputs/eval/ridge_failure_analysis.html

    # full, uncapped local version (larger file, not artifact-friendly):
    python scripts/ridge_failure_viz.py --no-cap --disp-w 560 --jpeg-q 70
"""
import argparse
import base64
import glob as _glob
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from vox_tracer.scoring import (
    iou_1d, bbox_to_time, merge_intervals, load_gt_map, score_combined,
)

# categories the cap applies to: the "not ridge-specific" ones (hard for everyone /
# noise that trips multiple methods). The ridge-specific categories are never capped.
CAPPED_CATS = ("miss_all", "fp_shared")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt-csv", default="data/sampled_contiguous_detections.csv",
                   help="global contiguous-detections CSV that defines GT + coverage")
    p.add_argument("--spec-base", default="outputs/spectrograms",
                   help="base dir of per-recording spectrogram chunk PNGs")
    p.add_argument("--ridge-dir",     default="outputs/ridge_filtered")
    p.add_argument("--squeakout-dir", default="outputs/squeakout_filtered")
    p.add_argument("--sam3-dir",      default="outputs/sam3_best")
    p.add_argument("--ridge-rejected-dir", default="outputs/ridge_rejected",
                   help="stage-2-rejected ridge stage-1 candidates (tagged reject_reason) from "
                        "run_ridge's --reject-out-dir; overlaid on 'ridge missed' cards to show "
                        "whether ridge saw the call and a gate cut it, or never saw it at all. "
                        "Pass a nonexistent path to skip.")
    p.add_argument("--out", default="outputs/eval/ridge_failure_analysis.html")
    p.add_argument("--channels", default="118,35")
    p.add_argument("--disp-w", type=int, default=440,
                   help="encoded spectrogram width in px (smaller = smaller HTML)")
    p.add_argument("--jpeg-q", type=int, default=46, help="JPEG quality for embedded images")
    p.add_argument("--cap", type=int, default=90,
                   help=f"max cards per non-ridge-specific category ({', '.join(CAPPED_CATS)})")
    p.add_argument("--no-cap", action="store_true", help="keep every card (large file)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    def abspath(p):
        p = Path(p)
        return p if p.is_absolute() else ROOT / p

    spec_base = abspath(args.spec_base)
    gt_csv    = abspath(args.gt_csv)
    out_html  = abspath(args.out)
    channels  = [int(c) for c in args.channels.split(",")]
    methods = {
        "ridge":     abspath(args.ridge_dir),
        "squeakout": abspath(args.squeakout_dir),
        "sam3":      abspath(args.sam3_dir),
    }
    ridge_rejected_dir = abspath(args.ridge_rejected_dir)

    gt_by_rec, coverage_by_rec = load_gt_map(str(gt_csv))
    print(f"GT: {sum(len(v) for v in gt_by_rec.values())} vox across {len(gt_by_rec)} recordings")

    # ── embedded-image cache (dedup identical chunk PNGs across cards) ──────────
    img_cache = {}     # (session, ch, fname) -> img_id | None
    img_store = []     # [{b64, w, h}]

    def get_img(session, ch, fname):
        key = (session, ch, fname)
        if key in img_cache:
            return img_cache[key]
        gray = cv2.imread(str(spec_base / session / fname), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            img_cache[key] = None
            return None
        h, w = gray.shape
        disp = cv2.resize(gray, (args.disp_w, int(round(h * args.disp_w / w))),
                          interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_q])
        img_id = len(img_store)
        img_store.append({"b64": base64.b64encode(buf).decode(), "w": w, "h": h})
        img_cache[key] = img_id
        return img_id

    def load_method(pred_dir, session):
        """Per-channel annotations (with masks) + pooled (t0,t1,score) within coverage."""
        coverage = coverage_by_rec.get(session)
        per_ch, pooled = {}, []
        for ch in channels:
            p = Path(pred_dir) / session / f"coco_ch_{ch}.json"
            if not p.exists():
                continue
            coco = json.load(open(p))
            img_by_id = {im["id"]: im for im in coco["images"]}
            anns = []
            for ann in coco["annotations"]:
                im = img_by_id[ann["image_id"]]
                t0, t1 = bbox_to_time(ann["bbox"], im["window_start_sec"],
                                      im["window_end_sec"], im["width"])
                if coverage is not None and not (coverage[0] <= (t0 + t1) / 2 <= coverage[1]):
                    continue
                anns.append({"t0": t0, "t1": t1, "seg": ann.get("segmentation", []),
                             "bbox": ann["bbox"]})
                pooled.append((t0, t1, ann.get("score")))
            per_ch[ch] = anns
        return per_ch, pooled

    def load_rejected(pred_dir, session):
        """Per-channel stage-2-rejected ridge candidates, tagged with why they were cut.

        Diagnostic only: never scored, only overlaid on 'ridge missed' cards so you can
        tell "ridge's stage-1 filter never even saw this call" (no candidate here) apart
        from "ridge saw it but a stage-2 gate cut it" (candidate present, reason shown).
        """
        coverage = coverage_by_rec.get(session)
        per_ch = {}
        for ch in channels:
            p = Path(pred_dir) / session / f"coco_ch_{ch}.json"
            if not p.exists():
                continue
            coco = json.load(open(p))
            img_by_id = {im["id"]: im for im in coco["images"]}
            anns = []
            for ann in coco["annotations"]:
                im = img_by_id[ann["image_id"]]
                t0, t1 = bbox_to_time(ann["bbox"], im["window_start_sec"],
                                      im["window_end_sec"], im["width"])
                if coverage is not None and not (coverage[0] <= (t0 + t1) / 2 <= coverage[1]):
                    continue
                anns.append({"t0": t0, "t1": t1, "seg": ann.get("segmentation", []),
                             "bbox": ann["bbox"], "reason": ann.get("reject_reason", "?")})
            per_ch[ch] = anns
        return per_ch

    def chunk_for_time(session, ch, t):
        """(fname, ws, we) of the 1-second spectrogram chunk covering time t."""
        sec = int(np.floor(t))
        m = _glob.glob(str(spec_base / session / f"headmic_{ch}_*chunk_{sec:05d}_*.png"))
        if not m:
            allm = sorted(_glob.glob(str(spec_base / session / f"headmic_{ch}_*chunk_*.png")))
            if not allm:
                return None
            m = [allm[-1] if sec > 0 else allm[0]]
        return Path(m[0]).name, float(sec), float(sec + 1)

    def collect_channels(data, session, t0, t1):
        """Render payload per channel: image ref, focus band, per-method polygons."""
        mid = (t0 + t1) / 2
        out = []
        for ch in channels:
            info = chunk_for_time(session, ch, mid)
            if info is None:
                continue
            fname, ws, we = info
            img_id = get_img(session, ch, fname)
            if img_id is None:
                continue
            iw, ih = img_store[img_id]["w"], img_store[img_id]["h"]
            dur = we - ws

            def to_x(t):
                return max(0.0, min(1.0, (t - ws) / dur)) * iw

            meth_polys = {}
            for m in methods:
                polys = []
                for a in data[m]["per_ch"].get(ch, []):
                    if a["t1"] <= ws or a["t0"] >= we:      # outside this chunk window
                        continue
                    for seg in a["seg"]:
                        pts = [[seg[i], seg[i + 1]] for i in range(0, len(seg) - 1, 2)]
                        if len(pts) >= 3:
                            polys.append(pts)
                    if not a["seg"]:                         # box-only prediction
                        bx, by, bw, bh = a["bbox"]
                        polys.append([[bx, by], [bx + bw, by],
                                      [bx + bw, by + bh], [bx, by + bh]])
                meth_polys[m] = polys

            rej_polys = []
            for a in data.get("ridge_rejected", {}).get(ch, []):
                if a["t1"] <= ws or a["t0"] >= we:      # outside this chunk window
                    continue
                for seg in a["seg"]:
                    pts = [[seg[i], seg[i + 1]] for i in range(0, len(seg) - 1, 2)]
                    if len(pts) >= 3:
                        rej_polys.append({"pts": pts, "reason": a["reason"]})

            out.append({"ch": ch, "img": img_id, "iw": iw, "ih": ih,
                        "gt": [to_x(t0), to_x(t1)], "m": meth_polys, "rej": rej_polys})
        return out

    # ── analyze every recording ────────────────────────────────────────────────
    metrics = {m: {"tp": 0, "fn": 0, "pred_tp": 0, "fp": 0, "n_pred": 0} for m in methods}
    cards = []

    for session in sorted(gt_by_rec.keys()):
        gt = gt_by_rec[session]
        if not gt:
            continue
        data = {}
        for m, base in methods.items():
            per_ch, pooled = load_method(base, session)
            if not per_ch and not (base / session).exists():
                continue
            res = score_combined(gt, pooled, 0.0)
            data[m] = {"per_ch": per_ch, "pooled": pooled}
            for k in ("tp", "fn", "pred_tp", "fp", "n_pred"):
                metrics[m][k] += res[k]
        if "ridge" not in data:
            continue
        data["ridge_rejected"] = load_rejected(ridge_rejected_dir, session)

        def detects(m, gs, ge):
            return m in data and any(iou_1d(gs, ge, t0, t1) > 0
                                     for (t0, t1, _) in data[m]["pooled"])

        # ridge misses: GT events ridge did not overlap
        for (gs, ge) in gt:
            if detects("ridge", gs, ge):
                continue
            others = [m for m in ("sam3", "squeakout") if detects(m, gs, ge)]
            chans = collect_channels(data, session, gs, ge)
            cards.append({"session": session, "kind": "miss",
                          "cat": "miss_others_hit" if others else "miss_all",
                          "t0": gs, "t1": ge,
                          "hit": {m: detects(m, gs, ge) for m in methods},
                          "n_rejected": sum(len(ch["rej"]) for ch in chans),
                          "channels": chans})

        # ridge over-detections: merged ridge events overlapping no GT
        for (es, ee) in merge_intervals([(t0, t1) for (t0, t1, _) in data["ridge"]["pooled"]]):
            if any(iou_1d(es, ee, gs, ge) > 0 for gs, ge in gt):
                continue
            def fires(m):
                return m in data and any(iou_1d(es, ee, t0, t1) > 0
                                         for (t0, t1, _) in data[m]["pooled"])
            others = [m for m in ("sam3", "squeakout") if fires(m)]
            cards.append({"session": session, "kind": "fp",
                          "cat": "fp_shared" if others else "fp_only",
                          "t0": es, "t1": ee,
                          "fire": {m: fires(m) for m in methods},
                          "channels": collect_channels(data, session, es, ee)})

    cards = [c for c in cards if c["channels"]]
    print("cards:", len(cards), dict(Counter(c["cat"] for c in cards)))

    # ── cap non-ridge-specific categories (evenly sampled), keep ridge-specific ──
    full_counts = Counter(c["cat"] for c in cards)
    if not args.no_cap:
        by_cat = defaultdict(list)
        for c in cards:
            by_cat[c["cat"]].append(c)
        kept = []
        for cat, lst in by_cat.items():
            if cat in CAPPED_CATS and len(lst) > args.cap:
                kept.extend(lst[i] for i in sorted(random.sample(range(len(lst)), args.cap)))
            else:
                kept.extend(lst)
        cards = kept

    # prune image store to referenced images only, remap ids
    used = {}
    for c in cards:
        for ch in c["channels"]:
            used.setdefault(ch["img"], len(used))
    for c in cards:
        for ch in c["channels"]:
            ch["img"] = used[ch["img"]]
    pruned = [None] * len(used)
    for old, new in used.items():
        pruned[new] = img_store[old]
    img_store = pruned
    print(f"final: {len(cards)} cards, {len(img_store)} images")

    # ── metrics summary ─────────────────────────────────────────────────────────
    def slim_metrics(m):
        d = metrics[m]
        p = d["pred_tp"] / (d["pred_tp"] + d["fp"]) if (d["pred_tp"] + d["fp"]) else 0
        r = d["tp"] / (d["tp"] + d["fn"]) if (d["tp"] + d["fn"]) else 0
        f1 = 2 * p * r / (p + r) if (p + r) else 0
        return {"recall": r, "precision": p, "f1": f1, **d}

    for m in methods:
        s = slim_metrics(m)
        print(f"  {m:10s} R={s['recall']:.3f} P={s['precision']:.3f} F1={s['f1']:.3f}"
              f"  FN={s['fn']} FP={s['fp']}")

    payload = {
        "images": [{"b": im["b64"]} for im in img_store],
        "cards": cards,
        "metrics": {m: slim_metrics(m) for m in methods},
        "counts": dict(Counter(c["cat"] for c in cards)),
        "full_counts": dict(full_counts),
        "n_gt": sum(len(v) for v in gt_by_rec.values() if v),
    }
    data_json = json.dumps(payload, separators=(",", ":"))
    print("payload MB:", round(len(data_json) / 1e6, 1))

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(HTML_TEMPLATE.replace("__DATA__", data_json))
    print("wrote", out_html, round(out_html.stat().st_size / 1e6, 1), "MB")


HTML_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ridge detection failure analysis</title>
<style>
:root{--bg:#faf9f7;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;
  --line:#e6e4df;--ridge:#eb6834;--sam3:#2a78d6;--squeak:#8e5fd3;--gt:#00a000;--rej:#a8890a;}
@media(prefers-color-scheme:dark){:root{--bg:#151513;--card:#1f1f1d;--ink:#f5f4ef;
  --ink2:#c3c2b7;--muted:#8f8e86;--line:#33322e;--ridge:#f0824f;--sam3:#4f97ec;--squeak:#a985e6;--gt:#33c433;--rej:#e0b429;}}
:root[data-theme=light]{--bg:#faf9f7;--card:#fff;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--line:#e6e4df;--ridge:#eb6834;--sam3:#2a78d6;--squeak:#8e5fd3;--gt:#00a000;--rej:#a8890a;}
:root[data-theme=dark]{--bg:#151513;--card:#1f1f1d;--ink:#f5f4ef;--ink2:#c3c2b7;--muted:#8f8e86;--line:#33322e;--ridge:#f0824f;--sam3:#4f97ec;--squeak:#a985e6;--gt:#33c433;--rej:#e0b429;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;}
.wrap{max-width:1200px;margin:0 auto;padding:28px 20px 80px;}
h1{font-size:22px;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--ink2);margin:0 0 22px;max-width:70ch}
.metrics{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:26px}
.mcard{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;min-width:170px}
.mcard h3{margin:0 0 8px;font-size:13px;display:flex;align-items:center;gap:7px}
.dot{width:10px;height:10px;border-radius:3px;display:inline-block}
.mrow{display:flex;justify-content:space-between;font-variant-numeric:tabular-nums;color:var(--ink2)}
.mrow b{color:var(--ink);font-weight:600}
.bar{height:5px;border-radius:3px;background:var(--line);margin:2px 0 8px;overflow:hidden}
.bar>i{display:block;height:100%}
.controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:8px;position:sticky;top:0;
  background:var(--bg);padding:10px 0;z-index:5;border-bottom:1px solid var(--line)}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:9px;overflow:hidden;flex-wrap:wrap}
.seg button{border:0;background:var(--card);color:var(--ink2);padding:7px 12px;cursor:pointer;font-size:13px}
.seg button.on{background:var(--ink);color:var(--bg)}
.tog{display:inline-flex;gap:6px;align-items:center;margin-left:6px}
.chip{border:1px solid var(--line);background:var(--card);border-radius:20px;padding:5px 11px;cursor:pointer;
  font-size:12.5px;display:inline-flex;align-items:center;gap:6px;user-select:none}
.chip.off{opacity:.4}
.count{color:var(--muted);font-size:12.5px;margin-left:auto}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:16px;margin-top:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.card .hd{padding:9px 12px;border-bottom:1px solid var(--line);display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.tag{font-size:11px;font-weight:600;padding:2px 8px;border-radius:6px;letter-spacing:.02em}
.tag.miss{background:color-mix(in srgb,var(--ridge) 18%,transparent);color:var(--ridge)}
.tag.fp{background:color-mix(in srgb,var(--sam3) 16%,transparent);color:var(--sam3)}
.hd .sid{font-size:12px;color:var(--ink2);font-variant-numeric:tabular-nums}
.hd .note{font-size:11.5px;color:var(--muted);width:100%}
.chan{position:relative;line-height:0;border-top:1px solid var(--line)}
.chan:first-of-type{border-top:0}
.chan img{width:100%;display:block;image-rendering:auto}
.chan svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.chan .cl{position:absolute;top:4px;left:6px;font-size:10px;color:#fff;background:rgba(0,0,0,.55);
  padding:1px 5px;border-radius:4px;line-height:1.4}
.empty{padding:40px;text-align:center;color:var(--muted)}
.legend{font-size:12px;color:var(--ink2);display:flex;gap:14px;flex-wrap:wrap;margin:2px 0 0}
.legend span{display:inline-flex;align-items:center;gap:5px}
.swatch{width:14px;height:3px;border-radius:2px;display:inline-block}
footer{color:var(--muted);font-size:12px;margin-top:30px}
</style></head><body>
<div class="wrap">
<h1>Ridge detection — systematic misses &amp; over-detections</h1>
<p class="sub">Every ground-truth call ridge <b>missed</b> and every region ridge <b>over-detected</b>,
shown on the actual spectrogram with each method's segmentation mask. Scored at the recording level
by time-overlap against the sampled contiguous GT, within each recording's labeled coverage span.
Toggle methods to see whether SAM3 / SqueakOut succeeded where ridge failed.</p>
<div class="metrics" id="metrics"></div>
<div class="legend">
  <span><i class="swatch" style="background:var(--gt);border:1px dashed var(--gt)"></i>dashed band = the missed / over-detected region in focus</span>
  <span><i class="swatch" style="background:var(--ridge)"></i>ridge mask</span>
  <span><i class="swatch" style="background:var(--sam3)"></i>SAM3 mask</span>
  <span><i class="swatch" style="background:var(--squeak)"></i>SqueakOut mask</span>
  <span><i class="swatch" style="background:var(--rej);border:1px dashed var(--rej)"></i>ridge stage-1 candidate rejected by stage-2 (hover for reason)</span>
</div>
<div class="controls">
  <div class="seg" id="catseg"></div>
  <span class="tog">
    <span class="chip on" data-m="ridge"><i class="dot" style="background:var(--ridge)"></i>ridge</span>
    <span class="chip on" data-m="sam3"><i class="dot" style="background:var(--sam3)"></i>sam3</span>
    <span class="chip on" data-m="squeakout"><i class="dot" style="background:var(--squeak)"></i>squeakout</span>
    <span class="chip off" data-m="ridge_rejected"><i class="dot" style="background:var(--rej)"></i>ridge rejected</span>
  </span>
  <span class="count" id="count"></span>
</div>
<div class="grid" id="grid"></div>
<footer id="foot"></footer>
</div>
<script>
const DATA=__DATA__;
const MCOL={ridge:'var(--ridge)',sam3:'var(--sam3)',squeakout:'var(--squeak)'};
const CATS=[
 {k:'all',label:'All'},
 {k:'miss_others_hit',label:'Missed by ridge · others caught'},
 {k:'miss_all',label:'Missed by all'},
 {k:'fp_only',label:'Over-detected · ridge only'},
 {k:'fp_shared',label:'Over-detected · shared'},
];
let curCat='miss_others_hit';
const shown={ridge:true,sam3:true,squeakout:true,ridge_rejected:false};

function fmtPct(x){return (x*100).toFixed(1)+'%';}
function renderMetrics(){
  const el=document.getElementById('metrics');
  const cfg=[['ridge','ridge','var(--ridge)'],['sam3','SAM3 (best)','var(--sam3)'],['squeakout','SqueakOut','var(--squeak)']];
  el.innerHTML=cfg.map(([k,name,c])=>{
    const m=DATA.metrics[k];
    return `<div class="mcard"><h3><span class="dot" style="background:${c}"></span>${name}</h3>
      <div class="mrow"><span>Recall</span><b>${fmtPct(m.recall)}</b></div>
      <div class="bar"><i style="width:${m.recall*100}%;background:${c}"></i></div>
      <div class="mrow"><span>Precision</span><b>${fmtPct(m.precision)}</b></div>
      <div class="bar"><i style="width:${m.precision*100}%;background:${c}"></i></div>
      <div class="mrow"><span>F1</span><b>${fmtPct(m.f1)}</b></div>
      <div class="bar"><i style="width:${m.f1*100}%;background:${c}"></i></div>
      <div class="mrow" style="margin-top:6px;font-size:12px"><span>miss / over-det</span><b>${m.fn} / ${m.fp}</b></div>
    </div>`;
  }).join('');
}
function renderCatSeg(){
  const seg=document.getElementById('catseg');
  const counts=DATA.counts, full=DATA.full_counts||{};
  seg.innerHTML=CATS.map(c=>{
    let n = c.k==='all'?DATA.cards.length:(counts[c.k]||0);
    let tot = c.k==='all'?null:(full[c.k]||n);
    let label = (tot!=null && tot>n) ? `${c.label} <b>${n}</b><small style="opacity:.6"> /${tot}</small>` : `${c.label} <b>${n}</b>`;
    return `<button data-k="${c.k}" class="${c.k===curCat?'on':''}">${label}</button>`;
  }).join('');
  seg.querySelectorAll('button').forEach(b=>b.onclick=()=>{curCat=b.dataset.k;renderCatSeg();renderGrid();});
}
function poly(pts,color){
  const d=pts.map(p=>p[0].toFixed(1)+','+p[1].toFixed(1)).join(' ');
  return `<polygon points="${d}" fill="${color}" fill-opacity="0.22" stroke="${color}" stroke-width="1.4"/>`;
}
function rejPoly(pts,reason){
  const d=pts.map(p=>p[0].toFixed(1)+','+p[1].toFixed(1)).join(' ');
  return `<polygon points="${d}" fill="var(--rej)" fill-opacity="0.10" stroke="var(--rej)" stroke-width="1.2" stroke-dasharray="3 2"><title>ridge candidate rejected: ${reason}</title></polygon>`;
}
function cardHTML(c){
  const hit=c.hit||c.fire||{};
  let note='';
  if(c.kind==='miss'){
    const others=['sam3','squeakout'].filter(m=>hit[m]);
    note = others.length? `ridge missed — caught by ${others.join(' & ')}` : 'missed by every method (hard call)';
    if(c.n_rejected) note += ` · ${c.n_rejected} rejected ridge candidate${c.n_rejected!==1?'s':''} nearby (stage-1 saw something, stage-2 cut it)`;
    else note += ' · no ridge stage-1 candidate here at all (filter never responded)';
  }else{
    const others=['sam3','squeakout'].filter(m=>hit[m]);
    note = others.length? `ridge fired here — so did ${others.join(' & ')}` : 'ridge fired here — SAM3 & SqueakOut did not';
  }
  const chans=c.channels.map(ch=>{
    const im=DATA.images[ch.img];
    let svg=`<svg viewBox="0 0 ${ch.iw} ${ch.ih}" preserveAspectRatio="none">`;
    const bandc = c.kind==='miss' ? 'var(--gt)' : 'var(--ridge)';
    if(ch.gt){const bw=Math.max(2,ch.gt[1]-ch.gt[0]).toFixed(1);
      svg+=`<rect x="${ch.gt[0].toFixed(1)}" y="0" width="${bw}" height="${ch.ih}" fill="${bandc}" fill-opacity="0.14"/>`;
      svg+=`<rect x="${ch.gt[0].toFixed(1)}" y="0" width="${bw}" height="${ch.ih}" fill="none" stroke="${bandc}" stroke-width="1.6" stroke-dasharray="4 3"/>`;}
    for(const m of ['ridge','sam3','squeakout']){
      if(!shown[m])continue;
      for(const pts of (ch.m[m]||[]))svg+=poly(pts,MCOL[m]);
    }
    if(shown.ridge_rejected){
      for(const r of (ch.rej||[]))svg+=rejPoly(r.pts,r.reason);
    }
    svg+='</svg>';
    return `<div class="chan"><img loading="lazy" src="data:image/jpeg;base64,${im.b}"><span class="cl">ch ${ch.ch}</span>${svg}</div>`;
  }).join('');
  const tagcls=c.kind==='miss'?'miss':'fp';
  const tagtxt=c.kind==='miss'?'MISS':'OVER-DET';
  return `<div class="card"><div class="hd">
     <span class="tag ${tagcls}">${tagtxt}</span>
     <span class="sid">${c.session}  ·  t=${c.t0.toFixed(2)}–${c.t1.toFixed(2)}s</span>
     <span class="note">${note}</span></div>${chans}</div>`;
}
function renderGrid(){
  const grid=document.getElementById('grid');
  let list = curCat==='all'?DATA.cards:DATA.cards.filter(c=>c.cat===curCat);
  document.getElementById('count').textContent=`${list.length} example${list.length!==1?'s':''}`;
  if(!list.length){grid.innerHTML='<div class="empty">No examples in this category.</div>';return;}
  grid.innerHTML=list.map(cardHTML).join('');
}
document.querySelectorAll('.chip').forEach(ch=>ch.onclick=()=>{
  const m=ch.dataset.m;shown[m]=!shown[m];ch.classList.toggle('off',!shown[m]);renderGrid();
});
document.getElementById('foot').textContent=
 `${DATA.n_gt} ground-truth calls · ${DATA.cards.length} ridge failure windows · masks are per-method COCO segmentation polygons on 1-second spectrogram windows.`;
renderMetrics();renderCatSeg();renderGrid();
</script></body></html>
"""


if __name__ == "__main__":
    main()

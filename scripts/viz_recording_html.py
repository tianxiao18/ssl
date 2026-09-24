"""
Self-contained HTML browser for a recording's SAM3 segmentation masks and GT.

One page per recording: the whole recording in time order as contiguous
spectrogram strips (--cols consecutive 1-second windows cut from the HDF5 as a
single image, so a call spanning a window boundary is not sliced in two), with
its COCO segmentation polygons overlaid as SVG and the ground-truth
vocalization spans drawn as dashed bands. Pages are flipped with the arrow
keys; a dropdown jumps to any other recording generated in the same run, so a
batch of pages browses as one tool.

This is a *browser*, not a failure triage -- it shows every window, not just
the interesting ones. (scripts/archive/ridge_failure_viz.py is the triage view:
cards for missed / over-detected events only.)

The windows come from the prediction COCO's own `images` entries, so masks are
guaranteed to line up with the frame they are drawn on: each mask's x extent is
converted from its own window's pixels to seconds and then to the strip's own
time axis (`t` in the HDF5), and the SVG is stretched over the strip image.

Mask color is the event-level verdict from vox_tracer.scoring.score_combined --
the same scorer scripts/evaluate.py uses -- so what you see agrees with
metrics.csv: cyan for a mask matching a GT call (true positive), orange for one
that matches none (false positive). Toggle it off for flat coloring.

Size: ~4.3 KB per window at the defaults, so ~5 MB for a 20-min cohort2
recording and ~15 MB for a 60-min cohort4 one. Lower --disp-w / --jpeg-q for
smaller files. Generating all 1881 recordings would be ~15 GB, hence --limit.

Usage
-----
    python scripts/viz_recording_html.py \
        outputs/spectrograms_h5/dryad_gerbil_full \
        outputs/sam3_h5/dryad_gerbil_full \
        data/dryad_gerbil_full \
        outputs/eval/html \
        --channels 0 --prefix mic --limit 3

    # specific recordings
    python scripts/viz_recording_html.py ... \
        --recordings cohort2_formatted/2020_07_19_16_32_55_857727_merged

    # unannotated corpus: detections only, no GT bands / verdict coloring
    python scripts/viz_recording_html.py \
        outputs/spectrograms_h5/mongolia_wild_data \
        outputs/sam3_mongolia/mongolia_wild_data \
        data/mongolia_wild_data \
        outputs/eval/html_mongolia --no-gt --limit 0
"""
import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vox_tracer.scoring import bbox_to_time, iou_1d, load_gt_from_csv, score_combined

# Width of a strip's overlay coordinate system. Strips are cut at slightly
# different pixel widths (the columns per second are not integral), so masks
# are carried in a fixed unit space and the SVG is stretched to the image.
UNITS = 1000.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("spec_base",      help="base of per-recording HDF5 spectrograms")
    p.add_argument("pred_base",      help="base of per-recording coco_ch_<ch>.json")
    p.add_argument("recording_base", help="base of per-recording *annotations_gt.csv")
    p.add_argument("out_dir")
    p.add_argument("--channels", default="0")
    p.add_argument("--prefix",   default="mic", help="HDF5 filename prefix: <prefix>_<ch>_*.h5")
    p.add_argument("--recordings", nargs="*", default=None,
                   help="explicit relative recording paths; default = discover under pred_base")
    p.add_argument("--limit", type=int, default=3,
                   help="max recordings when discovering (guards against a 15 GB run)")
    p.add_argument("--disp-w",  type=int, default=440, help="encoded window width in px")
    p.add_argument("--jpeg-q",  type=int, default=46)
    p.add_argument("--cols",    type=int, default=6,
                   help="windows stitched into one contiguous strip row")
    p.add_argument("--rows",    type=int, default=5, help="strip rows per page")
    p.add_argument("--iou-threshold", type=float, default=0.0)
    p.add_argument("--no-gt", action="store_true",
                   help="dataset has no *annotations_gt.csv: draw detections only, with no "
                        "GT bands, no TP/FP verdict and no recall/precision")
    p.add_argument("--labels", default=None,
                   help="JSON file mapping a relative recording path to a short group "
                        "label; adds a group column to index.html (e.g. which selection "
                        "bucket the recording was drawn from)")
    return p.parse_args()


def discover(pred_base, channels, limit):
    """Relative paths of recordings that actually have predictions."""
    hits = sorted({p.parent.relative_to(pred_base)
                   for ch in channels
                   for p in Path(pred_base).glob(f"*/*/coco_ch_{ch}.json")})
    if not hits:   # flat layout
        hits = sorted({p.parent.relative_to(pred_base)
                       for ch in channels
                       for p in Path(pred_base).glob(f"*/coco_ch_{ch}.json")})
    return hits[:limit] if limit else hits


def build_recording(rel, args, channels):
    """-> dict of page data for one recording, or None if it can't be built."""
    spec_dir = Path(args.spec_base) / rel
    pred_dir = Path(args.pred_base) / rel
    rec_dir  = Path(args.recording_base) / rel

    if args.no_gt:
        vox_gt = []
    else:
        try:
            vox_gt, _ = load_gt_from_csv(rec_dir)
        except FileNotFoundError:
            print(f"  {rel}: no *annotations_gt.csv, skipping")
            return None

    # Predictions first: their COCO `images` define the window grid we render.
    per_ch = {}
    pooled = []
    for ch in channels:
        cp = pred_dir / f"coco_ch_{ch}.json"
        if not cp.exists():
            continue
        coco = json.loads(cp.read_text())
        img_by_id = {im["id"]: im for im in coco["images"]}
        anns = []
        for a in coco["annotations"]:
            im = img_by_id[a["image_id"]]
            t0, t1 = bbox_to_time(a["bbox"], im["window_start_sec"],
                                  im["window_end_sec"], im["width"])
            anns.append({"img": a["image_id"], "t0": t0, "t1": t1,
                         "seg": a.get("segmentation", []), "bbox": a["bbox"]})
            pooled.append((t0, t1, a.get("score")))
        per_ch[ch] = {"images": coco["images"], "img_by_id": img_by_id, "anns": anns}
    if not per_ch:
        print(f"  {rel}: no predictions, skipping")
        return None

    # Event-level verdicts, identical to scripts/evaluate.py. Without GT there is
    # nothing to score against, so every mask is drawn in the flat color.
    if args.no_gt:
        res = {"recall": None, "precision": None}
        tp_events = []
    else:
        res = score_combined(vox_gt, pooled, args.iou_threshold)
        tp_events = res["tp_events"]

    def is_tp(t0, t1):
        return any(iou_1d(t0, t1, s, e) > 0 for s, e in tp_events)

    pages_ch = {}
    win_ch = {}
    for ch, d in per_ch.items():
        h5s = sorted(spec_dir.glob(f"{args.prefix}_{ch}_*.h5"))
        if not h5s:
            print(f"  {rel} ch{ch}: no {args.prefix}_{ch}_*.h5 in {spec_dir}, skipping channel")
            continue

        anns_by_img = {}
        for a in d["anns"]:
            anns_by_img.setdefault(a["img"], []).append(a)

        windows = sorted(d["images"], key=lambda im: im["window_start_sec"])
        # Windows tile the recording without gaps, so --cols of them in a row are
        # cut from the spectrogram as one strip instead of --cols separate images:
        # a call straddling a window boundary stays whole. Masks keep their own
        # window's time span and are mapped into the strip's x axis below.
        strips = [windows[i:i + args.cols] for i in range(0, len(windows), args.cols)]
        cells = []
        n_win = 0
        # One open handle for the whole recording: read_h5_window() reopens the
        # file per call, which is ~1200 reopens for a single cohort2 page.
        with h5py.File(h5s[0], "r") as f:
            t_axis = f["t"][:]
            spec = f["spec"]
            for grp in strips:
                ws, we = grp[0]["window_start_sec"], grp[-1]["window_end_sec"]
                c0, c1 = np.searchsorted(t_axis, [ws, we])
                if c1 - c0 < 2:
                    continue
                gray = spec[:, c0:c1]
                # The strip's pixels span the sampled columns, not the nominal
                # window edges, so overlays are placed against the axis actually
                # drawn -- otherwise they drift by up to a bin at the seams.
                lo, hi = float(t_axis[c0]), float(t_axis[c1 - 1])
                if hi <= lo:
                    continue
                sw = args.disp_w * len(grp)
                dh = int(round(gray.shape[0] * sw / gray.shape[1]))
                disp = cv2.resize(gray, (sw, dh), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_q])
                if not ok:
                    continue
                n_win += len(grp)

                # Overlay coords are strip-relative: x in UNITS, y in COCO rows.
                H = grp[0]["height"]

                def x_of(t, lo=lo, hi=hi):
                    return (t - lo) / (hi - lo) * UNITS

                gt_bands = []
                for g0, g1 in vox_gt:
                    if g1 > lo and g0 < hi:
                        x0 = max(0.0, x_of(max(g0, lo)))
                        x1 = min(UNITS, x_of(min(g1, hi)))
                        if x1 > x0:
                            gt_bands.append([round(x0, 1), round(x1, 1)])

                polys, boxes = [], []
                for im in grp:
                    iw = im["width"]
                    idur = im["window_end_sec"] - im["window_start_sec"]
                    ysc = H / im["height"]

                    def t_of(px, s=im["window_start_sec"], iw=iw, idur=idur):
                        return s + px / iw * idur

                    for a in anns_by_img.get(im["id"], []):
                        v = 1 if is_tp(a["t0"], a["t1"]) else 0
                        if a["seg"]:
                            for s in a["seg"]:
                                pts = []
                                for i in range(0, len(s) - 1, 2):
                                    pts.append(round(x_of(t_of(s[i])), 1))
                                    pts.append(round(s[i + 1] * ysc, 1))
                                polys.append([v, pts])
                        else:
                            x, y, w, h = a["bbox"]
                            x0, x1 = x_of(t_of(x)), x_of(t_of(x + w))
                            boxes.append([v, round(x0, 1), round(y * ysc, 1),
                                          round(x1 - x0, 1), round(h * ysc, 1)])

                cells.append({"b": base64.b64encode(buf).decode(), "w": UNITS, "h": H,
                              "t0": round(ws, 2), "t1": round(we, 2),
                              "gt": gt_bands, "p": polys, "bx": boxes})
        pages_ch[str(ch)] = cells
        win_ch[str(ch)] = n_win

    if not pages_ch:
        return None
    n_masks = sum(len(c["p"]) + len(c["bx"]) for cs in pages_ch.values() for c in cs)
    return {"session": str(rel), "channels": pages_ch, "n_gt": len(vox_gt),
            "has_gt": not args.no_gt,
            "n_win": sum(win_ch.values()), "windows": win_ch,
            "n_masks": n_masks, "recall": res["recall"], "precision": res["precision"]}


HTML_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>__TITLE__</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--line:#2a2f3a;--fg:#e6e8ee;--dim:#8b93a5;
      --tp:#22d3ee;--fp:#fb923c;--gt:#f472b6;--flat:#22d3ee}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:13px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:10;background:var(--panel);
       border-bottom:1px solid var(--line);padding:10px 14px;display:flex;
       flex-wrap:wrap;gap:10px 16px;align-items:center}
select,input,button{background:#0f1115;color:var(--fg);border:1px solid var(--line);
                    border-radius:6px;padding:5px 8px;font:inherit}
button{cursor:pointer}button:hover{border-color:#3d4553}
button:disabled{opacity:.35;cursor:default}
.stats{color:var(--dim)}
.stats b{color:var(--fg);font-weight:600}
.chip{cursor:pointer;user-select:none;border:1px solid var(--line);border-radius:999px;
      padding:4px 11px;display:inline-flex;align-items:center;gap:6px}
.chip.off{opacity:.4}
.sw{width:9px;height:9px;border-radius:2px;display:inline-block}
.nav{margin-left:auto;display:flex;align-items:center;gap:8px}
#grid{display:flex;flex-direction:column;gap:10px;padding:14px}
.cell{background:var(--panel);border:1px solid var(--line);border-radius:8px;
      overflow:hidden;position:relative}
.cell.hasgt{border-color:var(--gt)}
.frame{position:relative;line-height:0}
.frame img{width:100%;display:block;image-rendering:auto}
.frame svg{position:absolute;inset:0;width:100%;height:100%}
.cap{padding:4px 7px;color:var(--dim);font-size:11px;
     display:flex;justify-content:space-between;gap:6px}
.cap .n{color:var(--fg)}
.empty{padding:40px;text-align:center;color:var(--dim)}
kbd{background:#0f1115;border:1px solid var(--line);border-radius:4px;padding:1px 5px;font-size:11px}
</style></head><body>
<header>
  <select id="rec" title="recording"></select>
  <select id="ch" title="channel"></select>
  <span class="stats" id="stats"></span>
  <span class="chip" data-t="masks"><span class="sw" style="background:var(--flat)"></span>masks</span>
  <span class="chip" data-t="gt"><span class="sw" style="background:var(--gt)"></span>GT</span>
  <span class="chip" data-t="verdict" title="cyan = mask matched a GT call (TP); orange = matched none (FP)"><span class="sw" style="background:var(--tp)"></span><span class="sw" style="background:var(--fp)"></span>verdict: TP / FP</span>
  <span class="nav">
    <input id="jump" size="7" placeholder="t=sec" title="jump to time (seconds)">
    <button id="prev">←</button>
    <span id="pageno" class="stats"></span>
    <button id="next">→</button>
  </span>
</header>
<div id="grid"></div>
<script>
const DATA=__DATA__;
const HASGT=DATA.stats.has_gt;
const shown={masks:true,gt:HASGT,verdict:HASGT};
let ch=Object.keys(DATA.channels)[0], page=0;
const PER=DATA.rows;
const cells=()=>DATA.channels[ch];
const nPages=()=>Math.max(1,Math.ceil(cells().length/PER));

function poly(flat,color){
  let d='';for(let i=0;i+1<flat.length;i+=2)d+=flat[i]+','+flat[i+1]+' ';
  return `<polygon points="${d.trim()}" fill="${color}" fill-opacity="0.22" stroke="${color}" stroke-width="1.4" vector-effect="non-scaling-stroke"/>`;
}
function cellHTML(c,idx){
  let svg=`<svg viewBox="0 0 ${c.w} ${c.h}" preserveAspectRatio="none">`;
  if(shown.gt) for(const [x0,x1] of c.gt){
    const w=Math.max(0.8,x1-x0);
    svg+=`<rect x="${x0}" y="0" width="${w}" height="${c.h}" fill="var(--gt)" fill-opacity="0.13"/>`;
    svg+=`<rect x="${x0}" y="0" width="${w}" height="${c.h}" fill="none" stroke="var(--gt)" stroke-width="1.6" stroke-dasharray="4 3" vector-effect="non-scaling-stroke"/>`;
  }
  if(shown.masks){
    const col=v=>shown.verdict?(v?'var(--tp)':'var(--fp)'):'var(--flat)';
    for(const [v,pts] of c.p) svg+=poly(pts,col(v));
    for(const [v,x,y,w,h] of c.bx)
      svg+=`<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="none" stroke="${col(v)}" stroke-width="1.6" vector-effect="non-scaling-stroke"/>`;
  }
  svg+='</svg>';
  const n=c.p.length+c.bx.length;
  return `<div class="cell${c.gt.length?' hasgt':''}">
    <div class="frame"><img loading="lazy" src="data:image/jpeg;base64,${c.b}">${svg}</div>
    <div class="cap"><span>${c.t0.toFixed(1)}–${c.t1.toFixed(1)}s</span>
      <span class="n">${n?n+' mask'+(n!==1?'s':''):''}</span></div></div>`;
}
function render(){
  const all=cells(), grid=document.getElementById('grid');
  page=Math.min(Math.max(0,page),nPages()-1);
  const slice=all.slice(page*PER,(page+1)*PER);
  grid.innerHTML=slice.length?slice.map(cellHTML).join(''):'<div class="empty">No windows.</div>';
  const t0=slice.length?slice[0].t0.toFixed(0):'0', t1=slice.length?slice[slice.length-1].t1.toFixed(0):'0';
  document.getElementById('pageno').textContent=`page ${page+1} / ${nPages()}  ·  t=${t0}–${t1}s`;
  document.getElementById('prev').disabled=page===0;
  document.getElementById('next').disabled=page>=nPages()-1;
  const s=DATA.stats;
  document.getElementById('stats').innerHTML=
    `<b>${DATA.windows[ch]}</b> windows in <b>${all.length}</b> strips · `+
    `<b>${s.n_masks}</b> masks`+
    (HASGT?` · <b>${s.n_gt}</b> GT calls · recall <b>${s.recall.toFixed(3)}</b>`+
           ` · prec <b>${s.precision.toFixed(3)}</b>`:' · no ground truth');
  window.scrollTo(0,0);
}
const rec=document.getElementById('rec');
rec.innerHTML=DATA.manifest.map(m=>`<option value="${m.file}"${m.session===DATA.session?' selected':''}>${m.session}</option>`).join('');
rec.onchange=()=>{location.href=rec.value;};
const chSel=document.getElementById('ch');
chSel.innerHTML=Object.keys(DATA.channels).map(c=>`<option value="${c}">ch ${c}</option>`).join('');
chSel.style.display=Object.keys(DATA.channels).length>1?'':'none';
chSel.onchange=()=>{ch=chSel.value;page=0;render();};
document.querySelectorAll('.chip').forEach(el=>{
  const t=el.dataset.t;
  if(!HASGT&&(t==='gt'||t==='verdict')){el.style.display='none';return;}
  el.onclick=()=>{shown[t]=!shown[t];el.classList.toggle('off',!shown[t]);render();};
});
document.getElementById('prev').onclick=()=>{page--;render();};
document.getElementById('next').onclick=()=>{page++;render();};
document.getElementById('jump').onchange=e=>{
  const t=parseFloat(e.target.value);if(isNaN(t))return;
  const i=cells().findIndex(c=>c.t1>t);
  if(i>=0){page=Math.floor(i/PER);render();}
};
addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;
  if(e.key==='ArrowRight'||e.key===' '){page++;render();e.preventDefault();}
  else if(e.key==='ArrowLeft'){page--;render();}
  else if(e.key==='Home'){page=0;render();}
  else if(e.key==='End'){page=nPages()-1;render();}
});
render();
</script></body></html>"""


INDEX_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>Recording browser index</title>
<style>
body{margin:0;background:#0f1115;color:#e6e8ee;
     font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;padding:28px}
h1{font-size:17px;margin:0 0 4px}p{color:#8b93a5;margin:0 0 20px}
table{border-collapse:collapse;width:100%;max-width:900px}
th,td{text-align:left;padding:8px 12px;border-bottom:1px solid #2a2f3a}
th{color:#8b93a5;font-weight:600;font-size:12px}
td.num{text-align:right;font-variant-numeric:tabular-nums}
a{color:#22d3ee;text-decoration:none}a:hover{text-decoration:underline}
</style></head><body>
<h1>SAM3 segmentation browser</h1>
<p>__SUB__</p>
<table><tr><th>recording</th>__LABELHDR__<th class="num">windows</th>__GTHDR__
<th class="num">masks</th>__METRICHDR__</tr>
__ROWS__</table></body></html>"""


def main():
    args = parse_args()
    channels = [int(c) for c in args.channels.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = json.loads(Path(args.labels).read_text()) if args.labels else {}

    rels = ([Path(r) for r in args.recordings] if args.recordings
            else discover(Path(args.pred_base), channels, args.limit))
    if not rels:
        print("no recordings found")
        return
    print(f"building {len(rels)} recording page(s) → {out_dir}")

    built = []
    for rel in rels:
        print(f"  {rel} …", flush=True)
        d = build_recording(Path(rel), args, channels)
        if d:
            d["file"] = str(rel).replace("/", "__") + ".html"
            built.append(d)
    if not built:
        print("nothing built")
        return

    manifest = [{"session": d["session"], "file": d["file"]} for d in built]
    for d in built:
        payload = {"session": d["session"], "channels": d["channels"],
                   "windows": d["windows"],
                   "manifest": manifest, "cols": args.cols, "rows": args.rows,
                   "stats": {"n_gt": d["n_gt"], "n_masks": d["n_masks"],
                             "recall": d["recall"], "precision": d["precision"],
                             "has_gt": d["has_gt"]}}
        html = (HTML_TEMPLATE.replace("__TITLE__", d["session"])
                             .replace("__DATA__", json.dumps(payload, separators=(",", ":"))))
        p = out_dir / d["file"]
        p.write_text(html)
        print(f"  wrote {p}  ({p.stat().st_size / 1e6:.1f} MB)")

    has_gt = not args.no_gt
    rows = "\n".join(
        f'<tr><td><a href="{d["file"]}">{d["session"]}</a></td>'
        + (f'<td>{labels.get(d["session"], "")}</td>' if labels else "")
        + f'<td class="num">{d["n_win"]}</td>'
        + (f'<td class="num">{d["n_gt"]}</td>' if has_gt else "")
        + f'<td class="num">{d["n_masks"]}</td>'
        + (f'<td class="num">{d["recall"]:.3f}</td>'
           f'<td class="num">{d["precision"]:.3f}</td>' if has_gt else "")
        + '</tr>'
        for d in built)
    sub = (f"{len(built)} recordings · masks and ground truth overlaid on every window"
           if has_gt else
           f"{len(built)} recordings · masks overlaid on every window · no ground truth")
    idx = out_dir / "index.html"
    idx.write_text(INDEX_TEMPLATE.replace("__SUB__", sub)
                                 .replace("__LABELHDR__", "<th>group</th>" if labels else "")
                                 .replace("__GTHDR__", '<th class="num">GT calls</th>' if has_gt else "")
                                 .replace("__METRICHDR__",
                                          '<th class="num">recall</th><th class="num">precision</th>'
                                          if has_gt else "")
                                 .replace("__ROWS__", rows))
    print(f"  index → {idx}")


if __name__ == "__main__":
    main()

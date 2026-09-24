"""Launch the blind labeling GUI.

With no arguments the browser opens a setup page: pick the dataset, resume a campaign
that is part way through, or freeze a new one over an existing candidate pool with the
checkpoint grid entered there.

    python scripts/label_gui.py --annotator tianxiao

Naming a campaign skips setup and binds it straight away, as before:

    python scripts/label_gui.py --campaign outputs/label_campaigns/gerbil_ssl_k4 \
        --annotator tianxiao

Then open the printed URL. If you are on a workstation over SSH, forward the port
first:  ssh -L 8765:localhost:8765 <workstation>

Any --pad / --disp-w / band flags given here are the setup page's starting values; a
campaign remembers what it was last labeled with in view.json beside its spec.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vox_label.server import Session, serve


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", default=None,
                    help="campaign directory to bind at startup; omit to choose one "
                         "on the setup page")
    ap.add_argument("--annotator", default="anon")
    ap.add_argument("--dataset", default=None,
                    help="corpus the campaign annotates; default is the one recorded "
                         "in its spec")
    ap.add_argument("--backend", default=None, choices=["auto", "png", "h5"])
    ap.add_argument("--prefix", default=None,
                    help="chunk-PNG stream prefix (headmic, mic, ...); detected from "
                         "the spectrograms when not given")
    ap.add_argument("--pad", type=float, default=None,
                    help="seconds of context shown either side of the candidate")
    ap.add_argument("--disp-w", type=int, default=None)
    ap.add_argument("--jpeg-q", type=int, default=None)
    ap.add_argument("--reveal-every", type=int, default=None,
                    help="annotations between checkpoint reveals")
    ap.add_argument("--f-lo-khz", type=float, default=None,
                    help="bottom of the displayed frequency band")
    ap.add_argument("--f-hi-khz", type=float, default=None,
                    help="top of the displayed frequency band; narrow it (e.g. 15-45) "
                         "to make each panel shorter and the calls larger")
    ap.add_argument("--nyquist-khz", type=float, default=None,
                    help="sr/2 of the source audio; read from the data when not given")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    view = {k: getattr(args, k) for k in
            ("pad", "disp_w", "jpeg_q", "reveal_every", "f_lo_khz", "f_hi_khz",
             "nyquist_khz", "prefix", "backend")}
    session = Session(annotator=args.annotator,
                      view_overrides={k: v for k, v in view.items() if v is not None})
    if args.campaign:
        session.bind(args.campaign, dataset=args.dataset)
    serve(session, args.host, args.port)


if __name__ == "__main__":
    main()

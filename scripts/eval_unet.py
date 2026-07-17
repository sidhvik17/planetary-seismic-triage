"""Evaluate a trained spectrogram U-Net on the frozen benchmark protocol.

Usage:
  python scripts/eval_unet.py --model runs/unet_lunar/best.pt --eval-body lunar

Protocol (matches run_eval.py):
  * operating point — mask-energy threshold AND minimum event duration —
    tuned on the eval body's VAL continuous traces only (mars_ext has no
    val: tuned on train, stated in the output)
  * metrics on TEST continuous traces, same scorer and ±120 s tolerance
The model forward runs once per trace (cached curve); the (threshold,
duration) grid is swept on cached curves. Writes
results/{run_name}_to_{body}.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import CODA_SEC, DEFAULT as CFG, DATA_CACHE, PROJECT_ROOT
from planetseis.detect_spec import (SEC_PER_BIN, compute_curve,
                                    curve_to_detections, refine_arrivals)
from planetseis.evaluate import Scores, score_trace
from planetseis.unet import UNET_ARCHS, SpecUNet

THR_GRID = [0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
            0.50, 0.60, 0.70]
# Minimum event durations (s). Lunar events ring for tens of minutes; the
# packet's martian events last minutes. 30 s = effectively no gate.
DUR_GRID = [30, 120, 240, 430, 600]


def curves_for_split(model, body, split, device):
    out = []
    d = DATA_CACHE / body / "continuous" / split
    for p in sorted(d.glob("*.npz")):
        z = np.load(p)
        curve = compute_curve(model, z["trace"], device)
        out.append((p.stem, curve, list(z["picks"]), z["trace"]))
    return out


def score_at(curves, thr, min_dur_sec, coda, tol,
             refine_with=None, device="cpu"):
    total = Scores()
    min_bins = max(1, int(round(min_dur_sec / SEC_PER_BIN)))
    for _, curve, picks, trace in curves:
        dets = curve_to_detections(curve, thr, suppress_sec=coda,
                                   min_bins=min_bins)
        if refine_with is not None:
            dets = refine_arrivals(refine_with, trace, dets, 6.625, device)
        total.merge(score_trace([d.time_sec for d in dets], picks, tol))
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval-body", required=True,
                    choices=["lunar", "mars", "mars_ext"])
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--min-dur", type=float, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    model = SpecUNet(base=UNET_ARCHS[ckpt.get("arch", "base")])
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    coda = CODA_SEC[args.eval_body]
    tol = CFG.window.match_tolerance_sec
    has_val = any((DATA_CACHE / args.eval_body / "continuous" / "val").glob("*.npz"))
    tune_split = None

    if args.threshold is not None and args.min_dur is not None:
        best_thr, best_dur = args.threshold, args.min_dur
        print(f"fixed operating point thr={best_thr} min_dur={best_dur}s")
    else:
        tune_split = "val" if has_val else "train"
        print(f"computing {tune_split} curves ...")
        curves = curves_for_split(model, args.eval_body, tune_split, device)
        results = []
        for dur in DUR_GRID:
            for thr in THR_GRID:
                s = score_at(curves, thr, dur, coda, tol)
                results.append((thr, dur, s.f1))
        best_f1 = max(f1 for _, _, f1 in results)
        tied = [(t, d) for t, d, f1 in results if f1 >= best_f1 - 0.02]
        # plateau center on threshold within the near-tie set
        tied.sort()
        best_thr, best_dur = tied[len(tied) // 2]
        for dur in DUR_GRID:
            row = [f"{f1:.2f}" for t, d, f1 in results if d == dur]
            print(f"  dur={dur:4.0f}s F1@thr: {row}")
        print(f"selected thr={best_thr} min_dur={best_dur}s "
              f"(val F1={best_f1:.3f} plateau of {len(tied)})")

    print("computing test curves ...")
    test_curves = curves_for_split(model, args.eval_body, "test", device)
    s = score_at(test_curves, best_thr, best_dur, coda, tol)
    s_ref = score_at(test_curves, best_thr, best_dur, coda, tol,
                     refine_with=model, device=device)
    print(f"refined arrivals: MAE {s.mae_sec:.1f}s -> {s_ref.mae_sec:.1f}s, "
          f"F1 {s.f1:.3f} -> {s_ref.f1:.3f}")

    run_name = Path(args.model).parent.name
    result = {
        "model": run_name,
        "eval_body": args.eval_body,
        "threshold": best_thr,
        "min_dur_sec": best_dur,
        "threshold_tuned_on": tune_split or "fixed",
        "tolerance_sec": tol,
        "unet": s.as_dict(),
        "unet_refined_arrivals": s_ref.as_dict(),
    }
    out_path = PROJECT_ROOT / "results" / f"{run_name}_to_{args.eval_body}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()

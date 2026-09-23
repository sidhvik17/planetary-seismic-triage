"""Evaluate a trained checkpoint (and STA/LTA baseline) on continuous traces.

Usage:
  python scripts/run_eval.py --model runs/lunar/best.pt --eval-body lunar
  python scripts/run_eval.py --model runs/lunar/best.pt --eval-body mars   # transfer

Protocol (FR-9/FR-10, F-4, F-6):
  * threshold tuned on the eval body's VAL continuous traces (best F1)
  * metrics reported on TEST continuous traces
  * same scorer + tolerance for CNN and STA/LTA
Writes results/{model_body}_to_{eval_body}.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.baseline import sta_lta_detect
from planetseis.config import CODA_SEC, DEFAULT as CFG, DATA_CACHE, PROJECT_ROOT
from planetseis.detect import detect_events
from planetseis.evaluate import Scores, score_trace
from planetseis.model import ARCHS, SeisCNN


def load_continuous(body: str, split: str):
    d = DATA_CACHE / body / "continuous" / split
    for p in sorted(d.glob("*.npz")):
        z = np.load(p)
        yield p.stem, z["trace"], float(z["rate"]), list(z["picks"])


def eval_cnn(model, body, split, threshold, device):
    total = Scores()
    for _, trace, rate, picks in load_continuous(body, split):
        dets, _, _ = detect_events(model, trace, rate, CFG, threshold, device,
                                   suppress_sec=CODA_SEC[body])
        total.merge(score_trace([d.time_sec for d in dets], picks,
                                CFG.window.match_tolerance_sec))
    return total


def eval_stalta(body, split, thr_on):
    total = Scores()
    for _, trace, rate, picks in load_continuous(body, split):
        dets = sta_lta_detect(trace, rate, thr_on=thr_on, thr_off=max(1.2, thr_on / 2))
        # same dead-time rule as the CNN so the comparison is fair
        kept = []
        for t in sorted(dets):
            if not kept or t - kept[-1] >= CODA_SEC[body]:
                kept.append(t)
        total.merge(score_trace(kept, picks, CFG.window.match_tolerance_sec))
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval-body", required=True,
                    choices=["lunar", "mars", "lunar_dn"])
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--threshold", type=float, default=None,
                    help="skip val sweep, use this threshold (transfer protocol: "
                         "reuse the threshold tuned on the model's own body)")
    ap.add_argument("--eval-split", choices=["test", "all"], default="test",
                    help="'all' = every labeled file of the eval body; only valid "
                         "for cross-body transfer (model never trained on them)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.model, map_location=device, weights_only=True)
    model = SeisCNN(channels=ARCHS[ckpt.get("arch", "base")])
    model.load_state_dict(ckpt["model"])
    model_body = ckpt.get("config", {}).get("body", Path(args.model).parent.name)

    if args.eval_split == "all" and model_body == args.eval_body:
        ap.error("--eval-split all would evaluate on the model's own training "
                 "files; only allowed for cross-body transfer")

    has_val = any((DATA_CACHE / args.eval_body / "continuous" / "val").glob("*.npz"))
    if args.threshold is not None:
        best_thr = args.threshold
        print(f"using fixed threshold {best_thr:.2f}")
    elif not has_val:
        best_thr = 0.5
        print("no val traces — defaulting threshold to 0.50")
    else:
        print("tuning threshold on val ...")
        sweep = []
        for thr in list(np.arange(0.2, 0.95, 0.05)) + [0.95, 0.97, 0.98, 0.99, 0.995]:
            s = eval_cnn(model, args.eval_body, "val", float(thr), device)
            print(f"  thr={thr:.2f} P={s.precision:.3f} R={s.recall:.3f} F1={s.f1:.3f}")
            sweep.append((float(thr), s.f1))
        best_f1 = max(f1 for _, f1 in sweep)
        # near-ties (val is small) break toward the LOWER threshold: extreme
        # thresholds overfit the val set and cost recall on test
        best_thr = min(thr for thr, f1 in sweep if f1 >= best_f1 - 0.02)

    splits = ["train", "val", "test"] if args.eval_split == "all" else ["test"]
    print(f"eval on {splits} with thr={best_thr:.2f}")
    cnn_test = Scores()
    for sp in splits:
        cnn_test.merge(eval_cnn(model, args.eval_body, sp, best_thr, device))

    result = {
        "model_body": model_body,
        "eval_body": args.eval_body,
        "threshold": best_thr,
        "eval_split": args.eval_split,
        "tolerance_sec": CFG.window.match_tolerance_sec,
        "cnn": cnn_test.as_dict(),
    }

    if not args.skip_baseline:
        # STA/LTA has no trained weights, so tuning on train is legitimate
        # when no val split exists (mars has too few files for one).
        tune_split = "val" if has_val else "train"
        print(f"STA/LTA baseline sweep on {tune_split} ...")
        best_on, best_bf1 = 3.0, -1.0
        for thr_on in [2.0, 2.5, 3.0, 4.0, 5.0, 7.0]:
            s = eval_stalta(args.eval_body, tune_split, thr_on)
            print(f"  on={thr_on} P={s.precision:.3f} R={s.recall:.3f} F1={s.f1:.3f}")
            if s.f1 > best_bf1:
                best_on, best_bf1 = thr_on, s.f1
        base_test = Scores()
        for sp in splits:
            base_test.merge(eval_stalta(args.eval_body, sp, best_on))
        result["sta_lta"] = {"thr_on": best_on, **base_test.as_dict()}

    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    run_name = Path(args.model).parent.name  # e.g. lunar, lunar_noaug, mars
    out_path = out_dir / f"{run_name}_to_{args.eval_body}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()

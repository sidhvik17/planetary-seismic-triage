"""Hybrid detector: SpecUNet mask curve x SeisCNN window probability.

The two detectors fail differently: the injection-trained U-Net finds every
event-like energy burst (high recall, fires on uncatalogued signals); the
window-classifier SeisCNN learned the catalog's implicit selection function
(high precision at its operating point, misses weak events). Their false
positives are largely uncorrelated, so the pointwise PRODUCT of the U-Net's
stitched mask-energy curve and the CNN's window probability (held over each
window's span) suppresses single-detector noise while keeping events both
agree on.

Protocol unchanged: (threshold, min-duration) tuned on VAL only, metrics on
TEST, same scorer/tolerance. Writes results/hybrid_to_lunar.json.

Usage: python scripts/eval_hybrid.py --unet runs/unet_lunar/best.pt
           --cnn runs/lunar/best.pt --eval-body lunar
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
from planetseis.detect import _forward_windows
from planetseis.detect_spec import SEC_PER_BIN, compute_curve, curve_to_detections
from planetseis.evaluate import Scores, score_trace
from planetseis.model import ARCHS, SeisCNN
from planetseis.preprocessing import normalize_window
from planetseis.spectral import NOVERLAP
from planetseis.unet import UNET_ARCHS, SpecUNet

THR_GRID = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30,
            0.40, 0.50]
DUR_GRID = [30, 240, 430, 600]


def cnn_bin_scores(cnn, trace, n_bins, device):
    """SeisCNN window probabilities spread onto the curve's bin timeline.

    Each 8192-sample window (hop 4096) covers 128 bins; a bin's score is the
    max probability among windows covering it (max, not mean: the CNN's
    arrival can sit anywhere inside its window).
    """
    n, hop = CFG.window.n_samples, CFG.window.hop
    if len(trace) < n:
        trace = np.pad(trace, (0, n - len(trace)))
    starts = list(range(0, len(trace) - n + 1, hop))
    if starts[-1] != len(trace) - n:
        starts.append(len(trace) - n)
    with torch.no_grad():
        cnn.eval().to(device)
        probs, _ = _forward_windows(cnn, trace, starts, n, device)
    out = np.zeros(n_bins)
    for s, p in zip(starts, probs):
        b0 = s // NOVERLAP
        b1 = min(b0 + n // NOVERLAP, n_bins)
        out[b0:b1] = np.maximum(out[b0:b1], p)
    return out


def curves_for_split(unet, cnn, body, split, device):
    out = []
    for p in sorted((DATA_CACHE / body / "continuous" / split).glob("*.npz")):
        z = np.load(p)
        trace = z["trace"]
        u = compute_curve(unet, trace, device)
        c = cnn_bin_scores(cnn, trace, len(u), device)
        out.append((p.stem, u * c, list(z["picks"])))
    return out


def score_at(curves, thr, min_dur_sec, coda, tol):
    total = Scores()
    min_bins = max(1, int(round(min_dur_sec / SEC_PER_BIN)))
    for _, curve, picks in curves:
        dets = curve_to_detections(curve, thr, suppress_sec=coda,
                                   min_bins=min_bins)
        total.merge(score_trace([d.time_sec for d in dets], picks, tol))
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet", required=True)
    ap.add_argument("--cnn", required=True)
    ap.add_argument("--eval-body", default="lunar",
                    choices=["lunar", "mars", "mars_ext"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    uck = torch.load(args.unet, map_location=device, weights_only=True)
    unet = SpecUNet(base=UNET_ARCHS[uck.get("arch", "base")])
    unet.load_state_dict(uck["model"])
    cck = torch.load(args.cnn, map_location=device, weights_only=True)
    cnn = SeisCNN(channels=ARCHS[cck.get("arch", "base")])
    cnn.load_state_dict(cck["model"])

    coda = CODA_SEC[args.eval_body]
    tol = CFG.window.match_tolerance_sec
    print("computing val curves ...")
    val = curves_for_split(unet, cnn, args.eval_body, "val", device)
    results = []
    for dur in DUR_GRID:
        for thr in THR_GRID:
            s = score_at(val, thr, dur, coda, tol)
            results.append((thr, dur, s.f1))
        print(f"  dur={dur:4.0f}s F1@thr:",
              [f"{f1:.2f}" for t, d, f1 in results if d == dur])
    best_f1 = max(f1 for _, _, f1 in results)
    tied = sorted((t, d) for t, d, f1 in results if f1 >= best_f1 - 0.02)
    best_thr, best_dur = tied[len(tied) // 2]
    print(f"selected thr={best_thr} min_dur={best_dur}s (val F1={best_f1:.3f})")

    print("computing test curves ...")
    test = curves_for_split(unet, cnn, args.eval_body, "test", device)
    s = score_at(test, best_thr, best_dur, coda, tol)

    result = {
        "model": "hybrid_unet_x_seiscnn",
        "unet": Path(args.unet).parent.name,
        "cnn": Path(args.cnn).parent.name,
        "eval_body": args.eval_body,
        "threshold": best_thr,
        "min_dur_sec": best_dur,
        "threshold_tuned_on": "val",
        "tolerance_sec": tol,
        "scores": s.as_dict(),
    }
    out = PROJECT_ROOT / "results" / f"hybrid_to_{args.eval_body}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Denoise-then-detect chaining: SpecUNet denoiser feeding SeisCNN.

MQNet 2022's design logic is joint detection + noise removal. This repo has
both halves but they never touched: the mask-denoiser gains ~8 dB SDR over
bandpass exactly at low SNR, and the supervised SeisCNN's 8 lunar false
negatives are mostly low-SNR events. Chain them:

  variants evaluated (all thresholds re-tuned on VAL only, test untouched):
    raw      SeisCNN on the raw preprocessed trace (reference = benchmark)
    chain    SeisCNN on the SpecUNet-denoised trace
    fusion   per-window max of SeisCNN probabilities on raw and denoised
             (an event visible in either stream fires)

Denoised traces are computed once per file and cached under
data/cache/{body}/denoised/. Writes results/chain_to_{body}.json.

Usage: python scripts/eval_chain.py --unet runs/unet_lunar/best.pt
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
from planetseis.detect import _forward_windows, cluster_detections
from planetseis.detect_spec import denoise_trace
from planetseis.evaluate import Scores, score_trace
from planetseis.model import ARCHS, SeisCNN
from planetseis.unet import UNET_ARCHS, SpecUNet

THR_GRID = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995]


def denoised_for(unet, body, split, stem, trace, device):
    cache = DATA_CACHE / body / "denoised" / split / f"{stem}.npz"
    if cache.exists():
        return np.load(cache)["trace"]
    den = denoise_trace(unet, trace, 6.625, device)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, trace=den.astype(np.float32))
    return den


def cnn_probs(cnn, trace, device):
    n, hop = CFG.window.n_samples, CFG.window.hop
    if len(trace) < n:
        trace = np.pad(trace, (0, n - len(trace)))
    starts = list(range(0, len(trace) - n + 1, hop))
    with torch.no_grad():
        probs, offs = _forward_windows(cnn, trace, starts, n, device,
                                       batch_size=64)
    return np.array(starts), probs, offs


def load_split(unet, cnn, body, split, device):
    """Per file: (picks, win data for raw / chain / fusion variants)."""
    import gc

    out = []
    for p in sorted((DATA_CACHE / body / "continuous" / split).glob("*.npz")):
        z = np.load(p)
        trace, picks = z["trace"], list(z["picks"])
        den = denoised_for(unet, body, split, p.stem, trace, device)
        s_r, pr_r, off_r = cnn_probs(cnn, trace, device)
        s_d, pr_d, off_d = cnn_probs(cnn, den, device)
        del trace, den, z
        gc.collect()
        m = min(len(pr_r), len(pr_d))
        fusion_probs = np.maximum(pr_r[:m], pr_d[:m])
        # fusion arrival offset: take it from the stream that fired harder
        fusion_offs = np.where(pr_r[:m] >= pr_d[:m], off_r[:m], off_d[:m])
        out.append(dict(
            picks=picks,
            raw=(s_r, pr_r, off_r),
            chain=(s_d, pr_d, off_d),
            fusion=(s_r[:m], fusion_probs, fusion_offs),
        ))
    return out


def score_variant(files, variant, thr, body):
    total = Scores()
    n = CFG.window.n_samples
    win_sec = n / 6.625
    for f in files:
        starts, probs, offs = f[variant]
        dets = cluster_detections(probs, offs, np.array(starts) / 6.625,
                                  win_sec, thr,
                                  suppress_sec=CODA_SEC[body])
        total.merge(score_trace([d.time_sec for d in dets], f["picks"],
                                CFG.window.match_tolerance_sec))
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
    unet.eval().to(device)
    cck = torch.load(args.cnn, map_location=device, weights_only=True)
    cnn = SeisCNN(channels=ARCHS[cck.get("arch", "base")])
    cnn.load_state_dict(cck["model"])
    cnn.eval().to(device)

    print("scoring val (threshold tuning) ...")
    val = load_split(unet, cnn, args.eval_body, "val", device)
    best = {}
    for variant in ("raw", "chain", "fusion"):
        sweep = [(thr, score_variant(val, variant, thr, args.eval_body).f1)
                 for thr in THR_GRID]
        best_f1 = max(f1 for _, f1 in sweep)
        tied = [t for t, f1 in sweep if f1 >= best_f1 - 0.02]
        best[variant] = tied[len(tied) // 2]
        print(f"  {variant:7s} val sweep {[f'{f1:.2f}' for _, f1 in sweep]} "
              f"-> thr={best[variant]}")

    print("scoring test ...")
    test = load_split(unet, cnn, args.eval_body, "test", device)
    result = {"unet": Path(args.unet).parent.name,
              "cnn": Path(args.cnn).parent.name,
              "eval_body": args.eval_body,
              "tolerance_sec": CFG.window.match_tolerance_sec}
    for variant in ("raw", "chain", "fusion"):
        s = score_variant(test, variant, best[variant], args.eval_body)
        result[variant] = {"threshold": best[variant], **s.as_dict()}
        print(f"  {variant:7s} thr={best[variant]:.3f} P={s.precision:.3f} "
              f"R={s.recall:.3f} F1={s.f1:.3f} MAE={s.mae_sec}")

    out = PROJECT_ROOT / "results" / f"chain_to_{args.eval_body}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

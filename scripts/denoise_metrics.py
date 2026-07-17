"""Denoising quality metrics (Dahmen & Stott GJI 2024 suite).

On synthetic-injection validation samples the clean event component is
known exactly, so denoising can be scored objectively — the same argument
the MQNet2024 denoising paper uses. For each val sample with an event:

  input   = noise + scaled event (what the lander records)
  target  = the scaled event component alone
  denoised = ISTFT(predicted mask x STFT(input))
  baseline = the bandpassed input itself (classical filtering already
             applied by the shared preprocessing)

Metrics per sample, reported by injection SNR bin:
  CC    cross-correlation with the clean event
  SDR   signal-to-distortion ratio, 10 log10(||target||^2 / ||target - x||^2)
  AMP   peak-amplitude ratio recovered/true

Usage: python scripts/denoise_metrics.py --body lunar [--n 200]
Writes results/denoise_metrics_{body}.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import PROJECT_ROOT, RUNS_DIR, SEED
from planetseis.injection import InjectionDataset, inject
from planetseis.spectral import istft_window, normalize_stft, stft_window
from planetseis.unet import UNET_ARCHS, SpecUNet

SNR_BINS = [(0.4, 1.0), (1.0, 2.5), (2.5, 6.0), (6.0, 12.0)]


def metrics(x: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    n = min(len(x), len(target)) - 64          # drop unreliable last frame
    x, t = x[:n], target[:n]
    denom = np.linalg.norm(x) * np.linalg.norm(t)
    cc = float(np.dot(x, t) / denom) if denom > 0 else 0.0
    err = np.linalg.norm(t - x) ** 2
    sdr = float(10 * np.log10(np.linalg.norm(t) ** 2 / err)) if err > 0 else np.inf
    amp = float(np.abs(x).max() / (np.abs(t).max() + 1e-30))
    return cc, sdr, amp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default="lunar",
                    choices=["lunar", "mars", "mars_ext"])
    ap.add_argument("--run", default=None,
                    help="run name (default unet_{body})")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    run = args.run or f"unet_{args.body}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(RUNS_DIR / run / "best.pt", map_location=device,
                      weights_only=False)
    model = SpecUNet(base=UNET_ARCHS[ckpt.get("arch", "base")])
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    # val-split generators (falls back to train pools only for mars, which
    # has none — mars_ext has a real val split)
    try:
        ds = InjectionDataset(args.body, "val", epoch_len=args.n, seed=SEED)
    except RuntimeError:
        ds = InjectionDataset(args.body, "train", epoch_len=args.n,
                              seed=SEED + 1)
    rng = np.random.default_rng(SEED)

    per_bin = {f"{lo}-{hi}": {"den": [], "bp": []} for lo, hi in SNR_BINS}
    for i in range(args.n):
        # regenerate a controlled sample: known SNR, one event
        srng = np.random.default_rng([SEED, i])
        noise = ds.pool.sample(srng)
        tpl = ds.bank[srng.integers(len(ds.bank))]
        lo, hi = SNR_BINS[i % len(SNR_BINS)]
        snr = float(np.exp(srng.uniform(np.log(lo), np.log(hi))))
        ev, _, _ = inject(noise, tpl, snr, float(srng.uniform(0.1, 0.7)))
        x_in = ev + noise
        Z = stft_window(x_in)
        with torch.no_grad():
            xt = torch.from_numpy(normalize_stft(Z)[None]).to(device)
            mask = torch.sigmoid(model(xt))[0, 0].cpu().numpy()
        den = istft_window(mask * Z)
        cc_d, sdr_d, amp_d = metrics(den, ev)
        cc_b, sdr_b, amp_b = metrics(x_in, ev)
        key = f"{lo}-{hi}"
        per_bin[key]["den"].append((cc_d, sdr_d, amp_d))
        per_bin[key]["bp"].append((cc_b, sdr_b, amp_b))

    def summarize(vals):
        a = np.array(vals)
        return {"cc": round(float(np.median(a[:, 0])), 3),
                "sdr_db": round(float(np.median(a[:, 1])), 2),
                "amp_ratio": round(float(np.median(a[:, 2])), 3),
                "n": len(vals)}

    result = {
        "model": run,
        "body": args.body,
        "note": "median over injection val samples; target = clean scaled "
                "event component; bp = bandpassed input (classical baseline)",
        "by_snr": {k: {"denoised": summarize(v["den"]),
                       "bandpass": summarize(v["bp"])}
                   for k, v in per_bin.items()},
    }
    out = PROJECT_ROOT / "results" / f"denoise_metrics_{args.body}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

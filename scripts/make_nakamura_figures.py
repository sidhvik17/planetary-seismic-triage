"""Waveform evidence for benchmark FPs that match Nakamura catalog events.

For the top-confidence matched detections, plot a 3-hour window: filtered
waveform, detection curve, our detection time, and the Nakamura catalog
entry (event start, minute-quantized) with its type code. Visual companion
to results/nakamura_crosscheck.json.

Usage: python scripts/make_nakamura_figures.py --model runs/unet_lunar/best.pt
           [--n 3]
Writes docs/figures/nakamura_match_*.png.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import DATA_CACHE, PROJECT_ROOT
from planetseis.detect_spec import SEC_PER_BIN, compute_curve
from planetseis.unet import UNET_ARCHS, SpecUNet
from scripts.crosscheck_nakamura import load_nakamura, trace_start_utc

TYPE_NAMES = {"A": "deep moonquake (nest)", "M": "deep moonquake",
              "C": "meteoroid impact", "H": "shallow moonquake",
              "Z": "mostly SP event", "L": "LM impact", "S": "S-IVB impact"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    det_csv = PROJECT_ROOT / "results" / "nakamura_crosscheck_detections.csv"
    df = pd.read_csv(det_csv)
    matched = (df[df.status == "fp_matches_nakamura"]
               .sort_values("confidence", ascending=False).head(args.n))
    if not len(matched):
        print("no matched FPs in", det_csv)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    model = SpecUNet(base=UNET_ARCHS[ckpt.get("arch", "base")])
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    nak = load_nakamura()
    nak_times = np.array([float(t) for t in nak["utc"]])
    fig_dir = PROJECT_ROOT / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for i, (_, row) in enumerate(matched.iterrows()):
        stem = row["file"]
        z = np.load(DATA_CACHE / "lunar" / "continuous" / "test" / f"{stem}.npz")
        trace = z["trace"]
        t0 = trace_start_utc(stem)
        abs_t = float(t0) + row["time_sec"]
        k = int(np.argmin(np.abs(nak_times - abs_t)))
        ev = nak.iloc[k]
        ev_rel = nak_times[k] - float(t0)
        etype = TYPE_NAMES.get(str(ev["EventType"]).strip(),
                               str(ev["EventType"]))

        curve = compute_curve(model, trace, device)
        lo = max(0.0, row["time_sec"] - 5400)
        hi = min(len(trace) / 6.625, row["time_sec"] + 5400)
        s0, s1 = int(lo * 6.625), int(hi * 6.625)
        t = np.arange(s0, s1) / 6.625 / 3600
        b0, b1 = int(lo / SEC_PER_BIN), int(hi / SEC_PER_BIN)
        tc = np.arange(b0, b1) * SEC_PER_BIN / 3600

        fig, (a0, a1) = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                                     height_ratios=[2, 1])
        a0.plot(t, trace[s0:s1], lw=0.3, color="#3D3833")
        a0.axvline(row["time_sec"] / 3600, color="#C74E00", ls="--", lw=1.5,
                   label="U-Net detection")
        a0.axvline(ev_rel / 3600, color="#2E6FB8", ls=":", lw=2,
                   label=f"Nakamura {int(ev['Year'])+1900} DOY{int(ev['DOY'])} "
                         f"{int(ev['StartTime']):04d} — {etype}")
        a0.legend(loc="upper right", fontsize=9)
        a0.set_ylabel("filtered amplitude")
        a0.set_title(f"{stem} — benchmark FP matching Nakamura event "
                     f"(Δt={row['nakamura_dt_sec']:.0f}s, "
                     f"conf={row['confidence']:.2f})")
        a1.plot(tc, curve[b0:b1], color="#C74E00", lw=1)
        a1.set_ylabel("mask energy")
        a1.set_xlabel("time (h)")
        fig.tight_layout()
        out = fig_dir / f"nakamura_match_{i+1}_{stem[-11:]}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print("saved", out)


if __name__ == "__main__":
    main()

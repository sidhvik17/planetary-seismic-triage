"""Report figures for the spectrogram U-Net (MQNet-style) results.

Usage: python scripts/make_unet_figures.py
Writes docs/figures/unet_*.png. Requires trained runs/unet_lunar and
(optionally) runs/unet_mars_ext checkpoints plus the data caches.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import CODA_SEC, DEFAULT as CFG, DATA_CACHE, PROJECT_ROOT, RUNS_DIR
from planetseis.detect_spec import denoise_trace, detect_events_spec
from planetseis.spectral import SEC_PER_BIN, stft_window, N_SAMPLES
from planetseis.unet import UNET_ARCHS, SpecUNet

FIG_DIR = PROJECT_ROOT / "docs" / "figures"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(run: str):
    ckpt = torch.load(RUNS_DIR / run / "best.pt", map_location=DEVICE,
                      weights_only=True)
    m = SpecUNet(base=UNET_ARCHS[ckpt.get("arch", "base")])
    m.load_state_dict(ckpt["model"])
    return m.eval().to(DEVICE)


def fig_training_curves():
    fig, ax = plt.subplots(figsize=(7, 4))
    for run, color in (("unet_lunar", "#C74E00"), ("unet_mars_ext", "#2E6FB8")):
        log = RUNS_DIR / run / "log.csv"
        if not log.exists():
            continue
        rows = list(csv.DictReader(open(log)))
        ep = [int(r["epoch"]) for r in rows]
        ax.plot(ep, [float(r["train_loss"]) for r in rows], color=color,
                alpha=0.4, label=f"{run} train")
        ax.plot(ep, [float(r["val_loss"]) for r in rows], color=color,
                label=f"{run} val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("weighted BCE")
    ax.legend()
    ax.set_title("SpecUNet training on synthetic-injection data")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "unet_training.png", dpi=130)
    plt.close(fig)


def fig_detection_example(body: str, run: str, threshold: float):
    model = load_model(run)
    test_dir = DATA_CACHE / body / "continuous" / "test"
    files = sorted(test_dir.glob("*.npz"))
    if not files:
        return
    # pick the file whose event count is highest for a busy example
    best = max(files, key=lambda p: len(np.load(p)["picks"]))
    z = np.load(best)
    trace, rate, picks = z["trace"], float(z["rate"]), list(z["picks"])
    dets, curve = detect_events_spec(model, trace, rate, CFG, threshold,
                                     DEVICE, suppress_sec=CODA_SEC[body])
    t_tr = np.arange(len(trace)) / rate / 3600
    t_cv = np.arange(len(curve)) * SEC_PER_BIN / 3600

    fig, (a0, a1) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True,
                                 height_ratios=[2, 1])
    step = max(1, len(trace) // 100_000)
    a0.plot(t_tr[::step], trace[::step], lw=0.3, color="#3D3833")
    for p in picks:
        a0.axvline(p / 3600, color="k", ls=":", lw=1.5)
    for d in dets:
        a0.axvline(d.time_sec / 3600, color="#C74E00", ls="--", lw=1.2)
    a0.set_ylabel("filtered amplitude")
    a0.set_title(f"{best.stem} — catalog picks (dotted) vs U-Net detections "
                 f"(dashed), thr={threshold}")
    a1.plot(t_cv, curve, color="#C74E00", lw=1)
    a1.axhline(threshold, color="#3D3833", ls=":")
    a1.set_ylabel("mask energy")
    a1.set_xlabel("time (h)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"unet_detection_{body}.png", dpi=130)
    plt.close(fig)


def fig_denoise_example(body: str, run: str):
    model = load_model(run)
    test_dir = DATA_CACHE / body / "continuous" / "test"
    files = sorted(test_dir.glob("*.npz"))
    if not files:
        return
    z = np.load(files[0])
    trace, rate, picks = z["trace"], float(z["rate"]), list(z["picks"])
    if not len(picks):
        return
    # 8192-sample window centered on the first pick
    c = int(picks[0] * rate)
    s = max(0, min(c - N_SAMPLES // 3, len(trace) - N_SAMPLES))
    seg = trace[s : s + N_SAMPLES]
    den = denoise_trace(model, seg, rate, DEVICE)
    S_in = np.abs(stft_window(seg))
    S_out = np.abs(stft_window(den.astype(np.float32)))
    vmax = np.percentile(S_in, 99.5)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, S, title in ((axes[0], S_in, "input"),
                         (axes[1], S_out, "denoised (mask × STFT)")):
        ax.imshow(S, aspect="auto", origin="lower", cmap="magma",
                  vmax=vmax, extent=[0, N_SAMPLES / rate / 60, 0, 3.3125])
        ax.set_title(title)
        ax.set_xlabel("minutes")
    axes[0].set_ylabel("frequency (Hz)")
    fig.suptitle(f"{files[0].stem} — event at {picks[0]:.0f}s")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"unet_denoise_{body}.png", dpi=130)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_training_curves()
    results = {}
    for body, run in (("lunar", "unet_lunar"), ("mars_ext", "unet_mars_ext")):
        res_path = PROJECT_ROOT / "results" / f"{run}_to_{body}.json"
        thr = 0.3
        if res_path.exists():
            thr = json.loads(res_path.read_text())["threshold"]
        if (RUNS_DIR / run / "best.pt").exists():
            fig_detection_example(body, run, thr)
            fig_denoise_example(body, run)
    print("figures ->", FIG_DIR)


if __name__ == "__main__":
    main()

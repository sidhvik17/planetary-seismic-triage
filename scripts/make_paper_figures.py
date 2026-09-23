"""Paper-only figures (written to paper/figures/, which is gitignored).

fig1_detection_example: full test day + zoom, catalog pick vs model arrival.
fig2_architecture: dual-head SeisCNN schematic.
Also copies the five results figures from docs/figures/.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrow, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import CODA_SEC, DEFAULT as CFG, DATA_CACHE, PROJECT_ROOT
from planetseis.detect import detect_events
from planetseis.model import ARCHS, SeisCNN

ORANGE, SLATE, INK, GRID = "#C74E00", "#2E6FB8", "#3D3833", "#D8CBAA"
FIGS = PROJECT_ROOT / "paper" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
})

STEM = "xa.s12.00.mhz.1970-12-11HR00_evid00017"


def fig1():
    z = np.load(DATA_CACHE / "lunar" / "continuous" / "test" / f"{STEM}.npz")
    trace, rate, picks = z["trace"], float(z["rate"]), list(z["picks"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(PROJECT_ROOT / "runs" / "lunar" / "best.pt",
                    map_location=device, weights_only=True)
    model = SeisCNN(channels=ARCHS[ck.get("arch", "base")]).to(device)
    model.load_state_dict(ck["model"])
    dets, ws, wp = detect_events(model, trace, rate, CFG, 0.99, device,
                                 suppress_sec=CODA_SEC["lunar"])
    t = np.arange(len(trace)) / rate / 3600  # hours

    fig, (a, b) = plt.subplots(2, 1, figsize=(7.2, 4.6),
                               gridspec_kw={"height_ratios": [1, 1.2]})
    step = max(1, len(trace) // 40000)
    a.plot(t[::step], trace[::step], lw=0.4, color=INK)
    for p in picks:
        a.axvline(p / 3600, color=INK, ls=":", lw=1.4)
    for d in dets:
        a.axvline(d.time_sec / 3600, color=ORANGE, ls="--", lw=1.4)
    a.set_xlabel("time (h)")
    a.set_ylabel("velocity (filtered)")
    a.set_title(f"Apollo 12, 1970-12-11 (held-out) — catalog pick (dotted) vs "
                f"model arrival (dashed, conf {dets[0].confidence:.3f})", fontsize=9)

    p0 = picks[0]
    lo, hi = int((p0 - 1800) * rate), int((p0 + 5400) * rate)
    tz = np.arange(lo, hi) / rate
    b.plot(tz / 3600, trace[lo:hi], lw=0.5, color=INK)
    b.axvline(p0 / 3600, color=INK, ls=":", lw=1.6, label="NASA catalog pick")
    for d in dets:
        if lo / rate < d.time_sec < hi / rate:
            b.axvline(d.time_sec / 3600, color=ORANGE, ls="--", lw=1.6,
                      label=f"model ({abs(d.time_sec - p0):.0f} s error)")
    b.set_xlabel("time (h)")
    b.set_ylabel("velocity (filtered)")
    b.legend(frameon=False, fontsize=8, loc="upper right")
    b.set_title("2-hour zoom around the event onset", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_detection_example.png", dpi=150, bbox_inches="tight")
    print("fig1 saved; detection error:",
          [f"{abs(d.time_sec - p0):.0f}s" for d in dets])


def _box(ax, x, y, w, h, text, fc="#F3E7CB"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                fc=fc, ec=INK, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)


def fig2():
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    _box(ax, 0.1, 1.5, 1.5, 1.0, "window\n8192 × 1\n(~21 min)")
    _box(ax, 2.0, 1.5, 2.6, 1.0,
         "conv backbone\n6 × [Conv1d s2 + BN + ReLU]\nk 9→3, ch 16→96")
    _box(ax, 5.0, 1.5, 1.3, 1.0, "features\n96 × 128")
    _box(ax, 6.9, 2.4, 2.9, 0.9,
         "detection head\nmean+max pool → MLP(64)\n→ P(event)", fc="#F8D9BD")
    _box(ax, 6.9, 0.6, 2.9, 0.9,
         "arrival head\nconv → softmax(time)\n→ soft-argmax offset", fc="#F8D9BD")
    for x0, y0, x1, y1 in [(1.6, 2.0, 2.0, 2.0), (4.6, 2.0, 5.0, 2.0),
                           (6.3, 2.2, 6.9, 2.8), (6.3, 1.8, 6.9, 1.1)]:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.text(5, 3.6, "SeisCNN — 117,842 parameters, dual-head", ha="center",
            fontsize=10, fontweight="bold")
    fig.savefig(FIGS / "fig2_architecture.png", dpi=150, bbox_inches="tight")
    print("fig2 saved")


def copy_result_figures():
    for name in ["pareto.png", "reliability.png", "pr_curve.png",
                 "sample_efficiency.png", "active_learning.png"]:
        shutil.copy(PROJECT_ROOT / "docs" / "figures" / name, FIGS / name)
    print("result figures copied")


if __name__ == "__main__":
    fig1()
    fig2()
    copy_result_figures()

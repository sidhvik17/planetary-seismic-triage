"""Report figures: efficiency-accuracy Pareto + reliability diagram.

Reads results/*.json; writes docs/figures/pareto.png, reliability.png.
Palette (validated): CNN #C74E00, secondary #2E6FB8, ink #3D3833.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import DEFAULT as CFG, PROJECT_ROOT
from planetseis.model import ARCHS, SeisCNN, count_params

ORANGE, SLATE, INK, GRID = "#C74E00", "#2E6FB8", "#3D3833", "#D8CBAA"
FIGS = PROJECT_ROOT / "docs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
})


def cpu_ms(arch: str) -> float:
    m = SeisCNN(channels=ARCHS[arch]).eval()
    x = torch.randn(1, 1, CFG.window.n_samples)
    with torch.no_grad():
        for _ in range(5):
            m(x)
        t0 = time.perf_counter()
        for _ in range(50):
            m(x)
    return (time.perf_counter() - t0) / 50 * 1000


def pareto():
    res = {}
    for arch, fname in [("tiny", "lunar_tiny_to_lunar.json"),
                        ("base", "lunar_to_lunar.json"),
                        ("large", "lunar_large_to_lunar.json")]:
        p = PROJECT_ROOT / "results" / fname
        if p.exists():
            res[arch] = json.loads(p.read_text())
    stalta_f1 = json.loads(
        (PROJECT_ROOT / "results" / "lunar_to_lunar.json").read_text()
    )["sta_lta"]["f1"]

    params = {a: count_params(SeisCNN(channels=ARCHS[a])) for a in res}
    lat = {a: cpu_ms(a) for a in res}
    print("latency ms:", lat)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, xs, xlabel in [
        (axes[0], params, "parameters"),
        (axes[1], lat, "CPU latency per window (ms)"),
    ]:
        ax.axhline(stalta_f1, color=INK, ls="--", lw=1)
        ax.annotate("tuned STA/LTA", xy=(0.98, stalta_f1), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=8, color=INK)
        for a in ("tiny", "base", "large"):
            if a not in res:
                continue
            f1 = res[a]["cnn"]["f1"]
            ax.scatter(xs[a], f1, s=70, color=ORANGE, zorder=3,
                       edgecolor="white", linewidth=1)
            ax.annotate(f"{a}\n({params[a]/1e3:.0f}K)", xy=(xs[a], f1),
                        xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("test F1 (lunar)")
        ax.set_ylim(0, 0.75)
        if xlabel == "parameters":
            ax.set_xscale("log")
    fig.suptitle("Efficiency vs accuracy — capacity beyond ~120K params buys nothing", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "pareto.png", dpi=150, bbox_inches="tight")
    print("saved pareto.png")
    # persist measured latencies for the report
    (PROJECT_ROOT / "results" / "pareto.json").write_text(json.dumps({
        a: {"params": params[a], "cpu_ms": round(lat[a], 1), "f1": res[a]["cnn"]["f1"],
            "precision": res[a]["cnn"]["precision"], "recall": res[a]["cnn"]["recall"],
            "mae_sec": res[a]["cnn"]["mae_sec"]}
        for a in res}, indent=2))


def reliability():
    p = PROJECT_ROOT / "results" / "uncertainty.json"
    if not p.exists():
        print("uncertainty.json missing — run scripts/uncertainty_eval.py first")
        return
    u = json.loads(p.read_text())
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.plot([0, 1], [0, 1], color=INK, ls=":", lw=1, label="perfect calibration")
    for key, color, label in [("raw", ORANGE, f"raw (ECE {u['ece_raw']:.3f})"),
                              ("mc", SLATE, f"MC-Dropout (ECE {u['ece_mc']:.3f})")]:
        pts = [(c, b[1]) for c, b, n in u["reliability_bins"][key] if b]
        ax.plot([x for x, _ in pts], [y for _, y in pts], "o-", color=color,
                lw=1.6, ms=5, label=label, markeredgecolor="white")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed event frequency")
    ax.set_title("Reliability — lunar test windows", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(FIGS / "reliability.png", dpi=150, bbox_inches="tight")
    print("saved reliability.png")


if __name__ == "__main__":
    pareto()
    reliability()

"""Goal 5 — uncertainty-gated active labeling under extreme scarcity.

Simulates the mission loop the triage dashboard implies: start with 5 labeled
files, train, then repeatedly choose which unlabeled file to send for human
labeling. Acquisition = the file whose windows the current model is most
UNCERTAIN about (max MC-Dropout std among candidate windows), vs a random-
acquisition control. Metric: test F1 as a function of labels spent.

If uncertainty-guided acquisition reaches target performance with fewer
labels than random, the review queue is not just a diagnostic — it is a
label-efficient learning policy ("uncertainty-gated active triage").

Writes results/active_learning.json + docs/figures/active_learning.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import CODA_SEC, DEFAULT as CFG, DATA_CACHE, PROJECT_ROOT
from planetseis.detect import _enable_mc_dropout, _forward_windows, _window_starts
from planetseis.model import ARCHS, SeisCNN
from scripts.sample_efficiency import build_fewshot_windows, eval_body, train_detector, tune_thr

ORANGE, SLATE, INK, GRID = "#C74E00", "#2E6FB8", "#3D3833", "#D8CBAA"
ROUNDS = [5, 10, 15, 20]      # cumulative labeled files after each round
SEEDS = [0, 1, 2]
MC_PASSES = 10

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
})


def file_uncertainty(model, path, device):
    """Acquisition score: max MC std among the file's candidate windows."""
    z = np.load(path)
    trace, rate = z["trace"], float(z["rate"])
    n, hop = CFG.window.n_samples, CFG.window.hop
    trace, starts = _window_starts(trace, n, hop)
    _enable_mc_dropout(model)
    reps = []
    with torch.no_grad():
        for _ in range(MC_PASSES):
            p, _ = _forward_windows(model, trace, starts, n, device)
            reps.append(p)
    model.eval()
    reps = np.stack(reps)
    std = reps.std(0)
    mean = reps.mean(0)
    cand = std[mean >= 0.2]
    return float(cand.max()) if len(cand) else float(std.max())


def run_strategy(strategy, seed, pool, device):
    rng = np.random.default_rng(seed)
    order = list(rng.permutation(len(pool)))
    labeled = [pool[i] for i in order[:ROUNDS[0]]]
    unlabeled = [pool[i] for i in order[ROUNDS[0]:]]
    curve = []
    model = None
    for target in ROUNDS:
        while len(labeled) < target and unlabeled:
            if strategy == "random" or model is None:
                pick = unlabeled.pop(0)
            else:
                scores = [file_uncertainty(model, p, device) for p in unlabeled]
                pick = unlabeled.pop(int(np.argmax(scores)))
            labeled.append(pick)
        X, y, off = build_fewshot_windows("lunar", labeled, rng)
        model = train_detector(X, y, off, init_ssl=True, device=device, seed=seed)
        thr = tune_thr(model, "lunar", device)
        s = eval_body(model, "lunar", "test", thr, device)
        curve.append({"n_labels": len(labeled), "f1": round(s.f1, 4),
                      "recall": round(s.recall, 4), "precision": round(s.precision, 4)})
        print(f"{strategy} seed{seed} n={len(labeled)}: F1={s.f1:.3f} R={s.recall:.3f}")
    return curve


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pool = sorted((DATA_CACHE / "lunar" / "continuous" / "train").glob("*.npz"))
    out_path = PROJECT_ROOT / "results" / "active_learning.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    for strategy in ("uncertainty", "random"):
        for seed in SEEDS:
            key = f"{strategy}_seed{seed}"
            if key in results:
                print(f"skip {key}")
                continue
            results[key] = run_strategy(strategy, seed, pool, device)
            out_path.write_text(json.dumps(results, indent=2))

    # aggregate + figure
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    for strategy, color, label in [("random", INK, "random acquisition"),
                                   ("uncertainty", ORANGE, "uncertainty-guided")]:
        ys = np.array([[pt["f1"] for pt in results[f"{strategy}_seed{s}"]] for s in SEEDS])
        mean, std = ys.mean(0), ys.std(0, ddof=1)
        ax.errorbar(ROUNDS, mean, yerr=std, fmt="o-", color=color, lw=1.6, ms=5,
                    capsize=3, label=label, markeredgecolor="white")
    ax.set_xlabel("labeled files spent")
    ax.set_ylabel("test F1")
    ax.set_xticks(ROUNDS)
    ax.set_title("Active labeling: uncertainty-guided vs random", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "docs" / "figures" / "active_learning.png",
                dpi=150, bbox_inches="tight")
    print("saved active_learning.png")


if __name__ == "__main__":
    main()

"""Goal 1 money figure: F1 vs number of labeled events, SSL-init vs scratch.

For n in {5,10,20,45} labeled lunar events (files), 3 seeds each, train the
detector (a) from scratch, (b) init from the SSL-pretrained backbone
(runs/ssl/encoder.pt). Threshold tuned on the fixed val split (constant
across conditions), F1 measured on the fixed 19-event test split.

Also Goal 2: few-shot Mars with the SSL backbone (n = 1 file, all we have),
vs the scratch Mars model — anecdotal n, reported as such.

Writes results/sample_efficiency.json + docs/figures/sample_efficiency.png.
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import CODA_SEC, DEFAULT as CFG, DATA_CACHE, PROJECT_ROOT, RUNS_DIR
from planetseis.dataset import WindowDataset
from planetseis.detect import detect_events
from planetseis.evaluate import Scores, score_trace
from planetseis.model import ARCHS, SeisCNN
from planetseis.ssl import transfer_backbone
from planetseis.train import run_epoch, set_seed
from planetseis.windows import make_windows, positives_around_pick

ORANGE, SLATE, INK, GRID = "#C74E00", "#2E6FB8", "#3D3833", "#D8CBAA"
N_SHOTS = [5, 10, 20, 45]
SEEDS = [0, 1, 2]
EPOCHS = 40

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
})


def build_fewshot_windows(body, files, rng):
    """Balanced window set from the selected labeled files only."""
    X, y, off = [], [], []
    for p in files:
        z = np.load(p)
        trace, rate, picks = z["trace"], float(z["rate"]), list(z["picks"])
        wins = make_windows(trace, rate, picks, CFG.window)
        pos = [w for w in wins if w.label == 1]
        coda = CODA_SEC[body]
        win_sec = CFG.window.n_samples / rate
        neg = [w for w in wins if w.label == 0 and not any(
            w.start_sec < pk + coda and w.start_sec + win_sec > pk for pk in picks)]
        for pk in picks:
            pos += positives_around_pick(trace, rate, pk, CFG.window, n_shifts=12, rng=rng)
        k = min(len(neg), int(np.ceil(len(pos) * CFG.train.neg_pos_ratio)) or 8)
        neg = [neg[i] for i in rng.choice(len(neg), size=k, replace=False)] if neg else []
        for w in pos + neg:
            X.append(w.data)
            y.append(w.label)
            off.append(w.offset_frac)
    return np.stack(X), np.array(y, np.int64), np.array(off, np.float32)


def train_detector(X, y, off, init_ssl, device, seed):
    set_seed(seed)
    model = SeisCNN(channels=ARCHS["base"]).to(device)
    if init_ssl:
        transfer_backbone(RUNS_DIR / "ssl" / "encoder.pt", model)
    dl = DataLoader(WindowDataset(X, y, off, augment=True, seed=seed),
                    batch_size=min(64, len(y)), shuffle=True)
    n_pos = int(y.sum())
    pw = torch.tensor((len(y) - n_pos) / max(n_pos, 1), dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(EPOCHS):
        run_epoch(model, dl, device, opt, CFG.train.lambda_reg, pw)
    return model


def eval_body(model, body, split, thr, device):
    s = Scores()
    for p in sorted((DATA_CACHE / body / "continuous" / split).glob("*.npz")):
        z = np.load(p)
        dets, _, _ = detect_events(model, z["trace"], float(z["rate"]), CFG, thr,
                                   device, suppress_sec=CODA_SEC[body])
        s.merge(score_trace([d.time_sec for d in dets], list(z["picks"]),
                            CFG.window.match_tolerance_sec))
    return s


def tune_thr(model, body, device):
    best_thr, best_f1 = 0.5, -1.0
    for thr in [0.5, 0.7, 0.9, 0.95, 0.98, 0.99]:
        f1 = eval_body(model, body, "val", thr, device).f1
        if f1 > best_f1:
            best_thr, best_f1 = thr, f1
    return best_thr


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_files = sorted((DATA_CACHE / "lunar" / "continuous" / "train").glob("*.npz"))
    results = {"lunar": {}, "mars_fewshot": {}}

    for n in N_SHOTS:
        for init in ("scratch", "ssl"):
            key = f"n{n}_{init}"
            f1s = []
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                files = [train_files[i] for i in
                         rng.choice(len(train_files), size=min(n, len(train_files)),
                                    replace=False)]
                X, y, off = build_fewshot_windows("lunar", files, rng)
                model = train_detector(X, y, off, init == "ssl", device, seed)
                thr = tune_thr(model, "lunar", device)
                s = eval_body(model, "lunar", "test", thr, device)
                f1s.append(s.f1)
                print(f"{key} seed{seed}: thr={thr} F1={s.f1:.3f} "
                      f"(P={s.precision:.3f} R={s.recall:.3f})")
            results["lunar"][key] = {
                "n_events": n, "init": init,
                "f1_runs": [round(f, 4) for f in f1s],
                "f1_mean": round(float(np.mean(f1s)), 4),
                "f1_std": round(float(np.std(f1s, ddof=1)), 4),
            }

    # Goal 2: Mars few-shot (1 labeled file — anecdotal, stated)
    mars_files = sorted((DATA_CACHE / "mars" / "continuous" / "train").glob("*.npz"))
    for init in ("scratch", "ssl"):
        rng = np.random.default_rng(0)
        X, y, off = build_fewshot_windows("mars", mars_files, rng)
        model = train_detector(X, y, off, init == "ssl", device, seed=0)
        s = eval_body(model, "mars", "test", 0.5, device)
        results["mars_fewshot"][init] = {**s.as_dict(), "note": "n=1 test event, anecdotal"}
        print(f"mars {init}: {s.as_dict()}")

    (PROJECT_ROOT / "results" / "sample_efficiency.json").write_text(
        json.dumps(results, indent=2))

    # figure
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    for init, color, label in [("scratch", INK, "from scratch"),
                               ("ssl", ORANGE, "SSL-pretrained")]:
        xs, ys, es = [], [], []
        for n in N_SHOTS:
            r = results["lunar"][f"n{n}_{init}"]
            xs.append(n)
            ys.append(r["f1_mean"])
            es.append(r["f1_std"])
        ax.errorbar(xs, ys, yerr=es, fmt="o-", color=color, lw=1.6, ms=5,
                    capsize=3, label=label, markeredgecolor="white")
    ax.set_xlabel("labeled events (files)")
    ax.set_ylabel("test F1")
    ax.set_xscale("log")
    ax.set_xticks(N_SHOTS)
    ax.set_xticklabels([str(n) for n in N_SHOTS])
    ax.set_title("Sample efficiency: SSL pretraining vs scratch (lunar)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "docs" / "figures" / "sample_efficiency.png",
                dpi=150, bbox_inches="tight")
    print("saved sample_efficiency.png")


if __name__ == "__main__":
    main()

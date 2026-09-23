"""PRD-promised analyses: P/R tradeoff curve (F-2) and SNR-stratified recall (F-3).

Writes results/pr_curve.json, results/snr_recall.json and
docs/figures/pr_curve.png. Lunar only (Mars test is a single event).
Corrected split: --data-dir data/cache/lunar_grouped_v1 --model <best.pt>
--threshold <val-selected> --tag <name> writes *_<name> files and never
overwrites. The test sweep is descriptive; it selects nothing.
"""
from __future__ import annotations

import argparse
import hashlib
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
from planetseis.detect import detect_events
from planetseis.evaluate import Scores, score_trace
from planetseis.model import ARCHS, SeisCNN

ORANGE, SLATE, INK, GRID = "#C74E00", "#2E6FB8", "#3D3833", "#D8CBAA"
FIGS = PROJECT_ROOT / "docs" / "figures"
OPERATING_THR = 0.99

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load_split(split, root=DATA_CACHE / "lunar"):
    d = root / "continuous" / split
    for p in sorted(d.glob("*.npz")):
        z = np.load(p)
        yield p.stem, z["trace"], float(z["rate"]), list(z["picks"])


def sweep(model, split, thresholds, device, root=DATA_CACHE / "lunar"):
    out = []
    per_thr_hits = {t: [] for t in thresholds}  # (pick, hit) at each thr
    data = list(load_split(split, root))
    for thr in thresholds:
        s = Scores()
        for _, trace, rate, picks in data:
            dets, _, _ = detect_events(model, trace, rate, CFG, thr, device,
                                       suppress_sec=CODA_SEC["lunar"])
            times = [d.time_sec for d in dets]
            s.merge(score_trace(times, picks, CFG.window.match_tolerance_sec))
            for p in picks:
                hit = any(abs(t - p) <= CFG.window.match_tolerance_sec for t in times)
                per_thr_hits[thr].append(hit)
        out.append({"threshold": float(thr), "precision": s.precision,
                    "recall": s.recall, "f1": s.f1})
        print(f"{split} thr={thr:.3f} P={s.precision:.3f} R={s.recall:.3f}")
    return out, data


def event_snr_db(trace, rate, pick):
    """Post-onset signal RMS vs pre-onset noise RMS, both 5-minute windows."""
    i = int(pick * rate)
    sig = trace[i: i + int(300 * rate)]
    noise = trace[max(0, i - int(900 * rate)): max(1, i - int(300 * rate))]
    if len(sig) < 10 or len(noise) < 10:
        return None
    return float(20 * np.log10(np.sqrt(np.mean(sig ** 2)) /
                               (np.sqrt(np.mean(noise ** 2)) + 1e-12)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=PROJECT_ROOT / "runs" / "lunar" / "best.pt")
    ap.add_argument("--threshold", type=float, default=OPERATING_THR,
                    help="the model's validation-selected operating threshold")
    ap.add_argument("--data-dir", type=Path, help="versioned dataset root; requires --tag")
    ap.add_argument("--tag", default="", help="suffix for every output file")
    args = ap.parse_args()
    operating_thr = args.threshold
    root = args.data_dir or DATA_CACHE / "lunar"
    sfx = f"_{args.tag}" if args.tag else ""
    outputs = [PROJECT_ROOT / "results" / f"pr_curve{sfx}.json",
               PROJECT_ROOT / "results" / f"snr_recall{sfx}.json",
               FIGS / f"pr_curve{sfx}.png"]
    provenance = {"benchmark_id": "historical filename split", "data_manifest_sha256": None}
    if args.data_dir is not None:
        if not args.tag:
            ap.error("--data-dir requires --tag")
        existing = [str(o) for o in outputs if o.exists()]
        if existing:
            sys.exit(f"refusing to overwrite {existing}")
        raw_manifest = (root / "manifest.json").read_bytes()
        provenance = {"benchmark_id": json.loads(raw_manifest)["benchmark_id"],
                      "data_manifest_sha256": hashlib.sha256(raw_manifest).hexdigest()}
    provenance["model"] = args.model.name
    provenance["model_sha256"] = hashlib.sha256(args.model.read_bytes()).hexdigest()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.model, map_location=device, weights_only=True)
    model = SeisCNN(channels=ARCHS[ck.get("arch", "base")]).to(device)
    model.load_state_dict(ck["model"])

    thresholds = sorted({0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995,
                         operating_thr})
    val_curve, _ = sweep(model, "val", thresholds, device, root)
    test_curve, test_data = sweep(model, "test", thresholds, device, root)
    outputs[0].write_text(
        json.dumps({**provenance, "operating_threshold": operating_thr,
                    "note": "descriptive sweep; the operating threshold was selected on val",
                    "val": val_curve, "test": test_curve,
                    "tolerance_sec": CFG.window.match_tolerance_sec}, indent=2))

    # ---- SNR-stratified recall at the operating threshold (test set)
    events = []
    for _, trace, rate, picks in test_data:
        dets, _, _ = detect_events(model, trace, rate, CFG, operating_thr, device,
                                   suppress_sec=CODA_SEC["lunar"])
        times = [d.time_sec for d in dets]
        for p in picks:
            snr = event_snr_db(trace, rate, p)
            if snr is None:
                continue
            hit = any(abs(t - p) <= CFG.window.match_tolerance_sec for t in times)
            events.append({"snr_db": round(snr, 2), "detected": bool(hit)})
    snrs = np.array([e["snr_db"] for e in events])
    hits = np.array([e["detected"] for e in events])
    med = float(np.median(snrs))
    lo, hi = snrs < med, snrs >= med
    snr_out = {
        **provenance,
        "operating_threshold": operating_thr,
        "n_events": len(events),
        "median_snr_db": round(med, 2),
        "recall_low_snr": round(float(hits[lo].mean()), 3) if lo.any() else None,
        "recall_high_snr": round(float(hits[hi].mean()), 3) if hi.any() else None,
        "events": events,
    }
    outputs[1].write_text(json.dumps(snr_out, indent=2))
    print(json.dumps({k: v for k, v in snr_out.items() if k != "events"}, indent=2))

    # ---- figure
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for curve, color, label in [(val_curve, SLATE, "val (tuning set)"),
                                (test_curve, ORANGE, "test (held-out)")]:
        r = [c["recall"] for c in curve]
        p = [c["precision"] for c in curve]
        ax.plot(r, p, "o-", color=color, lw=1.6, ms=4.5, label=label,
                markeredgecolor="white")
    op = next(c for c in test_curve if abs(c["threshold"] - operating_thr) < 1e-9)
    ax.scatter([op["recall"]], [op["precision"]], s=130, facecolor="none",
               edgecolor=INK, linewidth=1.5, zorder=5)
    ax.annotate(f"operating point (thr {operating_thr})",
                xy=(op["recall"], op["precision"]), xytext=(8, 10),
                textcoords="offset points", fontsize=8)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    label = "lunar" if args.data_dir is None else provenance["benchmark_id"]
    ax.set_title(f"Detection tradeoff across thresholds ({label})", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(outputs[2], dpi=150, bbox_inches="tight")
    print(f"saved {outputs[2].name}")


if __name__ == "__main__":
    main()

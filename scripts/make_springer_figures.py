"""Redraw the three figures of the Springer manuscript from results/.

    python scripts/make_springer_figures.py                 # -> docs/figures/paper/ (git-ignored)
    python scripts/make_springer_figures.py --out some/dir

Same drawing code as the manuscript's own figure script, so the output is the
figure printed in the paper. Every value is read from the result JSON files;
nothing is typed in. Figures are grayscale-safe (markers, line styles and
hatching carry the distinctions), use a sans-serif font and are written as
vector PDF with embedded fonts plus a 300 dpi PNG.

  fig1_benchmark   per-seed test F1: SeisCNN, SpecUNet, screened SpecUNet,
                   with the validation-tuned classical baselines
  fig2_catalog     catalog tolerance diagnostic over five SpecUNet seeds
  fig3_archive     corrected continuous-archive scan: recall vs unmatched
                   detections per station-day, and recall by event class
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import PROJECT_ROOT

RES = PROJECT_ROOT / "results"
OUT = PROJECT_ROOT / "docs" / "figures" / "paper"

K, DARK, MID, LIGHT = "#000000", "#404040", "#808080", "#C8C8C8"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "lines.linewidth": 1.0, "hatch.linewidth": 0.5, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False})
WIDTH = 4.75  # inches; proceedings text block is about 12.2 cm


def load(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)


def fig1_benchmark():
    s = load("lunar_grouped_v1_seed_summary.json")
    ab = load("lunar_grouped_v1_screening_ablation.json")
    arms = [("SeisCNN\n(supervised)", [r["f1"] for r in s["detectors"]["cnn"]["per_seed"]], "o"),
            ("SpecUNet\n(injection)", [r["f1"] for r in s["detectors"]["unet"]["per_seed"]], "s"),
            ("SpecUNet,\nscreened noise", [r["test"]["f1"] for r in ab["per_seed"]["screened"]], "^")]
    fig, ax = plt.subplots(figsize=(WIDTH, 2.3))
    rng = np.random.default_rng(0)
    for x, (label, f1, marker) in enumerate(arms):
        f1 = np.asarray(f1)
        ax.scatter(x + rng.uniform(-.09, .09, len(f1)), f1, marker=marker, s=18,
                   facecolor="white", edgecolor=K, linewidth=.8, zorder=3)
        m, sd = f1.mean(), f1.std(ddof=1)
        ax.plot([x - .22, x + .22], [m, m], color=K, lw=1.4, zorder=2)
        ax.errorbar(x, m, yerr=sd, color=DARK, capsize=3, lw=.8, zorder=2)
        ax.text(x + .26, m, f"{m:.3f} ± {sd:.3f}", va="center", fontsize=6.5)
    base = s["baselines"]
    for value, label, style, va in ((base["matched_filter"]["f1"], "template matching", "--", "bottom"),
                                    (base["sta_lta"]["f1"], "STA/LTA", ":", "top")):
        ax.axhline(value, color=MID, lw=.8, ls=style, zorder=1)
        ax.text(-.46, value + (.004 if va == "bottom" else -.004), f"{label} {value:.3f}",
                va=va, ha="left", fontsize=6, color=DARK)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a[0] for a in arms])
    ax.set_xlim(-.5, 2.7)
    ax.set_ylim(0, .65)
    ax.set_ylabel("test F1 per seed")
    save(fig, "fig1_benchmark")


def fig2_catalog():
    sec = load("lunar_grouped_v1_secondary_summary.json")
    audit = load("lunar_grouped_v1_catalog_tolerance_audit.json")
    rows = sec["per_seed"]
    keys = list(rows[0]["nakamura"]["match_count_by_tolerance"])
    pooled = [sum(r["nakamura"]["match_count_by_tolerance"][k] for r in rows) for k in keys]
    assoc = [x for r in audit["per_seed"] for x in r["associations"]]
    labelled, extra = [], []
    for key, total in zip(keys, pooled):
        hits = [x for x in assoc if x["detection_to_catalog_sec"] <= float(key.strip("±s"))]
        if len(hits) != total:
            raise ValueError(f"audit does not reproduce the {key} count")
        labelled.append(sum(x["catalog_entry_already_grade_a"] for x in hits))
        extra.append(len(hits) - labelled[-1])
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(WIDTH, 2.2), gridspec_kw={"width_ratios": [1.35, 1]})
    x = np.arange(len(keys))
    a0.bar(x, labelled, color=LIGHT, edgecolor=K, lw=.5, label="already a Grade-A label")
    a0.bar(x, extra, bottom=labelled, color="white", edgecolor=K, lw=.5, hatch="////",
           label="absent from Grade-A")
    for xi, (a, b) in enumerate(zip(labelled, extra)):
        a0.text(xi, a + b + 1, f"{a}+{b}", ha="center", fontsize=6)
    a0.set_xticks(x)
    a0.set_xticklabels(keys)
    a0.set_xlabel("match tolerance")
    a0.set_ylabel("benchmark false positives\n(pooled over 5 seeds)")
    a0.set_ylim(0, max(pooled) * 1.25)
    a0.legend(frameon=False, loc="upper left")
    obs = [r["nakamura"]["match_rate"] for r in rows]
    chance = [r["nakamura"]["chance_rate"] for r in rows]
    for xi, vals, marker in ((0, obs, "o"), (1, chance, "s")):
        a1.bar(xi, np.mean(vals), color=LIGHT if xi == 0 else "white", edgecolor=K, lw=.5, width=.55)
        a1.scatter(np.full(len(vals), xi) + np.linspace(-.12, .12, len(vals)), vals, marker=marker,
                   s=12, facecolor="white", edgecolor=K, lw=.7, zorder=3)
        a1.text(xi, max(vals) + .03, f"{np.mean(vals):.1%}", ha="center", fontsize=6.5)
    a1.set_xticks([0, 1])
    a1.set_xticklabels(["observed", "random\nplacement"])
    a1.set_ylabel("fraction within ±300 s")
    a1.set_ylim(0, max(obs) * 1.25)
    save(fig, "fig2_catalog")


def fig3_archive():
    sw = load("archive_scan/grouped_v1_seed42/sweep.json")
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(WIDTH, 2.3), gridspec_kw={"width_ratios": [1.1, 1]})
    styles = {"S12": ("-", "o"), "S15": ("--", "s"), "S16": (":", "^")}
    for st, (ls, marker) in styles.items():
        grid = sorted((r for r in sw["stations"][st]["sweep"]
                       if r["recall"] is not None and r["unmatched_per_station_day"]),
                      key=lambda r: (r["unmatched_per_station_day"], -r["recall"]))
        # Pareto frontier over the (threshold, duration) grid: the best recall
        # reachable at or below each false-alarm rate
        rows, best = [], -1.0
        for r in grid:
            if r["recall"] > best:
                rows.append(r)
                best = r["recall"]
        a0.plot([r["unmatched_per_station_day"] for r in rows], [r["recall"] for r in rows],
                ls=ls, marker=marker, ms=3, color=K, mfc="white", lw=.9,
                label=f"{st} ({sw['stations'][st]['valid_station_days']:.0f} d)")
    a0.set_xscale("log")
    a0.set_xlabel("unmatched detections per station-day")
    a0.set_ylabel("recall of Nakamura catalog")
    a0.legend(frameon=False, loc="upper left")
    want = [("deep_moonquake", "deep"), ("shallow_moonquake", "shallow"), ("code_C", "code C")]
    hatches = {"S12": "", "S15": "////", "S16": "...."}
    fills = {"S12": LIGHT, "S15": "white", "S16": "white"}
    w = .26
    for i, st in enumerate(styles):
        row = min(sw["stations"][st]["sweep"], key=lambda r: (r["threshold"], r["min_dur_sec"]))
        vals = [row["recall_by_type"].get(k, {}).get("recall") or 0 for k, _ in want]
        ns = [row["recall_by_type"].get(k, {}).get("n", 0) for k, _ in want]
        xs = np.arange(len(want)) + (i - 1) * w
        a1.bar(xs, vals, width=w, color=fills[st], edgecolor=K, lw=.5, hatch=hatches[st], label=st)
        for xv, v, n in zip(xs, vals, ns):
            a1.text(xv, v + .02, str(n), ha="center", fontsize=5, rotation=90)
    a1.set_xticks(range(len(want)))
    a1.set_xticklabels([lbl for _, lbl in want])
    a1.set_ylabel("recall (most permissive point)")
    a1.set_ylim(0, 1.05)
    a1.legend(frameon=False, ncol=3, loc="upper right", fontsize=6)
    save(fig, "fig3_archive")


def main():
    global RES, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RES)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    RES, OUT = args.results, args.out
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_benchmark()
    fig2_catalog()
    fig3_archive()
    print(f"figures -> {OUT} (results from {RES})")


if __name__ == "__main__":
    main()

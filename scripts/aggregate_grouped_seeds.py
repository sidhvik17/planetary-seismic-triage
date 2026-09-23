"""Seed-level summary of the acquisition-grouped lunar benchmark.

    python scripts/aggregate_grouped_seeds.py

Reads every results/lunar_grouped_v1_seed*.json written by
scripts/evaluate_grouped.py (each at its own validation-selected operating
point), checks they share one data manifest and distinct seeds, and reports:

  * per-seed test precision / recall / F1 / MAE for SeisCNN and SpecUNet
  * seed mean and sample SD (ddof=1), as benchmark rule 2 requires
  * Welch t-test on per-seed F1 (training stochasticity; unpaired, since a
    shared seed number does not pair two different architectures)
  * a 95 % bootstrap interval that resamples test acquisition GROUPS and
    averages over the five observed seed models (seeds are not resampled)
  * the deterministic STA/LTA and matched-filter baselines on the same split

Nothing here selects or tunes anything; it only summarises locked results.
Writes results/lunar_grouped_v1_seed_summary.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import PROJECT_ROOT

BENCHMARK_ID = "lunar_grouped_v1"
FAMILIES = {"cnn": "SeisCNN (supervised windows)",
            "unet": "SpecUNet (injection-trained masks)"}


def f1_from_counts(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def load_runs(results_dir: Path):
    runs = []
    for path in sorted(results_dir.glob(f"{BENCHMARK_ID}_seed*.json")):
        # The default summary shares this glob's prefix. Only seed-numbered
        # experiment files are inputs; summaries and selection locks are not.
        if not re.fullmatch(rf"{BENCHMARK_ID}_seed\d+(?:_(?:cnn|unet))?\.json", path.name):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("benchmark_id") != BENCHMARK_ID:
            raise ValueError(f"{path.name}: wrong benchmark_id")
        runs.append((path, data))
    if not runs:
        raise SystemExit(f"no {BENCHMARK_ID}_seed*.json results in {results_dir}")
    manifests = {d["data_manifest_sha256"] for _, d in runs}
    if len(manifests) != 1:
        raise ValueError("results come from different data manifests")
    for family in FAMILIES:
        seeds = [d["checkpoints"][family]["seed"] for _, d in runs if family in d["checkpoints"]]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"{family}: repeated training seed {seeds}")
    return runs, manifests.pop()


def group_counts(per_trace):
    """tp/fp/fn summed per acquisition group (the resampling unit)."""
    out = {}
    for row in per_trace:
        s = row["scores"]
        acc = out.setdefault(row["group_id"], np.zeros(3, dtype=np.int64))
        acc += (s["tp"], s["fp"], s["fn"])
    return out


def bootstrap(runs, n_boot: int, seed: int):
    """Resample test groups; average each family's per-seed F1 per replicate."""
    counts = {f: [group_counts(d["results"][f]["test_per_trace"])
                  for _, d in runs if f in d["results"]] for f in FAMILIES}
    groups = sorted(next(iter(counts["cnn"] or counts["unet"])).keys())
    rng = np.random.default_rng(seed)
    draws = {f: [] for f in FAMILIES}
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(len(groups), size=len(groups), replace=True)
        means = {}
        for family, per_seed in counts.items():
            if not per_seed:
                continue
            f1s = []
            for table in per_seed:
                tp, fp, fn = np.sum([table[groups[i]] for i in pick], axis=0)
                f1s.append(f1_from_counts(tp, fp, fn))
            means[family] = float(np.mean(f1s))
            draws[family].append(means[family])
        if len(means) == 2:
            diffs.append(means["cnn"] - means["unet"])
    ci = lambda xs: [round(float(np.percentile(xs, 2.5)), 4),
                     round(float(np.percentile(xs, 97.5)), 4)] if xs else None
    return {"n_boot": n_boot, "resampling_unit": "test acquisition group",
            "n_groups": len(groups), "seed_mean_f1_ci95": {f: ci(v) for f, v in draws.items()},
            "cnn_minus_unet_f1_ci95": ci(diffs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    ap.add_argument("--output", type=Path,
                    default=PROJECT_ROOT / "results" / f"{BENCHMARK_ID}_seed_summary.json")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()

    runs, manifest_sha = load_runs(args.results_dir)
    detectors = {}
    for family, label in FAMILIES.items():
        rows = []
        for path, d in runs:
            if family not in d["results"]:
                continue
            r = d["results"][family]
            rows.append({"seed": d["checkpoints"][family]["seed"], "file": path.name,
                         "checkpoint_epoch": d["checkpoints"][family]["epoch"],
                         "operating_point": r["operating_point"],
                         "selected_val_f1": round(r["selected_val_f1"], 4),
                         **r["test_scores"]})
        if not rows:
            continue
        f1 = np.array([row["f1"] for row in rows])
        detectors[family] = {
            "label": label, "n_seeds": len(rows), "per_seed": rows,
            "mean_f1": round(float(f1.mean()), 4),
            "std_f1": round(float(f1.std(ddof=1)), 4) if len(f1) > 1 else None,
            "min_f1": round(float(f1.min()), 4), "max_f1": round(float(f1.max()), 4),
            "mean_precision": round(float(np.mean([r["precision"] for r in rows])), 4),
            "mean_recall": round(float(np.mean([r["recall"] for r in rows])), 4),
        }

    tests = {}
    if {"cnn", "unet"} <= detectors.keys() and min(
            detectors[f]["n_seeds"] for f in ("cnn", "unet")) > 1:
        a = [r["f1"] for r in detectors["cnn"]["per_seed"]]
        b = [r["f1"] for r in detectors["unet"]["per_seed"]]
        t, p = stats.ttest_ind(a, b, equal_var=False)
        tests["seiscnn_vs_specunet_welch"] = {
            "welch_t": round(float(t), 3), "p_value": round(float(p), 4),
            "mean_difference": round(float(np.mean(a) - np.mean(b)), 4),
            "ratio_specunet_to_seiscnn_mean_f1": round(float(np.mean(b) / np.mean(a)), 4)
            if np.mean(a) else None,
        }

    baselines = {}
    sta = {json.dumps(d["results"]["sta_lta"]["test_scores"], sort_keys=True)
           for _, d in runs if "sta_lta" in d["results"]}
    if len(sta) > 1:
        raise ValueError("deterministic STA/LTA baseline differs between runs")
    if sta:
        first = next(d for _, d in runs if "sta_lta" in d["results"])["results"]["sta_lta"]
        baselines["sta_lta"] = {"operating_point": first["operating_point"],
                                **first["test_scores"]}
    mf = args.results_dir / f"matched_filter_{BENCHMARK_ID}.json"
    if mf.exists():
        m = json.loads(mf.read_text(encoding="utf-8"))
        baselines["matched_filter"] = {
            "template_sec": m["template_sec"], "threshold_k_mad": m["threshold_k_mad"],
            "n_templates": m["n_templates"], "val_f1": m["val_f1"], **m["matched_filter"],
            "oracle_test_tuned_upper_bound_not_reportable": m["oracle_test_tuned_upper_bound"]}

    summary = {
        "benchmark_id": BENCHMARK_ID, "data_manifest_sha256": manifest_sha,
        "protocol": "Each seed trained from random initialization on lunar_grouped_v1 and "
                    "scored at its own validation-selected operating point "
                    "(scripts/evaluate_grouped.py); the test split never informs selection.",
        "detectors": detectors, "tests": tests,
        "bootstrap": bootstrap(runs, args.n_boot, seed=0),
        "baselines": baselines,
        "comparability": "New acquisition-grouped split with unioned picks (21 test spans, "
                         "23 events). Not directly comparable to historical filename-split "
                         "scores, which included two train/test duplicate waveforms.",
    }
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("tests", "baselines")}, indent=2))
    for family, d in detectors.items():
        print(f"{family}: F1 {d['mean_f1']} +/- {d['std_f1']} (n={d['n_seeds']})")
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()

"""Seed-level summary of the corrected-split secondary analyses.

    python scripts/aggregate_grouped_secondary.py

Reads, for every seed with a locked evaluation, the Nakamura cross-check,
both MC-Dropout uncertainty analyses and the SNR-stratified recall written by
scripts/run_grouped_secondary.py. Before summarising, it checks that each
file names the same data manifest and the exact checkpoint (SHA256) the
locked evaluation scored, and that its detection counts reproduce that
evaluation. It reports per-seed values and the mean and sample SD
(ddof = 1). Nothing here selects or tunes anything.

Writes results/lunar_grouped_v1_secondary_summary.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import PROJECT_ROOT

BENCHMARK_ID = "lunar_grouped_v1"


def load(results: Path, name: str) -> dict:
    return json.loads((results / name).read_text(encoding="utf-8"))


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {"mean": round(float(array.mean()), 4),
            "sd": round(float(array.std(ddof=1)), 4) if len(array) > 1 else None,
            "min": round(float(array.min()), 4), "max": round(float(array.max()), 4),
            "n": len(array)}


def seed_row(results: Path, seed: int, manifest: str) -> dict:
    tag = f"{BENCHMARK_ID}_seed{seed}"
    cnn_eval = load(results, f"{tag}_cnn.json")
    unet_eval = load(results, f"{tag}_unet.json")
    nak = load(results, f"nakamura_crosscheck_{tag}.json")
    unc_cnn = load(results, f"uncertainty_{tag}.json")
    unc_unet = load(results, f"uncertainty_unet_{tag}.json")
    snr = load(results, f"snr_recall_{tag}.json")

    for name, data, family in (("nakamura", nak, "unet"), ("uncertainty_unet", unc_unet, "unet"),
                               ("uncertainty", unc_cnn, "cnn"), ("snr_recall", snr, "cnn")):
        evaluation = cnn_eval if family == "cnn" else unet_eval
        if data.get("data_manifest_sha256") != manifest:
            raise ValueError(f"seed {seed} {name}: data manifest differs")
        if data.get("model_sha256") != evaluation["checkpoints"][family]["sha256"]:
            raise ValueError(f"seed {seed} {name}: not the checkpoint the evaluation scored")
    # The cross-check reruns the deterministic detector, so its counts must
    # reproduce the evaluation. MC-Dropout runs average stochastic passes and
    # legitimately differ from the deterministic counts; they are reported.
    unet_scores = unet_eval["results"]["unet"]["test_scores"]
    if any(nak["benchmark"][k] != unet_scores[k] for k in ("tp", "fp", "fn")):
        raise ValueError(f"seed {seed}: Nakamura counts differ from the locked evaluation")
    auto = unc_cnn["review_queue"]["auto_accept"]

    true_std, false_std = unc_cnn["mc_std_true_event_windows"], unc_cnn["mc_std_false_alarm_windows"]
    return {
        "seed": seed,
        "specunet_operating_point": unet_eval["results"]["unet"]["operating_point"],
        "seiscnn_operating_point": cnn_eval["results"]["cnn"]["operating_point"],
        "nakamura": {
            "benchmark_fp": nak["benchmark"]["fp"], "fp_matching": nak["fp_matching_nakamura"],
            "match_rate": nak["fp_nakamura_match_rate"], "chance_rate": nak["chance_match_rate"],
            "permutation_p": nak["permutation_p_value"],
            "benchmark_precision": nak["benchmark"]["precision"],
            "survey_precision": nak["survey_precision"],
            "match_count_by_tolerance": nak["match_count_by_tolerance"],
        },
        "seiscnn_uncertainty": {
            "mc_std_true_event_windows": true_std, "mc_std_false_alarm_windows": false_std,
            "false_over_true": round(false_std / true_std, 3) if true_std else None,
            "ece_raw": unc_cnn["ece_raw"], "ece_temp_scaled": unc_cnn["ece_temp_scaled"],
            "mc_auto_accept": {k: auto[k] for k in ("tp", "fp", "fn", "f1")},
            "review_queue_total": unc_cnn["review_queue"]["review_queue_total"],
            "review_queue_true_events": unc_cnn["review_queue"]["review_queue_true_events"],
        },
        "specunet_uncertainty": {
            "fp_over_tp": unc_unet["sigma_separation_fp_over_tp"],
            "cleanfp_over_real": unc_unet["sigma_separation_cleanfp_over_real"],
            "counts": unc_unet["counts"],
        },
        "seiscnn_snr_recall": {"median_snr_db": snr["median_snr_db"],
                               "recall_low_snr": snr["recall_low_snr"],
                               "recall_high_snr": snr["recall_high_snr"],
                               "n_events": snr["n_events"]},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    ap.add_argument("--output", type=Path,
                    default=PROJECT_ROOT / "results" / f"{BENCHMARK_ID}_secondary_summary.json")
    args = ap.parse_args()
    manifest = load(args.results_dir, f"{BENCHMARK_ID}_seed_summary.json")["data_manifest_sha256"]
    rows = [seed_row(args.results_dir, seed, manifest) for seed in args.seeds]

    def pick(section, key):
        return stats([row[section][key] for row in rows])

    fp = sum(r["nakamura"]["benchmark_fp"] for r in rows)
    matched = sum(r["nakamura"]["fp_matching"] for r in rows)
    summary = {
        "benchmark_id": BENCHMARK_ID, "data_manifest_sha256": manifest, "seeds": args.seeds,
        "protocol": "Each seed's checkpoint at its own validation-locked operating point "
                    "(scripts/run_grouped_secondary.py); counts verified against the locked "
                    "evaluations. Mean and sample SD across seeds.",
        "nakamura": {
            "match_rate": pick("nakamura", "match_rate"),
            "chance_rate": pick("nakamura", "chance_rate"),
            "survey_precision": pick("nakamura", "survey_precision"),
            "benchmark_precision": pick("nakamura", "benchmark_precision"),
            "pooled_fp_matching": f"{matched}/{fp}",
            "max_permutation_p": max(r["nakamura"]["permutation_p"] for r in rows),
            "tolerance_sec": 300,
        },
        "seiscnn_uncertainty": {
            "false_over_true": pick("seiscnn_uncertainty", "false_over_true"),
            "ece_raw": pick("seiscnn_uncertainty", "ece_raw"),
            "ece_temp_scaled": pick("seiscnn_uncertainty", "ece_temp_scaled"),
            "review_queue_total": pick("seiscnn_uncertainty", "review_queue_total"),
            "review_queue_true_events": pick("seiscnn_uncertainty", "review_queue_true_events"),
        },
        "specunet_uncertainty": {
            "fp_over_tp": pick("specunet_uncertainty", "fp_over_tp"),
            "cleanfp_over_real": pick("specunet_uncertainty", "cleanfp_over_real"),
        },
        "seiscnn_snr_recall": {
            "recall_low_snr": pick("seiscnn_snr_recall", "recall_low_snr"),
            "recall_high_snr": pick("seiscnn_snr_recall", "recall_high_snr"),
        },
        "per_seed": rows,
    }
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_seed"}, indent=2))
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()

"""Secondary summary must verify provenance and counts before averaging."""
import json
import sys

import pytest

from scripts import aggregate_grouped_secondary as aggregate

MANIFEST = "m" * 64


def write(directory, name, data):
    (directory / name).write_text(json.dumps(data))


def write_seed(directory, seed, cnn_sha="c", unet_sha="u", nak_fp=4, matched=1, std=(0.02, 0.1)):
    tag = f"lunar_grouped_v1_seed{seed}"
    counts = {"tp": 3, "fp": nak_fp, "fn": 2}
    write(directory, f"{tag}_cnn.json", {
        "checkpoints": {"cnn": {"sha256": "c"}},
        "results": {"cnn": {"operating_point": {"threshold": 0.9},
                            "test_scores": {"tp": 5, "fp": 1, "fn": 1}}}})
    write(directory, f"{tag}_unet.json", {
        "checkpoints": {"unet": {"sha256": "u"}},
        "results": {"unet": {"operating_point": {"threshold": 0.3, "min_dur_sec": 600},
                             "test_scores": {"tp": 3, "fp": 4, "fn": 2}}}})
    base = {"data_manifest_sha256": MANIFEST}
    write(directory, f"nakamura_crosscheck_{tag}.json", {
        **base, "model_sha256": unet_sha, "benchmark": {**counts, "precision": 0.43},
        "fp_matching_nakamura": matched, "fp_nakamura_match_rate": matched / nak_fp,
        "chance_match_rate": 0.02, "permutation_p_value": 0.0, "survey_precision": 0.5,
        "match_count_by_tolerance": {"±300s": matched}})
    write(directory, f"uncertainty_{tag}.json", {
        **base, "model_sha256": cnn_sha, "mc_std_true_event_windows": std[0],
        "mc_std_false_alarm_windows": std[1], "ece_raw": 0.03, "ece_temp_scaled": 0.02,
        "review_queue": {"auto_accept": {"tp": 5, "fp": 1, "fn": 1, "f1": 0.83},
                         "review_queue_total": 3, "review_queue_true_events": 0}})
    write(directory, f"uncertainty_unet_{tag}.json", {
        **base, "model_sha256": unet_sha, "sigma_separation_fp_over_tp": 0.8,
        "sigma_separation_cleanfp_over_real": 0.7, "counts": {"tp": 3, "fp": 3}})
    write(directory, f"snr_recall_{tag}.json", {
        **base, "model_sha256": cnn_sha, "median_snr_db": 14.0, "recall_low_snr": 0.5,
        "recall_high_snr": 0.8, "n_events": 7})


def run(directory, monkeypatch, seeds):
    write(directory, "lunar_grouped_v1_seed_summary.json", {"data_manifest_sha256": MANIFEST})
    output = directory / "summary.json"
    monkeypatch.setattr(sys, "argv", ["aggregate", "--results-dir", str(directory),
                                      "--output", str(output), "--seeds", *map(str, seeds)])
    aggregate.main()
    return json.loads(output.read_text())


def test_summary_pools_and_averages(tmp_path, monkeypatch):
    write_seed(tmp_path, 42, matched=1, std=(0.02, 0.10))
    write_seed(tmp_path, 1, matched=2, std=(0.02, 0.06))
    summary = run(tmp_path, monkeypatch, [42, 1])
    assert summary["nakamura"]["pooled_fp_matching"] == "3/8"
    assert summary["nakamura"]["match_rate"]["mean"] == pytest.approx(0.375)
    assert summary["seiscnn_uncertainty"]["false_over_true"]["mean"] == pytest.approx(4.0)
    assert summary["seiscnn_uncertainty"]["false_over_true"]["n"] == 2


@pytest.mark.parametrize("change,message", [
    ({"unet_sha": "other"}, "not the checkpoint"),
    ({"nak_fp": 5}, "Nakamura counts differ"),
])
def test_mismatched_seed_files_are_rejected(tmp_path, monkeypatch, change, message):
    write_seed(tmp_path, 42, **change)
    with pytest.raises(ValueError, match=message):
        run(tmp_path, monkeypatch, [42])

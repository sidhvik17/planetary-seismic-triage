"""Seed summary must refuse mixed datasets/seeds and compute plain statistics."""
import json
import sys

import numpy as np
import pytest

from scripts import aggregate_grouped_seeds as aggregate


def scores(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 4),
            "recall": round(r, 4), "f1": round(f1, 4), "mae_sec": None,
            "median_ae_sec": None}


def write_run(directory, seed, cnn, unet, manifest="a" * 64):
    per_trace = lambda s: [{"id": "t1", "group_id": "g1", "scores": s}]
    result = {
        "benchmark_id": "lunar_grouped_v1", "data_manifest_sha256": manifest,
        "checkpoints": {"cnn": {"seed": seed, "epoch": 3}, "unet": {"seed": seed, "epoch": 4}},
        "results": {
            "cnn": {"operating_point": {"threshold": 0.5}, "selected_val_f1": 0.5,
                    "test_scores": cnn, "test_per_trace": per_trace(cnn)},
            "unet": {"operating_point": {"threshold": 0.3, "min_dur_sec": 600.0},
                     "selected_val_f1": 0.4, "test_scores": unet,
                     "test_per_trace": per_trace(unet)},
            "sta_lta": {"operating_point": {"thr_on": 3.0}, "test_scores": scores(1, 9, 9)},
        },
    }
    (directory / f"lunar_grouped_v1_seed{seed}.json").write_text(json.dumps(result))
    # A locked selection file sits beside each result and must be ignored.
    (directory / f"lunar_grouped_v1_seed{seed}.selection.json").write_text("{}")


def run(directory, monkeypatch):
    output = directory / "summary.json"
    monkeypatch.setattr(sys, "argv", ["aggregate", "--results-dir", str(directory),
                                      "--output", str(output), "--n-boot", "50"])
    aggregate.main()
    return json.loads(output.read_text())


def test_summary_statistics(tmp_path, monkeypatch):
    write_run(tmp_path, 42, scores(6, 4, 4), scores(4, 6, 6))
    write_run(tmp_path, 1, scores(5, 5, 5), scores(3, 7, 7))
    summary = run(tmp_path, monkeypatch)
    cnn, unet = summary["detectors"]["cnn"], summary["detectors"]["unet"]
    assert cnn["n_seeds"] == unet["n_seeds"] == 2
    assert cnn["mean_f1"] == pytest.approx(np.mean([0.6, 0.5]))
    assert cnn["std_f1"] == pytest.approx(np.std([0.6, 0.5], ddof=1), abs=1e-4)
    assert summary["tests"]["seiscnn_vs_specunet_welch"]["mean_difference"] == pytest.approx(0.2)
    assert summary["baselines"]["sta_lta"]["f1"] == 0.1
    assert summary["bootstrap"]["n_groups"] == 1


def test_mixed_manifests_are_rejected(tmp_path, monkeypatch):
    write_run(tmp_path, 42, scores(1, 1, 1), scores(1, 1, 1))
    write_run(tmp_path, 1, scores(1, 1, 1), scores(1, 1, 1), manifest="b" * 64)
    with pytest.raises(ValueError, match="different data manifests"):
        run(tmp_path, monkeypatch)


def test_summary_can_be_regenerated_with_its_default_filename(tmp_path, monkeypatch):
    write_run(tmp_path, 42, scores(6, 4, 4), scores(4, 6, 6))
    write_run(tmp_path, 1, scores(5, 5, 5), scores(3, 7, 7))
    output = tmp_path / "lunar_grouped_v1_seed_summary.json"
    monkeypatch.setattr(sys, "argv", ["aggregate", "--results-dir", str(tmp_path),
                                      "--output", str(output), "--n-boot", "50"])
    aggregate.main()
    first = json.loads(output.read_text())
    aggregate.main()
    assert json.loads(output.read_text()) == first
    runs, _ = aggregate.load_runs(tmp_path)
    assert len(runs) == 2

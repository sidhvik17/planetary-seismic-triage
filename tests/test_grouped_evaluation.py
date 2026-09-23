"""Grouped evaluation must lock validation decisions before touching test data."""
from datetime import datetime, timedelta, timezone
import hashlib
import json

import numpy as np
import pytest
import torch

from planetseis.config import DEFAULT as CFG
from planetseis.detect import detect_events
from planetseis.evaluate import Scores
from planetseis.model import ARCHS, SeisCNN
from scripts import evaluate_grouped as evaluation


@pytest.fixture
def grouped_cache(tmp_path):
    traces, groups, splits = [], [], {}
    for index, split in enumerate(("train", "val", "test")):
        ident, group = f"trace{index}", f"group{index}"
        start = datetime(1971, index + 1, 1, tzinfo=timezone.utc)
        trace = np.random.default_rng(index).normal(size=8192).astype(np.float32)
        path = tmp_path / "continuous" / split / f"{ident}.npz"
        path.parent.mkdir(parents=True)
        np.savez(path, trace=trace, rate=6.625, picks=[600.0])
        traces.append({
            "id": ident, "group_id": group, "split": split,
            "channel": "XA.S12.00.MHZ", "start_time": start.isoformat(),
            "end_time": (start + timedelta(seconds=len(trace) / 6.625)).isoformat(),
            "n_samples": len(trace), "rate_hz": 6.625,
            "picks_rel_sec": [600.0], "source_files": [f"{ident}.mseed"],
            "cache_file": path.relative_to(tmp_path).as_posix(),
            "trace_sha256": hashlib.sha256(trace.tobytes()).hexdigest(),
        })
        groups.append({"id": group, "split": split, "trace_ids": [ident]})
        splits[split] = [ident]
    manifest = {"benchmark_id": evaluation.BENCHMARK_ID, "traces": traces,
                "groups": groups, "splits": splits}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path, manifest


def save_manifest(directory, manifest):
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def cached_cnn_row(record, probability=0.8):
    return {"record": record, "picks": record["picks_rel_sec"], "cnn": {
        "probs": np.array([probability]), "offsets": np.array([0.5]),
        "start_secs": np.array([0.0]), "win_sec": 1200.0,
    }}


def test_manifest_inventory_validation_does_not_open_test_arrays(grouped_cache, monkeypatch):
    directory, expected = grouped_cache
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("manifest check opened an array"))
    manifest, digest = evaluation.load_manifest(directory)
    assert manifest == expected
    assert digest == hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()


@pytest.mark.parametrize("failure", ["interval", "hash", "group", "inventory", "source", "path"])
def test_manifest_rejects_leakage_and_wrong_file_inventory(grouped_cache, failure):
    directory, manifest = grouped_cache
    train, _, test = manifest["traces"]
    if failure == "interval":
        test["start_time"], test["end_time"] = train["start_time"], train["end_time"]
    elif failure == "hash":
        test["trace_sha256"] = train["trace_sha256"]
    elif failure == "group":
        test["group_id"] = train["group_id"]
    elif failure == "inventory":
        np.savez(directory / "continuous" / "test" / "extra.npz", trace=[1])
    elif failure == "source":
        test["source_files"] = train["source_files"]
    else:
        test["cache_file"] = "../outside.npz"
    save_manifest(directory, manifest)
    with pytest.raises(ValueError):
        evaluation.load_manifest(directory)


@pytest.mark.parametrize("failure", ["waveform", "picks", "rate"])
def test_actual_payload_is_verified_against_manifest(grouped_cache, failure):
    directory, manifest = grouped_cache
    record = manifest["traces"][1]
    trace, rate, picks = evaluation.load_trace(directory, record)
    if failure == "waveform":
        trace[0] += 1
    elif failure == "picks":
        picks = [650.0]
    else:
        rate = 10.0
    np.savez(directory / record["cache_file"], trace=trace, rate=rate, picks=picks)
    with pytest.raises(ValueError, match="SHA256|picks|rate"):
        evaluation.load_trace(directory, record)


@pytest.mark.parametrize("field,value", [
    ("data_manifest_sha256", None), ("data_manifest_sha256", "wrong"),
    ("benchmark_id", "lunar"), ("initialized_from_scratch", False),
    ("initialized_from_scratch", 1), ("config", {"finetune_from": "old.pt"}),
    ("config", {"hardneg": "historical_noise.npz"}),
])
def test_checkpoint_rejects_missing_or_contaminated_provenance(tmp_path, field, value):
    checkpoint = {"data_manifest_sha256": "expected", "benchmark_id": evaluation.BENCHMARK_ID,
                  "initialized_from_scratch": True, "config": {}}
    checkpoint[field] = value
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        evaluation.load_checkpoint(path, "cnn", "expected")


def test_checkpoint_load_keeps_sha_and_training_provenance(tmp_path):
    model = SeisCNN(channels=ARCHS["tiny"])
    path = tmp_path / "new.pt"
    torch.save({"model": model.state_dict(), "arch": "tiny", "epoch": 3,
                "data_manifest_sha256": "expected", "benchmark_id": evaluation.BENCHMARK_ID,
                "initialized_from_scratch": True, "config": {"seed": 42}}, path)
    loaded, provenance = evaluation.load_checkpoint(path, "cnn", "expected")
    assert provenance["sha256"] == evaluation.sha256_file(path)
    assert provenance["epoch"] == 3 and provenance["seed"] == 42
    assert not loaded.training
    assert all(torch.equal(value, loaded.state_dict()[key]) for key, value in model.state_dict().items())


def test_cnn_cached_outputs_match_frozen_detector():
    class ToyModel(torch.nn.Module):
        def forward(self, x):
            return x[:, 0, 0], torch.sigmoid(x[:, 0, 20]), x[:, 0, :2]

    trace = np.random.default_rng(123).normal(size=CFG.window.n_samples * 4).astype(np.float32)
    model = ToyModel()
    cached = evaluation.cnn_outputs(model, trace, 6.625, "cpu")
    row = {"record": {"id": "val", "group_id": "g", "source_files": [],
                       "trace_sha256": "x", "split": "val"}, "picks": [600.0], "cnn": cached}
    for threshold in [0.2, 0.5, 0.9]:
        _, rows = evaluation.score_outputs([row], "cnn", {"threshold": threshold})
        expected, _, _ = detect_events(model, trace, 6.625, CFG, threshold,
                                        "cpu", batch_size=64, suppress_sec=1800)
        assert rows[0]["predictions"] == [
            {"time_sec": d.time_sec, "confidence": d.confidence} for d in expected
        ]


def test_original_cnn_near_tie_prefers_lower_threshold(grouped_cache, monkeypatch):
    # F1=.989899 for 49 TP/1 FN is within .02 of the higher-threshold optimum.
    directory, manifest = grouped_cache
    row = cached_cnn_row(manifest["traces"][1])
    monkeypatch.setattr(evaluation, "CNN_GRID", [0.2, 0.5, 0.9])

    def scores(outputs, family, point):
        return (Scores(tp=49, fn=1) if point["threshold"] == 0.2 else Scores(tp=50)), []

    monkeypatch.setattr(evaluation, "score_outputs", scores)
    selected = evaluation.select_operating_point([row], "cnn")
    assert selected["operating_point"] == {"threshold": 0.2}
    assert selected["best_val_f1"] == 1
    assert selected["selected_val_f1"] < selected["best_val_f1"]


def test_original_unet_tie_selects_sorted_upper_median(grouped_cache, monkeypatch):
    _, manifest = grouped_cache
    row = cached_cnn_row(manifest["traces"][1])
    monkeypatch.setattr(evaluation, "THR_GRID", [0.1, 0.3, 0.5])
    monkeypatch.setattr(evaluation, "DUR_GRID", [30, 120])
    monkeypatch.setattr(evaluation, "score_outputs", lambda *a: (Scores(tp=2), []))
    selected = evaluation.select_operating_point([row], "unet")
    assert selected["operating_point"] == {"threshold": 0.3, "min_dur_sec": 120}


def test_test_records_cannot_be_used_for_operating_point_selection(grouped_cache):
    _, manifest = grouped_cache
    with pytest.raises(ValueError, match="only on validation"):
        evaluation.select_operating_point([cached_cnn_row(manifest["traces"][2])], "cnn")


def test_pipeline_locks_selection_before_opening_test_and_never_retunes(grouped_cache, monkeypatch):
    directory, manifest = grouped_cache
    output = directory / "results.json"
    selection_file = directory / "results.selection.json"
    monkeypatch.setattr(evaluation, "load_checkpoint", lambda *a: (object(), {"sha256": "checkpoint"}))
    sequence = []

    def infer(data_dir, loaded, split, *args):
        sequence.append(split)
        record = next(r for r in loaded["traces"] if r["split"] == split)
        if split == "test":
            locked = json.loads(selection_file.read_text())
            assert locked["test_waveforms_opened"] is False
            assert locked["selected"]["cnn"]["operating_point"]["threshold"] == 0.2
            # Test would favor another threshold, but cannot affect the locked decision.
            return [cached_cnn_row(record, probability=0.1)]
        assert not selection_file.exists()
        return [cached_cnn_row(record)]

    monkeypatch.setattr(evaluation, "infer_split", infer)
    result = evaluation.evaluate(directory, output, cnn="new.pt", device="cpu", baseline=False)
    assert sequence == ["val", "test"]
    assert result["results"]["cnn"]["operating_point"]["threshold"] == 0.2
    assert result["results"]["cnn"]["test_scores"]["fn"] == 1
    assert result["selection_sha256"] == evaluation.sha256_file(selection_file)
    with pytest.raises(FileExistsError):
        evaluation.evaluate(directory, output, cnn="new.pt", device="cpu", baseline=False)
    assert sequence == ["val", "test"]


def test_failed_selection_persistence_never_opens_test(grouped_cache, monkeypatch):
    directory, manifest = grouped_cache
    monkeypatch.setattr(evaluation, "load_checkpoint", lambda *a: (object(), {}))
    sequence = []

    def infer(data_dir, loaded, split, *args):
        sequence.append(split)
        return [cached_cnn_row(next(r for r in loaded["traces"] if r["split"] == split))]

    def fail_write(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr(evaluation, "infer_split", infer)
    monkeypatch.setattr(evaluation, "write_new_json", fail_write)
    with pytest.raises(OSError, match="disk unavailable"):
        evaluation.evaluate(directory, directory / "out.json", cnn="new.pt", device="cpu", baseline=False)
    assert sequence == ["val"]

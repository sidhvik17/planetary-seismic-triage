"""Versioned datasets and exact CPU continuation without training real models."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from planetseis import train
from planetseis.injection import InjectionDataset, NoisePool, build_template_bank
from planetseis.training_state import (dataset_provenance, prepare_run,
                                      save_checkpoint, training_config)


_spec = importlib.util.spec_from_file_location(
    "train_unet_test_module", Path(__file__).resolve().parents[1] / "scripts/train_unet.py")
train_unet = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_unet)


def make_dataset(root):
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({
        "benchmark_id": "lunar_grouped_v1", "body": "lunar"}))
    rng = np.random.default_rng(5)
    for split in ("train", "val"):
        continuous = root / "continuous" / split
        continuous.mkdir(parents=True)
        np.savez(continuous / f"{split}_group.npz", trace=np.zeros(32), rate=6.625, picks=[2.0])
        np.savez(root / f"{split}_windows.npz",
                 X=rng.normal(size=(8, 16)).astype(np.float32),
                 y=np.array([0, 1] * 4), offset=np.array([-1, 0.4] * 4))
    return root


def test_explicit_data_dir_reads_its_windows_and_never_falls_back(tmp_path):
    root = make_dataset(tmp_path / "dataset")
    X, y, _ = train.load_split("lunar", "train", root)
    assert X.shape == (8, 16) and y.sum() == 4
    (root / "val_windows.npz").unlink()
    with pytest.raises(ValueError, match="Missing val windows"):
        train.load_split("lunar", "val", root)


def test_manifest_hash_covers_exact_bytes_and_requires_held_out_val(tmp_path):
    root = make_dataset(tmp_path / "dataset")
    provenance = dataset_provenance("lunar", root)
    assert provenance["data_manifest_sha256"] == hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
    assert provenance["benchmark_id"] == "lunar_grouped_v1"
    next((root / "continuous" / "val").glob("*.npz")).unlink()
    with pytest.raises(ValueError, match="nonempty val"):
        dataset_provenance("lunar", root)


def test_union_picks_reach_template_bank_and_every_noise_guard(tmp_path):
    root = tmp_path / "dataset"
    continuous = root / "continuous" / "train"
    continuous.mkdir(parents=True)
    rate = 6.625
    picks = [4000.0, 12000.0]
    trace = np.random.default_rng(7).normal(size=120000).astype(np.float32)
    np.savez(continuous / "one_acquisition.npz", trace=trace, rate=rate, picks=picks)
    bank = build_template_bank("lunar", data_dir=root)
    assert len(bank) == 2
    assert {template.source for template in bank} == {"one_acquisition"}
    pool = NoisePool("lunar", data_dir=root)
    assert len(pool.traces) == 1
    for pick in picks:
        lo = int((pick - 900) * rate) - 8192
        hi = int((pick + 2700) * rate)
        assert not np.any((pool.valid_starts[0] >= lo) & (pool.valid_starts[0] < hi))
    dataset = InjectionDataset("lunar", "train", epoch_len=3, seed=17, data_dir=root)
    x1, y1 = dataset[0]
    x2, y2 = dataset[0]
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
    dataset.set_epoch(1)
    assert not torch.equal(x1, dataset[0][0])


def test_output_directory_is_never_silently_overwritten(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    marker = out / "log.csv"
    marker.write_text("existing work")
    with pytest.raises(ValueError, match="not empty"):
        prepare_run(out, False, {}, {})
    assert marker.read_text() == "existing work"


def test_resume_rejects_changed_manifest_or_recipe_and_allows_relocation(tmp_path):
    root = make_dataset(tmp_path / "dataset")
    provenance = dataset_provenance("lunar", root)
    args = SimpleNamespace(body="lunar", seed=7, epochs=60, data_dir=root,
                           out_dir=tmp_path / "old", resume=False, device="cpu")
    signature = training_config(args, {"window": 8192})
    args.out_dir, args.data_dir, args.resume = tmp_path / "new", tmp_path / "copied", True
    assert training_config(args, {"window": 8192}) == signature
    out = tmp_path / "run"
    out.mkdir()
    state = {"checkpoint_version": 1, "training_config": signature,
             "training_origin": "random_initialization", "initialized_from_scratch": True,
             **provenance}
    save_checkpoint(out / "last.pt", state)
    assert prepare_run(out, True, signature, provenance)["initialized_from_scratch"]
    with pytest.raises(ValueError, match="manifest differs"):
        prepare_run(out, True, signature, {**provenance, "data_manifest_sha256": "changed"})
    with pytest.raises(ValueError, match="configuration differs"):
        prepare_run(out, True, {**signature, "extra": "changed"}, provenance)
    save_checkpoint(out / "last.pt", {**state, "initialized_from_scratch": False})
    with pytest.raises(ValueError, match="initialized from scratch"):
        prepare_run(out, True, signature, provenance)


class TinyCNN(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(2, 2))

    def forward(self, x):
        out = self.layers(x[:, 0, :2])
        return out[:, 0], torch.sigmoid(out[:, 1]), out


class TinyUNet(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Dropout2d(0.2), torch.nn.Conv2d(2, 1, 1))

    def forward(self, x):
        return self.layers(x)


class TinyInjection(torch.utils.data.Dataset):
    def __init__(self, body, split, seed=None, data_dir=None, **kwargs):
        assert data_dir is not None and seed is not None
        self.seed, self.epoch, self.bank = seed, 0, ["template"]

    def __len__(self):
        return 4

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        rng = np.random.default_rng([self.seed, self.epoch, index])
        return (torch.from_numpy(rng.normal(size=(2, 8, 8)).astype(np.float32)),
                torch.from_numpy(rng.uniform(size=(1, 8, 8)).astype(np.float32)))


class InterruptedRun(Exception):
    pass


@pytest.mark.parametrize("family", ["cnn", "unet"])
def test_interrupted_resume_matches_uninterrupted_cpu_training(tmp_path, monkeypatch, family):
    root = make_dataset(tmp_path / "dataset")
    module = train if family == "cnn" else train_unet
    if family == "cnn":
        monkeypatch.setattr(module, "SeisCNN", TinyCNN)
        extra = []
    else:
        monkeypatch.setattr(module, "SpecUNet", TinyUNet)
        monkeypatch.setattr(module, "InjectionDataset", TinyInjection)
        extra = ["--epoch-len", "4", "--batch-size", "2"]
    uninterrupted, interrupted = tmp_path / "full", tmp_path / "resumed"

    def run(out, resume=False):
        args = ["trainer", "--body", "lunar", "--data-dir", str(root), "--out-dir", str(out),
                "--epochs", "3", "--seed", "19", "--device", "cpu", *extra]
        if resume:
            args.append("--resume")
        monkeypatch.setattr("sys.argv", args)
        module.main()

    run(uninterrupted)
    original_save = module.save_checkpoint

    def stop_after_first_epoch(path, state):
        original_save(path, state)
        if path.name == "last.pt" and state["epoch"] == 1:
            raise InterruptedRun()

    monkeypatch.setattr(module, "save_checkpoint", stop_after_first_epoch)
    with pytest.raises(InterruptedRun):
        run(interrupted)
    # A log write from an epoch that never checkpointed must not be duplicated.
    with (interrupted / "log.csv").open("a") as handle:
        handle.write("2,0,0,0,0,0\n" if family == "cnn" else "2,0,0,0,0\n")
    monkeypatch.setattr(module, "save_checkpoint", original_save)
    run(interrupted, resume=True)
    full = torch.load(uninterrupted / "last.pt", weights_only=False)
    resumed = torch.load(interrupted / "last.pt", weights_only=False)
    assert full["epoch"] == resumed["epoch"] == 3
    assert full["best_val"] == resumed["best_val"]
    assert full["scheduler"] == resumed["scheduler"]
    assert torch.equal(full["rng"]["torch"], resumed["rng"]["torch"])
    for name, values in full["model"].items():
        assert torch.equal(values, resumed["model"][name]), name
    best = torch.load(interrupted / "best.pt", weights_only=False)
    assert best["benchmark_id"] == "lunar_grouped_v1" and best["seed"] == 19
    assert best["training_origin"] == "random_initialization"
    assert best["initialized_from_scratch"] is True
    with (interrupted / "log.csv").open() as handle:
        assert [int(row["epoch"]) for row in csv.DictReader(handle)] == [1, 2, 3]


@pytest.mark.parametrize("flag", ["--finetune-from", "--hardneg"])
def test_corrected_unet_rejects_legacy_training_artifacts(tmp_path, monkeypatch, flag):
    root = make_dataset(tmp_path / "dataset")
    monkeypatch.setattr("sys.argv", ["trainer", "--body", "lunar", "--data-dir", str(root),
                                     "--out-dir", str(tmp_path / "run"), flag, "legacy.pt"])
    with pytest.raises(SystemExit, match="2"):
        train_unet.main()
    assert not (tmp_path / "run").exists()

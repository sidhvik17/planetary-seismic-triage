"""Dataset provenance and epoch-boundary checkpoints shared by both trainers."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def dataset_provenance(body: str, data_dir: Path | None) -> dict:
    if data_dir is None:
        return {"benchmark_id": None, "data_manifest_sha256": None}
    manifest_path = Path(data_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Explicit dataset requires a manifest: {manifest_path}")
    contents = manifest_path.read_bytes()
    manifest = json.loads(contents)
    if not isinstance(manifest, dict) or not manifest.get("benchmark_id"):
        raise ValueError("Dataset manifest must declare benchmark_id.")
    if manifest.get("body", body) != body:
        raise ValueError("Dataset manifest body does not match --body.")
    for split in ("train", "val"):
        if not any((Path(data_dir) / "continuous" / split).glob("*.npz")):
            raise ValueError(f"Explicit dataset requires a nonempty {split} continuous split.")
    return {
        "benchmark_id": manifest["benchmark_id"],
        "data_manifest_sha256": hashlib.sha256(contents).hexdigest(),
    }


def training_config(args, pipeline: dict) -> dict:
    """Settings affecting training; paths may move between machines/accounts."""
    runtime = {"data_dir", "out_dir", "resume", "device"}
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items() if key not in runtime}
    return {"arguments": config, "pipeline": pipeline}


def prepare_run(out: Path, resume: bool, config: dict, provenance: dict):
    out = Path(out)
    if not resume:
        if out.exists() and any(out.iterdir()):
            raise ValueError(f"Output directory is not empty: {out}. Use a new --out-dir or --resume.")
        out.mkdir(parents=True, exist_ok=True)
        return None
    path = out / "last.pt"
    if not path.is_file():
        raise ValueError(f"Cannot resume without an epoch checkpoint: {path}")
    # last.pt is this run's own resume file; its NumPy RNG state is not
    # loadable under weights_only=True. Model-weight files never use this path.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("checkpoint_version") != 1:
        raise ValueError("Checkpoint does not contain resumable training state.")
    if state.get("training_config") != config:
        raise ValueError("Resume training configuration differs from the saved run.")
    if any(state.get(key) != value for key, value in provenance.items()):
        raise ValueError("Resume dataset manifest differs from the saved run.")
    if provenance["benchmark_id"] == "lunar_grouped_v1" and (
        state.get("training_origin") != "random_initialization"
        or state.get("initialized_from_scratch") is not True
    ):
        raise ValueError("Corrected grouped training requires a run initialized from scratch.")
    return state


def capture_rng(dataset=None) -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if dataset is not None and hasattr(dataset, "rng"):
        state["dataset"] = dataset.rng.bit_generator.state
    return state


def restore_rng(state: dict, dataset=None, device: str = "cpu"):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.startswith("cuda") and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if dataset is not None and "dataset" in state:
        dataset.rng.bit_generator.state = state["dataset"]


def save_checkpoint(path: Path, state: dict):
    """Publish a complete checkpoint so an interrupted write cannot replace it."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def prepare_log(path: Path, fieldnames: list[str], completed_epoch: int):
    """Discard log rows newer than last.pt before replaying an interrupted epoch."""
    rows = []
    if completed_epoch and path.exists():
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != fieldnames:
                raise ValueError("Training log columns differ from this trainer.")
            rows = [row for row in reader if int(row["epoch"]) <= completed_epoch]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_metadata(args, config: dict, provenance: dict, initialized_from_scratch=True):
    return {
        "checkpoint_version": 1,
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
        "training_config": config,
        "initialized_from_scratch": initialized_from_scratch,
        "training_origin": "random_initialization" if initialized_from_scratch else "finetune",
        **provenance,
    }

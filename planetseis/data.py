"""Locate traces and catalogs inside the Space Apps 2024 data packet.

Packet layout (verified after extraction):
    data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv
    data/lunar/training/data/S12_GradeA/*.{csv,mseed}
    data/lunar/test/data/S12_GradeB, S15_GradeA, ... (uncatalogued)
    data/mars/training/catalogs/Mars_InSight_training_catalog_final.csv
    data/mars/training/data/*.{csv,mseed}
    data/mars/test/data/*.{csv,mseed}

Ground truth = catalogued relative arrival times per file (PRD FR-2).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import PACKET_ROOT, SEED


@dataclass
class LabeledTrace:
    body: str                 # "lunar" | "mars"
    trace_path: Path          # miniSEED preferred over CSV
    arrival_rel_sec: list     # catalog picks, seconds from trace start
    evids: list


def _catalog_files(body: str) -> list[Path]:
    cat_dir = PACKET_ROOT / "data" / body / "training" / "catalogs"
    if not cat_dir.exists():
        return []
    cats = sorted(cat_dir.glob("*.csv"))
    # Mars ships two catalogs; the non-"final" one uses a different filename
    # scheme and lacks time_rel. Use only the curated final catalogs.
    finals = [c for c in cats if "final" in c.name.lower()]
    return finals or cats


def _find_trace(data_dir: Path, stem: str) -> Path | None:
    for ext in (".mseed", ".sac", ".csv"):
        hits = list(data_dir.rglob(stem + ext))
        if hits:
            return hits[0]
    return None


def load_labeled_traces(body: str) -> list[LabeledTrace]:
    """Parse training catalogs into one LabeledTrace per data file."""
    data_dir = PACKET_ROOT / "data" / body / "training" / "data"
    by_file: dict[str, LabeledTrace] = {}
    for cat_path in _catalog_files(body):
        cat = pd.read_csv(cat_path)
        fname_col = next(c for c in cat.columns if "filename" in c.lower())
        rel_col = next(c for c in cat.columns if "rel" in c.lower())
        evid_col = next((c for c in cat.columns if "evid" in c.lower()), None)
        for _, row in cat.iterrows():
            stem = str(row[fname_col]).replace(".csv", "").replace(".mseed", "")
            if stem not in by_file:
                path = _find_trace(data_dir, stem)
                if path is None:
                    continue  # catalog rows without data files exist; skip and log
                by_file[stem] = LabeledTrace(body, path, [], [])
            by_file[stem].arrival_rel_sec.append(float(row[rel_col]))
            by_file[stem].evids.append(str(row[evid_col]) if evid_col else stem)
    return list(by_file.values())


def split_traces(
    traces: list[LabeledTrace], frac_train=0.6, frac_val=0.15
) -> tuple[list, list, list]:
    """Historical filename split, retained to reproduce frozen experiments.

    Distinct event filenames can contain the same acquisition. The lunar
    manifest has confirmed train/test overlap; see scripts/audit_splits.py.
    New independent benchmarks require acquisition grouping and merged picks.
    """
    rng = random.Random(SEED)
    shuffled = sorted(traces, key=lambda t: t.trace_path.name)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = max(1, round(n * frac_train))
    n_val = max(1, round(n * frac_val)) if n > 3 else 0
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test

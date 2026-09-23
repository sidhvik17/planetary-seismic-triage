"""Acquisition-aware lunar benchmark construction, separate from frozen v1.

The split policy is metadata-only: connected time overlaps or identical raw
waveforms stay together; historical test > val > train; new sources -> train.
Exact acquisition copies are collapsed and catalog UTC picks are unioned.
Distinct partial spans keep their original preprocessing within one split.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import obspy
import pandas as pd

from .config import CODA_SEC, DEFAULT as CFG, SEED
from .preprocessing import load_trace, preprocess
from .windows import make_windows, positives_around_pick

BENCHMARK_ID = "lunar_grouped_v1"
SPLITS = ("train", "val", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trace_hash(trace: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(trace).tobytes()).hexdigest()


def interval_overlap(left: dict, right: dict) -> bool:
    if left["channel"] != right["channel"]:
        return False
    return min(float(obspy.UTCDateTime(left["end_time"])),
               float(obspy.UTCDateTime(right["end_time"]))) - max(
        float(obspy.UTCDateTime(left["start_time"])),
        float(obspy.UTCDateTime(right["start_time"]))) > 1e-6


def acquisition_components(sources: list[dict]) -> tuple[list[list[int]], list[dict]]:
    """Transitive interval/hash components; no label values or scores used."""
    parents = list(range(len(sources)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    links = []
    for i, left in enumerate(sources):
        for j in range(i):
            right = sources[j]
            identical = (left["raw_trace_sha256"] == right["raw_trace_sha256"]
                         and left["n_samples"] == right["n_samples"]
                         and left["rate_hz"] == right["rate_hz"])
            overlapping = interval_overlap(left, right)
            if identical or overlapping:
                parents[root(i)] = root(j)
                links.append({"left": right["filename"], "right": left["filename"],
                              "identical_samples": identical,
                              "interval_overlap": overlapping})
    components = {}
    for i in range(len(sources)):
        components.setdefault(root(i), []).append(i)
    return sorted(components.values(), key=lambda ids: sources[ids[0]]["filename"]), links


def resolve_catalog_path(data_dir: Path, filename: str) -> tuple[Path, str]:
    """Recover a unique event-ID alias only within the same channel prefix."""
    exact = sorted(data_dir.rglob(filename + ".mseed"))
    if len(exact) == 1:
        return exact[0], "exact_filename"
    if len(exact) > 1:
        raise ValueError(f"Ambiguous catalog filename: {filename}")
    prefix, evid = filename.rsplit("_evid", 1)
    channel_prefix = ".".join(prefix.split(".")[:4])
    matches = sorted(data_dir.rglob(f"{channel_prefix}.*_evid{evid}.mseed"))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one same-channel event-ID match for {filename}; "
                         f"found {len(matches)}")
    return matches[0], "unique_channel_event_id"


def collect_sources(packet_root: Path, historical_path: Path) -> tuple[list[dict], dict]:
    historical = json.loads(historical_path.read_text(encoding="utf-8"))
    membership = {}
    for split in SPLITS:
        for filename in historical[split]:
            if filename in membership:
                raise ValueError(f"Filename repeated in historical splits: {filename}")
            membership[filename] = split
    catalog_path = packet_root / "data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv"
    data_dir = packet_root / "data/lunar/training/data"
    catalog = pd.read_csv(catalog_path)
    abs_col = next(c for c in catalog.columns if c.startswith("time_abs"))
    rel_col = next(c for c in catalog.columns if c.startswith("time_rel"))
    sources = []
    for row in catalog.to_dict("records"):
        filename = str(row["filename"])
        path, resolution = resolve_catalog_path(data_dir, filename)
        headers = obspy.read(str(path), headonly=True)
        if len({t.id for t in headers}) != 1:
            raise ValueError(f"Ambiguous acquisition channel: {path}")
        raw, rate, start = load_trace(path)
        if start is None:
            raise ValueError(f"Missing acquisition start: {path}")
        pick = obspy.UTCDateTime(str(row[abs_col]))
        relative = float(pick - start)
        if not 0 <= relative < len(raw) / rate:
            raise ValueError(f"Catalog UTC pick lies outside {path.name}")
        sources.append({
            "filename": path.name, "source_file": path.relative_to(packet_root).as_posix(),
            "catalog_filename": filename, "resolution": resolution,
            "historical_split": membership.get(path.name, "unassigned"),
            "channel": headers[0].id, "start_time": str(start),
            "end_time": str(start + len(raw) / rate), "n_samples": len(raw), "rate_hz": rate,
            "raw_trace_sha256": trace_hash(np.asarray(raw, dtype="<f8")),
            "source_file_sha256": sha256_file(path),
            "evid": str(row["evid"]), "pick_utc": str(pick),
            "catalog_relative_sec": float(row[rel_col]), "actual_relative_sec": relative,
        })
    sources.sort(key=lambda s: s["filename"])
    if set(membership) - {s["filename"] for s in sources}:
        raise ValueError("Some historical filenames were not resolved in the catalog")
    if len({s["evid"] for s in sources}) != len(sources):
        raise ValueError("Repeated event IDs require explicit catalog reconciliation")
    return sources, {
        "catalog_file": catalog_path.relative_to(packet_root).as_posix(),
        "catalog_sha256": sha256_file(catalog_path),
        "historical_manifest_sha256": sha256_file(historical_path),
    }


def plan_manifest(sources: list[dict], source_info: dict) -> dict:
    components, links = acquisition_components(sources)
    manifest = {
        "schema_version": 1, "benchmark_id": BENCHMARK_ID, "body": "lunar",
        "policy": {
            "grouping": "connected half-open UTC intervals on one channel OR identical raw samples/rate",
            "holdout_precedence": ["test", "val", "train"],
            "unassigned_destination": "train", "picks": "catalog UTC minus actual trace start; union per span",
            "partial_spans": "kept separate within one split", "seed": SEED,
        },
        "preprocessing": asdict(CFG.preproc), "window_config": asdict(CFG.window),
        "source_provenance": source_info, "sources": sources, "grouping_links": links,
        "groups": [], "traces": [], "splits": {s: [] for s in SPLITS},
    }
    for ids in components:
        members = [sources[i] for i in ids]
        group_id = "acq_" + hashlib.sha256("\n".join(
            m["filename"] for m in members).encode()).hexdigest()[:16]
        old_splits = {m["historical_split"] for m in members}
        split = next((s for s in ("test", "val", "train") if s in old_splits), "train")
        unique_spans = {}
        for member in members:
            key = (member["channel"], member["start_time"], member["end_time"],
                   member["rate_hz"], member["raw_trace_sha256"])
            unique_spans.setdefault(key, []).append(member)
        trace_ids = []
        for copies in unique_spans.values():
            canonical = copies[0]
            trace_id = Path(canonical["filename"]).stem
            start = obspy.UTCDateTime(canonical["start_time"])
            duration = canonical["n_samples"] / canonical["rate_hz"]
            picks = {}
            for member in members:
                relative = float(obspy.UTCDateTime(member["pick_utc"]) - start)
                if 0 <= relative < duration:
                    picks[member["evid"]] = {"evid": member["evid"],
                                              "utc": member["pick_utc"], "relative_sec": relative}
            ordered = sorted(picks.values(), key=lambda p: p["relative_sec"])
            manifest["traces"].append({
                "id": trace_id, "group_id": group_id, "split": split,
                "channel": canonical["channel"], "start_time": canonical["start_time"],
                "end_time": canonical["end_time"], "n_samples": canonical["n_samples"],
                "rate_hz": canonical["rate_hz"],
                "source_files": [c["source_file"] for c in copies],
                "raw_trace_sha256": canonical["raw_trace_sha256"],
                "picks": ordered, "picks_rel_sec": [p["relative_sec"] for p in ordered],
                "cache_file": f"continuous/{split}/{trace_id}.npz",
            })
            trace_ids.append(trace_id)
            manifest["splits"][split].append(trace_id)
        manifest["groups"].append({"id": group_id, "split": split,
                                   "trace_ids": sorted(trace_ids),
                                   "source_files": [m["source_file"] for m in members]})
    manifest["traces"].sort(key=lambda t: t["id"])
    for split in SPLITS:
        manifest["splits"][split].sort()
    return manifest


def audit_manifest(manifest: dict, cache_dir: Path | None = None) -> dict:
    """Fail closed on grouping, label loss, or any cross-split time/hash reuse."""
    traces = manifest["traces"]
    if len({t["id"] for t in traces}) != len(traces):
        raise ValueError("Duplicate trace ID")
    groups = {g["id"]: g for g in manifest["groups"]}
    if len(groups) != len(manifest["groups"]):
        raise ValueError("Duplicate group ID")
    assigned = set()
    event_counts = {}
    source_counts = {}
    for trace in traces:
        group = groups[trace["group_id"]]
        if trace["split"] not in SPLITS or group["split"] != trace["split"]:
            raise ValueError("Acquisition group split mismatch")
        if trace["id"] not in group["trace_ids"]:
            raise ValueError("Trace absent from acquisition group")
        if trace["picks_rel_sec"] != sorted(set(trace["picks_rel_sec"])):
            raise ValueError("Picks must be sorted and unique")
        for pick in trace["picks"]:
            event_counts[pick["evid"]] = event_counts.get(pick["evid"], 0) + 1
        for source in trace["source_files"]:
            source_counts[source] = source_counts.get(source, 0) + 1
        if cache_dir is not None:
            with np.load(cache_dir / trace["cache_file"], allow_pickle=False) as z:
                actual = z["trace"]
                if actual.ndim != 1 or not np.isfinite(actual).all():
                    raise ValueError("Invalid processed waveform")
                if len(actual) != trace["n_samples"] or float(z["rate"]) != trace["rate_hz"]:
                    raise ValueError("Waveform shape/rate differs from manifest")
                if trace_hash(actual) != trace["trace_sha256"]:
                    raise ValueError("Processed waveform hash differs from manifest")
                if not np.array_equal(z["picks"], trace["picks_rel_sec"]):
                    raise ValueError("Cached pick union differs from manifest")
        for other in traces:
            if other["split"] == trace["split"]:
                continue
            if interval_overlap(trace, other):
                raise ValueError("Cross-split acquisition interval overlap")
            if trace["raw_trace_sha256"] == other["raw_trace_sha256"]:
                raise ValueError("Cross-split raw waveform duplicate")
            if trace.get("trace_sha256") and trace["trace_sha256"] == other.get("trace_sha256"):
                raise ValueError("Cross-split processed waveform duplicate")
    for split in SPLITS:
        expected = {t["id"] for t in traces if t["split"] == split}
        if set(manifest["splits"][split]) != expected or not expected:
            raise ValueError(f"Incomplete or empty {split} split")
        if assigned & expected:
            raise ValueError("Trace occurs in multiple splits")
        assigned |= expected
        if cache_dir is not None:
            actual = {p.stem for p in (cache_dir / "continuous" / split).glob("*.npz")}
            if actual != expected:
                raise ValueError(f"Unexpected or missing cached {split} files")
    expected_events = {s["evid"] for s in manifest["sources"]}
    if set(event_counts) != expected_events or any(n != 1 for n in event_counts.values()):
        raise ValueError("Lost or double-counted catalog event; reconcile spans before evaluation")
    expected_sources = {s["source_file"] for s in manifest["sources"]}
    if set(source_counts) != expected_sources or any(n != 1 for n in source_counts.values()):
        raise ValueError("Lost or duplicated source waveform")
    return {
        "status": "passed", "cross_split_interval_overlaps": 0,
        "cross_split_raw_duplicates": 0, "cross_split_processed_duplicates": 0,
        "catalog_events_accounted_for": len(event_counts), "sources_accounted_for": len(source_counts),
        "splits": {s: {
            "groups": sum(g["split"] == s for g in groups.values()),
            "traces": sum(t["split"] == s for t in traces),
            "events": sum(len(t["picks_rel_sec"]) for t in traces if t["split"] == s),
        } for s in SPLITS},
        "scope": "local Grade-A sources, acquisition intervals and complete-waveform identity; not proof against all latent dependence",
    }


def build_cache(manifest: dict, packet_root: Path, cache_dir: Path) -> dict:
    """Build only into a new directory; publish manifest last as completion marker."""
    if cache_dir.exists() and any(cache_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty cache: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    collected = {s: ([], [], []) for s in ("train", "val")}
    for entry in manifest["traces"]:
        raw, rate, start = load_trace(packet_root / entry["source_files"][0])
        if (str(start) != entry["start_time"] or rate != entry["rate_hz"] or
                trace_hash(np.asarray(raw, dtype="<f8")) != entry["raw_trace_sha256"]):
            raise ValueError("Source changed between inventory and cache construction")
        trace, rate = preprocess(raw, rate, CFG.preproc)
        entry["n_samples"], entry["rate_hz"] = len(trace), rate
        entry["trace_sha256"] = trace_hash(trace)
        output = cache_dir / entry["cache_file"]
        output.parent.mkdir(parents=True, exist_ok=True)
        picks = entry["picks_rel_sec"]
        np.savez_compressed(output, trace=trace, rate=rate,
                            picks=np.array(picks, dtype=np.float64),
                            group_id=entry["group_id"], start_time=entry["start_time"], channel=entry["channel"])
        print(f"{entry['split']:5s} {entry['id']} picks={len(picks)}", flush=True)
        if entry["split"] == "test":
            continue
        wins = make_windows(trace, rate, picks, CFG.window)
        positives = [w for w in wins if w.label == 1]
        win_sec = CFG.window.n_samples / rate
        negatives = [w for w in wins if w.label == 0 and not any(
            w.start_sec < p + CODA_SEC["lunar"] and w.start_sec + win_sec > p for p in picks)]
        for pick in picks:
            positives += positives_around_pick(trace, rate, pick, CFG.window, n_shifts=12, rng=rng)
        n_neg = min(len(negatives), int(np.ceil(len(positives) * CFG.train.neg_pos_ratio)) or 8)
        negatives = [negatives[i] for i in rng.choice(len(negatives), size=n_neg, replace=False)] if negatives else []
        X, y, offsets = collected[entry["split"]]
        for window in positives + negatives:
            # Copy releases the large parent trace once this acquisition is done.
            X.append(window.data.copy())
            y.append(window.label)
            offsets.append(window.offset_frac)
    manifest["windows"] = {}
    for split, (X, y, offsets) in collected.items():
        if not X or not any(y) or all(y):
            raise ValueError(f"{split} needs positive and negative training windows")
        path = cache_dir / f"{split}_windows.npz"
        np.savez_compressed(path, X=np.stack(X), y=np.array(y, dtype=np.int64),
                            offset=np.array(offsets, dtype=np.float32))
        manifest["windows"][split] = {"file": path.name, "sha256": sha256_file(path),
                                      "count": len(y), "positive": sum(y), "negative": len(y) - sum(y)}
    report = audit_manifest(manifest, cache_dir)
    manifest["audit"] = report
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return report

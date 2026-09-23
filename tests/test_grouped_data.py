"""Acquisition leakage and pick-union regressions for the corrected benchmark."""
import copy
from pathlib import Path

import numpy as np
import obspy
import pytest

from planetseis.grouped_data import (acquisition_components, audit_manifest,
                                    collect_sources, plan_manifest, resolve_catalog_path)


def source(name, start, end, split, evid, pick, digest=None):
    return {
        "filename": name + ".mseed", "source_file": name + ".mseed",
        "channel": "XA.S12.00.MHZ", "start_time": str(obspy.UTCDateTime(start)),
        "end_time": str(obspy.UTCDateTime(end)), "n_samples": int((end - start) * 10),
        "rate_hz": 10.0, "raw_trace_sha256": digest or name,
        "historical_split": split, "evid": evid, "pick_utc": str(obspy.UTCDateTime(pick)),
    }


def fixture_sources():
    return [source("a", 0, 10, "train", "e1", 4, "same"),
            source("b", 0, 10, "test", "e2", 8, "same"),
            source("c", 9, 19, "train", "e3", 15),
            source("d", 30, 40, "val", "e4", 35),
            source("e", 50, 60, "train", "e5", 55)]


def test_exact_copies_union_labels_and_partial_overlap_inherits_holdout():
    manifest = plan_manifest(fixture_sources(), {})
    report = audit_manifest(manifest)
    assert report["splits"]["test"] == {"groups": 1, "traces": 2, "events": 3}
    test = [t for t in manifest["traces"] if t["split"] == "test"]
    assert test[0]["picks_rel_sec"] == [4.0, 8.0]
    assert len(test[0]["source_files"]) == 2
    assert test[1]["picks_rel_sec"] == [6.0]
    assert test[0]["group_id"] == test[1]["group_id"]


def test_components_are_transitive_and_touching_exclusive_boundaries_do_not_overlap():
    sources = [source("a", 0, 10, "train", "1", 3),
               source("b", 9, 19, "train", "2", 14),
               source("c", 18, 28, "test", "3", 24),
               source("d", 28, 38, "val", "4", 34)]
    components, _ = acquisition_components(sources)
    assert components == [[0, 1, 2], [3]]


def test_identical_samples_group_even_if_metadata_times_differ():
    sources = [source("a", 0, 10, "train", "1", 3, "same"),
               source("b", 100, 110, "test", "2", 104, "same")]
    assert acquisition_components(sources)[0] == [[0, 1]]


def test_unassigned_sources_go_to_train_without_using_labels():
    items = fixture_sources()
    items[-1]["historical_split"] = "unassigned"
    assert plan_manifest(items, {})["splits"]["train"] == ["e"]


def test_audit_rejects_reintroduced_cross_split_overlap():
    manifest = plan_manifest(fixture_sources(), {})
    trace = next(t for t in manifest["traces"] if t["id"] == "c")
    trace["split"] = "train"
    trace["group_id"] = "bad"
    manifest["groups"].append({"id": "bad", "split": "train", "trace_ids": ["c"]})
    with pytest.raises(ValueError, match="Cross-split acquisition"):
        audit_manifest(manifest)


def test_audit_rejects_lost_labels_and_double_counted_overlap_events():
    manifest = plan_manifest(fixture_sources(), {})
    original = copy.deepcopy(manifest)
    manifest["traces"][0]["picks"] = manifest["traces"][0]["picks"][:1]
    with pytest.raises(ValueError, match="Lost or double-counted"):
        audit_manifest(manifest)
    original["traces"][1]["picks"].append(original["traces"][0]["picks"][0])
    with pytest.raises(ValueError, match="Lost or double-counted"):
        audit_manifest(original)


def test_event_alias_recovery_is_unique_and_same_channel(tmp_path):
    recovered = tmp_path / "xa.s12.00.mhz.1971-04-13HR02_evid00029.mseed"
    recovered.touch()
    name = "xa.s12.00.mhz.1971-04-13HR00_evid00029"
    assert resolve_catalog_path(tmp_path, name) == (recovered, "unique_channel_event_id")
    (tmp_path / "xa.s12.00.mhz.1971-04-13HR03_evid00029.mseed").touch()
    with pytest.raises(ValueError, match="exactly one"):
        resolve_catalog_path(tmp_path, name)


def test_catalog_relative_pick_uses_actual_waveform_utc_start(tmp_path):
    data = tmp_path / "data/lunar/training/data"
    cat_dir = tmp_path / "data/lunar/training/catalogs"
    data.mkdir(parents=True)
    cat_dir.mkdir(parents=True)
    actual = "xa.s12.00.mhz.1971-04-13HR02_evid00029"
    tr = obspy.Trace(np.arange(1000, dtype=np.float64), header={
        "network": "XA", "station": "S12", "location": "00", "channel": "MHZ",
        "sampling_rate": 10, "starttime": obspy.UTCDateTime("1971-04-13T02:42:48Z")})
    tr.write(str(data / (actual + ".mseed")), format="MSEED")
    (cat_dir / "apollo12_catalog_GradeA_final.csv").write_text(
        "filename,time_abs,time_rel,evid\n"
        "xa.s12.00.mhz.1971-04-13HR00_evid00029,1971-04-13T02:43:00,9780,evid00029\n")
    historical = tmp_path / "historical.json"
    historical.write_text('{"train":[],"val":[],"test":[]}')
    records, _ = collect_sources(tmp_path, historical)
    assert records[0]["actual_relative_sec"] == 12.0
    assert records[0]["catalog_relative_sec"] == 9780.0
    assert records[0]["historical_split"] == "unassigned"

"""Integrity checks must distinguish suspicious filenames from waveform proof."""
import importlib.util
import json
from pathlib import Path

import numpy as np

# scripts/ is deliberately not part of the installed planetseis package.
_spec = importlib.util.spec_from_file_location(
    "audit_splits", Path(__file__).resolve().parents[1] / "scripts/audit_splits.py"
)
_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)
acquisition_stem, audit_manifest, main = _audit.acquisition_stem, _audit.audit_manifest, _audit.main


def packet_name(event, day="1972-07-17"):
    return f"xa.s12.00.mhz.{day}HR00_evid{event}.mseed"


def write_cache(cache, split, filename, trace):
    directory = cache / split
    directory.mkdir(exist_ok=True)
    np.savez(directory / filename.replace(".mseed", ".npz"), trace=trace, rate=6.625)


def test_grouped_duplicates_distinguish_within_and_cross_split():
    manifest = {
        "train": [packet_name("001"), packet_name("002", "1973-06-05"), packet_name("003", "1973-06-05")],
        "test": [packet_name("004")],
    }
    report = audit_manifest(manifest)
    assert report["summary"]["duplicate_acquisitions"] == 2
    assert report["summary"]["cross_split_acquisitions"] == 1
    assert report["summary"]["confirmed_identical_cross_split_pairs"] == 0
    pair = report["duplicate_acquisitions"][0]["comparisons"][0]
    assert pair["status"] == "unverified"


def test_clean_split_and_unrecognizable_names_do_not_invent_overlap():
    report = audit_manifest({
        "train": [packet_name("001"), "XB.ELYSE.02.BHV.S0133a"],
        "test": [packet_name("002", "1973-06-05")],
    })
    assert report["summary"]["status"] == "no_candidates_detected"
    assert report["summary"]["unchecked_filenames"] == 1
    assert acquisition_stem(packet_name("003", "1973-02-30")) is None


def test_cache_comparison_confirms_identical_waveforms(tmp_path):
    left, right = packet_name("001"), packet_name("002")
    for split, filename in (("train", left), ("test", right)):
        write_cache(tmp_path, split, filename, np.arange(32, dtype=np.float32))
    report = audit_manifest({"train": [left], "test": [right]}, tmp_path)
    assert report["summary"]["confirmed_identical_cross_split_pairs"] == 1
    pair = report["duplicate_acquisitions"][0]["comparisons"][0]
    assert pair["status"] == "confirmed_identical"
    assert pair["waveforms"][0]["trace_sha256"] == pair["waveforms"][1]["trace_sha256"]
    assert str(tmp_path) not in json.dumps(report)


def test_different_arrays_do_not_confirm_name_candidate(tmp_path):
    left, right = packet_name("001"), packet_name("002")
    write_cache(tmp_path, "train", left, np.zeros(8))
    write_cache(tmp_path, "test", right, np.ones(8))
    report = audit_manifest({"train": [left], "test": [right]}, tmp_path)
    assert report["summary"]["cross_split_acquisitions"] == 1
    assert report["summary"]["confirmed_identical_cross_split_pairs"] == 0
    assert report["duplicate_acquisitions"][0]["comparisons"][0]["status"] == "not_identical"


def test_missing_cache_stays_unverified(tmp_path):
    report = audit_manifest({"train": [packet_name("001")], "test": [packet_name("002")]}, tmp_path)
    assert report["duplicate_acquisitions"][0]["comparisons"][0]["status"] == "unverified"


def test_extensionless_manifest_names_use_the_same_acquisition_and_cache(tmp_path):
    left, right = packet_name("001"), packet_name("002")
    for split, filename in (("train", left), ("test", right)):
        write_cache(tmp_path, split, filename, np.arange(32, dtype=np.float32))
    report = audit_manifest({"train": [left.removesuffix(".mseed")], "test": [right]}, tmp_path)
    assert report["summary"]["unchecked_filenames"] == 0
    assert report["summary"]["confirmed_identical_cross_split_pairs"] == 1


def test_cli_writes_portable_report_and_fails_on_overlap(tmp_path, capsys):
    manifest = tmp_path / "splits.json"
    output = tmp_path / "audit.json"
    manifest.write_text(json.dumps({"train": [packet_name("001")], "test": [packet_name("002")]}))
    assert main(["--manifest", str(manifest), "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["manifest"] == "splits.json"
    assert json.loads(capsys.readouterr().out) == report
    manifest.write_text(json.dumps({"train": [packet_name("001")], "test": []}))
    assert main(["--manifest", str(manifest)]) == 0

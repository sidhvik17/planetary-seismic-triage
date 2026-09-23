"""STA/LTA baseline on lunar_grouped_v1 with an extended threshold grid.

    python scripts/sta_lta_grouped.py --data-dir data/cache/lunar_grouped_v1

The locked evaluations selected thr_on = 7.0, the top of the historical grid
(2-7), on both the historical and the corrected split, so the baseline may
have been under-tuned. This reruns the same detector (60 s / 600 s windows,
thr_off = max(1.2, thr_on / 2), 1800 s dead time), the same scorer and the
same tie rule (first threshold at the maximum validation F1) over a wider
grid. The selection is written to an exclusive *.selection.json before any
test waveform is opened. It is a new result: the locked evaluation files are
not touched, and the extended grid was fixed before looking at test scores.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_grouped import (infer_split, load_manifest, score_outputs,
                                      sha256_file, write_new_json)
from planetseis.config import CODA_SEC, DEFAULT as CFG, PROJECT_ROOT

EXTENDED_GRID = [2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path,
                    default=PROJECT_ROOT / "results" / "sta_lta_extended_lunar_grouped_v1.json")
    args = ap.parse_args(argv)
    selection_path = args.output.with_name(args.output.stem + ".selection.json")
    if args.output.exists() or selection_path.exists():
        sys.exit("result or selection already exists; refusing to retune or overwrite")
    manifest, manifest_hash = load_manifest(args.data_dir)

    validation = infer_split(args.data_dir, manifest, "val", {}, "cpu", EXTENDED_GRID)
    sweep = []
    for threshold in EXTENDED_GRID:
        scores, _ = score_outputs(validation, "sta_lta", {"thr_on": threshold})
        sweep.append({"thr_on": threshold, "f1_unrounded": scores.f1, "scores": scores.as_dict()})
    best = max(row["f1_unrounded"] for row in sweep)
    selected = next(row for row in sweep if row["f1_unrounded"] == best)
    common = {
        "benchmark_id": manifest["benchmark_id"], "data_manifest_sha256": manifest_hash,
        "method": "STA/LTA (obspy classic_sta_lta, 60 s / 600 s, thr_off = max(1.2, thr_on/2))",
        "threshold_grid": EXTENDED_GRID, "tie_rule": "first threshold at maximum validation F1",
        "tolerance_sec": CFG.window.match_tolerance_sec, "suppress_sec": CODA_SEC["lunar"],
        "code_sha256": {name: sha256_file(PROJECT_ROOT / name) for name in
                        ("scripts/sta_lta_grouped.py", "scripts/evaluate_grouped.py",
                         "planetseis/baseline.py", "planetseis/evaluate.py")},
    }
    write_new_json(selection_path, {
        **common, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_sweep": sweep, "selected_thr_on": selected["thr_on"],
        "selected_val_f1": best, "test_waveforms_opened": False})
    print(f"validation selected thr_on={selected['thr_on']} (F1 {best:.4f}); locked "
          f"{selection_path.name}", flush=True)

    test = infer_split(args.data_dir, manifest, "test", {}, "cpu", [selected["thr_on"]])
    scores, per_trace = score_outputs(test, "sta_lta", {"thr_on": selected["thr_on"]})
    write_new_json(args.output, {
        **common, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_file": selection_path.name, "selection_sha256": sha256_file(selection_path),
        "operating_point": {"thr_on": selected["thr_on"]}, "selected_val_f1": best,
        "test_scores": scores.as_dict(), "test_per_trace": per_trace,
        "interpretation": "Extended-grid STA/LTA baseline on lunar_grouped_v1; the "
                          "locked evaluation's 2-7 grid result (F1 0.168) is unchanged.",
    })
    print(f"test: {json.dumps(scores.as_dict())}")


if __name__ == "__main__":
    main()

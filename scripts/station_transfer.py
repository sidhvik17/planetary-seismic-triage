"""Beyond-PRD studies: cross-station generalization + per-type recall.

1. Cross-station: the model is trained only on Apollo 12 (S12). The packet's
   lunar test folders hold uncatalogued-but-curated files from S12 (Grade B)
   and stations S15/S16 (Grades A+B) — each file is documented to contain one
   catalogued event (evid embedded in the filename), but arrival times are
   not shipped. So we report the DETECTION RATE (fraction of files where the
   model fires >= 1 detection at the lunar operating point) as a weak-label
   measure of generalization: same station harder events (S12 B), and same
   body but different station/instrument placement (S15/S16). This fills the
   transfer ladder's middle rung between same-station (F1 0.54) and
   cross-body (F1 ~ 0).

2. Per-type recall on the catalogued S12 test split (impact vs deep vs
   shallow moonquakes) at the same operating point.

Writes results/station_transfer.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import (CODA_SEC, DEFAULT as CFG, DATA_CACHE, PACKET_ROOT,
                               PROJECT_ROOT)
from planetseis.detect import detect_events
from planetseis.model import ARCHS, SeisCNN
from planetseis.preprocessing import load_trace, preprocess

THR = 0.99  # lunar operating point


def get_model(device):
    ck = torch.load(PROJECT_ROOT / "runs" / "lunar" / "best.pt",
                    map_location=device, weights_only=False)
    m = SeisCNN(channels=ARCHS[ck.get("arch", "base")]).to(device)
    m.load_state_dict(ck["model"])
    return m


def station_groups(model, device):
    root = PACKET_ROOT / "data" / "lunar" / "test" / "data"
    out = {}
    for grp in sorted(d for d in root.iterdir() if d.is_dir()):
        files = sorted(grp.glob("*.mseed"))
        n_hit, det_counts = 0, []
        for f in files:
            try:
                raw, rate, _ = load_trace(f)
                proc, prate = preprocess(raw, rate, CFG.preproc)
                dets, _, _ = detect_events(model, proc, prate, CFG, THR, device,
                                           suppress_sec=CODA_SEC["lunar"])
            except Exception as e:
                print(f"  skip {f.name}: {e}")
                continue
            det_counts.append(len(dets))
            n_hit += bool(dets)
        out[grp.name] = {
            "n_files": len(det_counts),
            "files_with_detection": n_hit,
            "detection_rate": round(n_hit / max(len(det_counts), 1), 3),
            "mean_detections_per_file": round(float(np.mean(det_counts)), 2),
        }
        print(grp.name, out[grp.name])
    return out


def per_type_recall(model, device):
    cat = pd.read_csv(PACKET_ROOT / "data" / "lunar" / "training" / "catalogs" /
                      "apollo12_catalog_GradeA_final.csv")
    fcol = next(c for c in cat.columns if "filename" in c.lower())
    types = dict(zip(cat[fcol].astype(str), cat["mq_type"].astype(str)))
    hits: dict[str, list[bool]] = {}
    for p in sorted((DATA_CACHE / "lunar" / "continuous" / "test").glob("*.npz")):
        z = np.load(p)
        trace, rate, picks = z["trace"], float(z["rate"]), list(z["picks"])
        dets, _, _ = detect_events(model, trace, rate, CFG, THR, device,
                                   suppress_sec=CODA_SEC["lunar"])
        times = [d.time_sec for d in dets]
        ev_type = types.get(p.stem, "unknown")
        for pk in picks:
            hit = any(abs(t - pk) <= CFG.window.match_tolerance_sec for t in times)
            hits.setdefault(ev_type, []).append(hit)
    return {k: {"n": len(v), "recall": round(float(np.mean(v)), 3)}
            for k, v in sorted(hits.items())}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(device)
    print("=== cross-station detection rate (weak labels) ===")
    stations = station_groups(model, device)
    print("=== per-type recall (S12 test split) ===")
    types = per_type_recall(model, device)
    print(json.dumps(types, indent=2))
    (PROJECT_ROOT / "results" / "station_transfer.json").write_text(json.dumps({
        "operating_threshold": THR,
        "note": "test folders carry weak labels only (each file curated around "
                "one catalogued event; no arrival times shipped)",
        "stations": stations,
        "per_type_recall_s12_test": types,
    }, indent=2))
    print("saved -> results/station_transfer.json")


if __name__ == "__main__":
    main()

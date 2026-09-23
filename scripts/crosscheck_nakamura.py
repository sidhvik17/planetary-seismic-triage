"""Validate benchmark 'false positives' against the full Nakamura catalog.

The frozen lunar benchmark scores against the Space Apps packet's Grade-A
labels — 76 events. The complete Apollo PSE long-period catalog (Nakamura et
al., 2008 revision; UTIG Technical Report 18) holds 13,058, most of which
the packet simply does not label. MarsQuakeNet's headline was exactly such
detections beyond the working catalog, verified manually; here the
verification is automatic: every U-Net detection on the lunar test split
that the benchmark counts as a false positive is cross-referenced against
Nakamura events flagged as detected at station 12. A match means the model
found a real, catalogued moonquake the benchmark cannot credit.

Reports benchmark precision alongside 'survey precision' (Nakamura-matched
detections counted as true). Match tolerance is generous-but-bounded: the
catalog's signal-start times are minute-quantized, so ±5 min.

Usage: python scripts/crosscheck_nakamura.py --model runs/unet_lunar/best.pt
           --threshold 0.3 --min-dur 600
Corrected split (never overwrites an existing result):
       python scripts/crosscheck_nakamura.py --data-dir data/cache/lunar_grouped_v1
           --model models/unet_lunar_grouped_v1_seed42.pt --threshold 0.25
           --min-dur 430 --tag lunar_grouped_v1_seed42
Requires data/raw/levent.1008.dat. Writes results/nakamura_crosscheck.json
plus a per-detection CSV.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import obspy
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import CODA_SEC, DEFAULT as CFG, DATA_CACHE, PACKET_ROOT, PROJECT_ROOT
from planetseis.detect_spec import SEC_PER_BIN, compute_curve, curve_to_detections
from planetseis.evaluate import score_trace
from planetseis.unet import UNET_ARCHS, SpecUNet

MATCH_TOL_SEC = 300.0
COLSPECS = [(2, 4), (5, 8), (9, 13), (14, 18), (19, 23), (23, 27), (27, 31),
            (31, 35), (36, 40), (41, 45), (46, 76), (76, 77), (77, 80)]
NAMES = ["Year", "DOY", "StartTime", "StopTime", "A1112Amp", "A14Amp",
         "A15Amp", "A16Amp", "Availability", "Quality", "Comments",
         "EventType", "DeepClass"]


def load_nakamura() -> pd.DataFrame:
    df = pd.read_fwf(PROJECT_ROOT / "data" / "raw" / "levent.1008.dat",
                     colspecs=COLSPECS, header=None, names=NAMES)
    # station-12 column also covers station 11 (1969 only); benchmark files
    # are all 1970+ S12, so a nonzero amplitude there = detected at S12
    df = df[pd.to_numeric(df["A1112Amp"], errors="coerce").notna()]
    hhmm = df["StartTime"].astype(int)
    df = df.assign(
        utc=[obspy.UTCDateTime(year=1900 + int(y), julday=int(d))
             + (t // 100) * 3600 + (t % 100) * 60
             for y, d, t in zip(df["Year"], df["DOY"], hhmm)])
    return df


def trace_start_utc(stem: str) -> obspy.UTCDateTime | None:
    data_dir = PACKET_ROOT / "data" / "lunar" / "training" / "data"
    hits = list(data_dir.rglob(stem + ".mseed"))
    if not hits:
        return None
    return obspy.read(str(hits[0]), headonly=True)[0].stats.starttime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--min-dur", type=float, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default="",
                    help="output suffix. Without it this overwrites the "
                         "frozen v1.0 crosscheck the README cites — pass a "
                         "tag whenever evaluating anything but unet_lunar.")
    ap.add_argument("--data-dir", type=Path,
                    help="versioned dataset root (e.g. data/cache/lunar_grouped_v1); "
                         "requires --tag and never overwrites a result")
    args = ap.parse_args()
    root = args.data_dir or DATA_CACHE / "lunar"
    out_dir = PROJECT_ROOT / "results"
    sfx = f"_{args.tag}" if args.tag else ""
    out_json = out_dir / f"nakamura_crosscheck{sfx}.json"
    manifest_sha, benchmark_id = None, "historical filename split"
    if args.data_dir is not None:
        if not args.tag:
            ap.error("--data-dir requires --tag")
        if out_json.exists():
            sys.exit(f"refusing to overwrite {out_json}")
        raw_manifest = (root / "manifest.json").read_bytes()
        manifest_sha = hashlib.sha256(raw_manifest).hexdigest()
        benchmark_id = json.loads(raw_manifest)["benchmark_id"]

    # Parse the catalog before any CUDA context exists: mass UTCDateTime
    # construction next to a live context has segfaulted on this machine.
    nak = load_nakamura()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.model, map_location=device, weights_only=True)
    model = SpecUNet(base=UNET_ARCHS[ckpt.get("arch", "base")])
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    nak_times = np.array([float(t) for t in nak["utc"]])
    print(f"Nakamura S12-detected events: {len(nak)}")
    rng = np.random.default_rng(42)
    null_hits, null_draws = 0, 0
    # per-file FP absolute times + span, for the permutation test and the
    # tolerance-sensitivity sweep
    fp_abs_times: list[float] = []
    file_spans: list[tuple[float, float, int]] = []   # (t0, span_sec, n_fp)

    tol = CFG.window.match_tolerance_sec
    min_bins = max(1, int(round(args.min_dur / SEC_PER_BIN)))
    rows = []
    n_tp = n_fp = n_fn = 0
    n_fp_nak = 0
    for p in sorted((root / "continuous" / args.split).glob("*.npz")):
        z = np.load(p)
        trace, picks = z["trace"], list(z["picks"])
        # grouped caches store the actual acquisition start; older caches
        # look it up from the packet file of the same stem
        t0 = (obspy.UTCDateTime(str(z["start_time"])) if "start_time" in z.files
              else trace_start_utc(p.stem))
        curve = compute_curve(model, trace, device)
        dets = curve_to_detections(curve, args.threshold,
                                   suppress_sec=CODA_SEC["lunar"],
                                   min_bins=min_bins)
        s = score_trace([d.time_sec for d in dets], picks, tol)
        n_tp += s.tp
        n_fp += s.fp
        n_fn += s.fn
        unmatched = list(picks)
        for d in dets:
            errs = [abs(d.time_sec - pk) for pk in unmatched]
            if errs and min(errs) <= tol:
                unmatched.pop(int(np.argmin(errs)))
                status = "benchmark_tp"
                nak_dt = None
            else:
                status = "benchmark_fp"
                nak_dt = None
                if t0 is not None:
                    abs_t = float(t0) + d.time_sec
                    fp_abs_times.append(abs_t)
                    k = int(np.argmin(np.abs(nak_times - abs_t)))
                    dt = abs(nak_times[k] - abs_t)
                    if dt <= MATCH_TOL_SEC:
                        status = "fp_matches_nakamura"
                        nak_dt = round(float(dt), 0)
                        n_fp_nak += 1
            rows.append(dict(file=p.stem, time_sec=round(d.time_sec, 1),
                             confidence=round(d.confidence, 3),
                             status=status, nakamura_dt_sec=nak_dt))
        print(f"{p.stem}: {len(dets)} dets, {s.tp} tp, {s.fp} fp "
              f"({sum(1 for r in rows if r['file'] == p.stem and r['status'] == 'fp_matches_nakamura')} nak-matched)")
        # empirical null: how often do random times in this file match?
        if t0 is not None and dets:
            span = len(trace) / 6.625
            n_fp_here = sum(1 for r in rows if r["file"] == p.stem
                            and r["status"] != "benchmark_tp")
            file_spans.append((float(t0), span, n_fp_here))
            rand_abs = float(t0) + rng.uniform(0, span, size=1000 * len(dets))
            d_near = np.abs(nak_times[None, :] - rand_abs[:, None]).min(axis=1)
            null_hits += int((d_near <= MATCH_TOL_SEC).sum())
            null_draws += len(rand_abs)

    # permutation test: re-place each file's unmatched detections uniformly
    # at random inside that file's span, 10k times; p = P(matches >= observed)
    n_perm = 10_000
    fp_arr = np.array(fp_abs_times)
    perm_counts = np.zeros(n_perm, dtype=int)
    for t0f, span, n_fp_here in file_spans:
        if n_fp_here == 0:
            continue
        rand = t0f + rng.uniform(0, span, size=(n_perm, n_fp_here))
        d_near = np.abs(nak_times[None, None, :]
                        - rand[:, :, None]).min(axis=2)
        perm_counts += (d_near <= MATCH_TOL_SEC).sum(axis=1)
    p_value = float((perm_counts >= n_fp_nak).mean())

    # sensitivity of the match count to the tolerance choice
    tol_sensitivity = {}
    for tol_s in (60.0, 120.0, 180.0, 300.0):
        m = int((np.abs(nak_times[None, :] - fp_arr[:, None]).min(axis=1)
                 <= tol_s).sum()) if len(fp_arr) else 0
        tol_sensitivity[f"±{int(tol_s)}s"] = m

    survey_tp = n_tp + n_fp_nak
    result = {
        "model": Path(args.model).parent.name if args.data_dir is None else Path(args.model).name,
        "benchmark_id": benchmark_id,
        "data_manifest_sha256": manifest_sha,
        "model_sha256": hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
        "split": args.split,
        "threshold": args.threshold,
        "min_dur_sec": args.min_dur,
        "nakamura_match_tol_sec": MATCH_TOL_SEC,
        "benchmark": {"tp": n_tp, "fp": n_fp, "fn": n_fn,
                      "precision": round(n_tp / max(n_tp + n_fp, 1), 4)},
        "fp_matching_nakamura": n_fp_nak,
        "fp_nakamura_match_rate": round(n_fp_nak / max(n_fp, 1), 4),
        "chance_match_rate": round(null_hits / max(null_draws, 1), 4),
        "permutation_p_value": p_value,
        "permutation_n": n_perm,
        "match_count_by_tolerance": tol_sensitivity,
        "survey_precision": round(survey_tp / max(n_tp + n_fp, 1), 4),
        "note": "survey_precision counts detections matching ANY Nakamura "
                "S12 event as true; recall is not restated because the "
                "benchmark's pick list stays the recall denominator",
    }
    pd.DataFrame(rows).to_csv(
        out_dir / f"nakamura_crosscheck_detections{sfx}.csv", index=False)
    out_json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

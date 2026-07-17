"""Label the packet's uncatalogued InSight files from the official MQS catalog.

The Space Apps packet ships 9 Martian test files with event ids in their
names but no picks. The Marsquake Service catalog (v14, served by the IRIS
mars-event web service) contains P/start picks for every catalogued event.
This script cross-references each file's UTC span against that catalog and
writes arrival labels, growing the labeled Martian set from 2 files/2 events
to 11 files — enough for a real (if still small) Mars evaluation.

Usage: python scripts/fetch_mqs_labels.py
Writes: data/raw/mqs_arrivals.csv
        data/cache/mars_ext/continuous/{train,test}/*.npz
        benchmark/mars_ext_splits.json
Original frozen mars splits are NOT touched; this is an additive benchmark
track ("mars_ext") with documented provenance.
"""
from __future__ import annotations

import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import obspy
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import DATA_CACHE, DEFAULT as CFG, PACKET_ROOT, PROJECT_ROOT
from planetseis.preprocessing import preprocess

SERVICE = "https://service.iris.edu/irisws/mars-event/1/query"
# MQS pick preference: P is the located phase; 'start' is the signal onset
# used for unlocated events; PP appears on some deep events.
PHASE_PREF = ["P", "Pg", "PP", "start", "P_spectral_start"]

# Frozen extended-track split: files sorted by date, alternating test/train,
# with the two original packet training files pinned to train (their events
# already influenced model development).
ORIG_TRAIN = {"XB.ELYSE.02.BHV.2022-02-03HR08_evid0005"}
ORIG_TEST = {"XB.ELYSE.02.BHV.2022-01-02HR04_evid0006"}


def fetch_events(t0: obspy.UTCDateTime, t1: obspy.UTCDateTime):
    url = (f"{SERVICE}?starttime={(t0 - 3600).isoformat()}"
           f"&endtime={t1.isoformat()}&includearrivals=true")
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
    except Exception as e:
        print(f"  fetch failed: {e}")
        return []
    if not data.strip():
        return []
    try:
        return obspy.read_events(io.BytesIO(data))
    except Exception as e:
        print(f"  quakeml parse failed: {e}")
        return []


def best_pick(event, t0, t1):
    """First pick matching PHASE_PREF that falls inside the file span."""
    picks = {}
    phase_of = {}
    for origin in event.origins:
        for arr in origin.arrivals:
            phase_of[arr.pick_id] = arr.phase
    for p in event.picks:
        ph = phase_of.get(p.resource_id)
        if ph and p.time and t0 <= p.time <= t1:
            picks.setdefault(str(ph), p.time)
    for ph in PHASE_PREF:
        if ph in picks:
            return ph, picks[ph]
    return None, None


def mqs_name(event):
    for d in event.event_descriptions:
        if d.type == "earthquake name":
            return d.text
    return str(event.resource_id).split("/")[-1]


def main():
    mars_dirs = [PACKET_ROOT / "data" / "mars" / "training" / "data",
                 PACKET_ROOT / "data" / "mars" / "test" / "data"]
    files = sorted({p for d in mars_dirs for p in d.glob("*.mseed")})
    print(f"{len(files)} martian mseed files")

    rows = []
    for f in files:
        st = obspy.read(str(f))
        t0, t1 = st[0].stats.starttime, st[-1].stats.endtime
        print(f"{f.stem}: {t0} .. {t1}")
        for ev in fetch_events(t0, t1):
            ph, pt = best_pick(ev, t0, t1)
            if ph is None:
                continue
            rows.append({
                "filename": f.stem,
                "path": str(f),
                "mqs_name": mqs_name(ev),
                "event_type": str(ev.event_type or ""),
                "phase": ph,
                "arrival_utc": str(pt),
                "rel_sec": float(pt - t0),
                "file_start_utc": str(t0),
            })
            print(f"  {mqs_name(ev)} {ph} @ {pt} (rel {pt - t0:.0f}s)")

    df = pd.DataFrame(rows)
    out_csv = PROJECT_ROOT / "data" / "raw" / "mqs_arrivals.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n{len(df)} labeled arrivals -> {out_csv}")

    # build mars_ext continuous cache — split by UTC SPAN, not filename: the
    # packet ships two files covering the identical hour (evid0033/0034);
    # putting them in different splits would leak test data into train.
    by_file = df.groupby("filename")
    stems = sorted(by_file.groups)
    span_of = {s: by_file.get_group(s).iloc[0]["file_start_utc"] for s in stems}
    split_of_span: dict[str, str] = {}
    for stem in stems:
        if stem in ORIG_TRAIN:
            split_of_span[span_of[stem]] = "train"
        elif stem in ORIG_TEST:
            split_of_span[span_of[stem]] = "test"
    k = 0
    for stem in stems:
        if span_of[stem] not in split_of_span:
            split_of_span[span_of[stem]] = "test" if k % 2 == 0 else "train"
            k += 1
    split_of = {s: split_of_span[span_of[s]] for s in stems}
    # identical spans collapse to one cache entry (first stem wins)
    seen_spans: set[str] = set()
    stems = [s for s in stems
             if span_of[s] not in seen_spans and not seen_spans.add(span_of[s])]
    for stem in stems:
        sub = by_file.get_group(stem)
        path = Path(sub.iloc[0]["path"])
        st = obspy.read(str(path))
        st.merge(method=1, fill_value=0)
        tr = st[0]
        data = np.nan_to_num(np.asarray(tr.data, dtype=np.float64))
        proc, rate = preprocess(data, float(tr.stats.sampling_rate), CFG.preproc)
        out_dir = DATA_CACHE / "mars_ext" / "continuous" / split_of[stem]
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / f"{stem}.npz", trace=proc.astype(np.float32),
                            rate=rate, picks=sub["rel_sec"].to_numpy())
        print(f"cached {stem} -> {split_of[stem]} ({len(sub)} picks)")

    splits = {"train": [s for s in stems if split_of[s] == "train"],
              "test": [s for s in stems if split_of[s] == "test"],
              "provenance": "picks from MQS catalog v14 via IRIS mars-event "
                            "web service (P > start preference); original "
                            "packet train/test membership preserved"}
    (PROJECT_ROOT / "benchmark" / "mars_ext_splits.json").write_text(
        json.dumps(splits, indent=2))
    print(f"splits: {len(splits['train'])} train / {len(splits['test'])} test")


if __name__ == "__main__":
    main()

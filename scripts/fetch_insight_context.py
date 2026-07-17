"""Extend each mars_ext file with real InSight context from IRIS/EarthScope.

The packet's Martian files are single hours. After the injection engine's
guard zones around each pick, an hour leaves almost no event-free noise and
several events sit too close to a file edge to cut a full template. This
script re-fetches each labeled span from the open XB.ELYSE archive with
2.5 h of pre-event and 1.5 h of post-event context (~5 h per file), rebuilds
the mars_ext continuous cache in place, and keeps the frozen split
assignment from benchmark/mars_ext_splits.json.

Usage: python scripts/fetch_insight_context.py
Requires: data/raw/mqs_arrivals.csv (from fetch_mqs_labels.py), network.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import obspy
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import DATA_CACHE, DEFAULT as CFG, PROJECT_ROOT
from planetseis.preprocessing import preprocess

PRE_H, POST_H = 2.5, 1.5


def main():
    df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "mqs_arrivals.csv")
    splits = json.loads(
        (PROJECT_ROOT / "benchmark" / "mars_ext_splits.json").read_text())
    split_of = {s: "train" for s in splits["train"]}
    split_of.update({s: "test" for s in splits["test"]})

    client = Client("EARTHSCOPE", timeout=300)
    for stem, sub in df.groupby("filename"):
        if stem not in split_of:
            continue  # duplicate-span file collapsed out of the benchmark
        t0 = UTCDateTime(str(sub.iloc[0]["file_start_utc"]))
        t_lo, t_hi = t0 - PRE_H * 3600, t0 + 3600 + POST_H * 3600
        print(f"{stem}: fetching {t_lo} .. {t_hi}")
        try:
            st = client.get_waveforms("XB", "ELYSE", "02", "BHV", t_lo, t_hi)
        except Exception as e:
            print(f"  FAILED ({e}) — keeping packet-hour cache entry")
            continue
        st.merge(method=1, fill_value=0)
        tr = st[0]
        data = np.nan_to_num(np.asarray(tr.data, dtype=np.float64))
        proc, rate = preprocess(data, float(tr.stats.sampling_rate), CFG.preproc)
        # picks relative to the ACTUAL fetched start (may differ from t_lo)
        picks = [float(UTCDateTime(u) - tr.stats.starttime)
                 for u in sub["arrival_utc"]]
        out = (DATA_CACHE / "mars_ext" / "continuous" / split_of[stem]
               / f"{stem}.npz")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, trace=proc.astype(np.float32), rate=rate,
                            picks=np.array(picks))
        hrs = len(proc) / rate / 3600
        print(f"  cached {hrs:.1f} h, picks at "
              f"{[round(p) for p in picks]} s -> {split_of[stem]}")


if __name__ == "__main__":
    main()

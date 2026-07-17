"""Expand the mars_ext TEST split with additional MQS v14 event spans.

n=5 test events makes every Mars number anecdotal. This fetches the next
strongest MQS catalog events that are (a) not already in any mars_ext
split, (b) not inside any existing span, caches ~4-h spans around their P
picks as ADDITIONAL TEST files, and appends them to the frozen splits with
provenance. The operating point stays whatever the val split chose — the
new spans are never used for tuning.

Usage: python scripts/fetch_mqs_test_events.py [--n 12]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import DATA_CACHE, DEFAULT as CFG, PROJECT_ROOT
from planetseis.preprocessing import preprocess
from scripts.fetch_mqs_train_events import (PRE_H, POST_H, best_pick,
                                            fetch_span_events, mqs_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()

    cat = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "mqs_catalog.txt", sep="|")
    cat["Magnitude"] = pd.to_numeric(cat["Magnitude"], errors="coerce")
    cat["utc"] = cat["Time"].map(UTCDateTime)

    splits_path = PROJECT_ROOT / "benchmark" / "mars_ext_splits.json"
    splits = json.loads(splits_path.read_text())
    used_names = set()
    used_spans = []
    for split in ("train", "val", "test"):
        for stem in splits.get(split, []):
            used_names.add(stem.split(".")[-1].replace("HR", " "))
            p = DATA_CACHE / "mars_ext" / "continuous" / split / f"{stem}.npz"
            if p.exists():
                # span guard via catalog names embedded in stems is loose;
                # exact UTC guard below uses fetched events themselves
                pass
    # names already covered (S-names from fetched stems + arrivals table)
    arr = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "mqs_arrivals.csv")
    known = set(arr["mqs_name"])
    for split in ("train", "val", "test"):
        for stem in splits.get(split, []):
            tail = stem.split(".")[-1]
            if tail.startswith("S"):
                known.add(tail)

    def excluded(row):
        return row["ContributorID"] in known

    cand = cat[~cat.apply(excluded, axis=1)].sort_values(
        "Magnitude", ascending=False).head(args.n * 3)

    client = Client("EARTHSCOPE", timeout=300)
    added = []
    for _, row in cand.iterrows():
        if len(added) >= args.n:
            break
        name = row["ContributorID"]
        t_ev = row["utc"]
        t_lo, t_hi = t_ev - PRE_H * 3600, t_ev + POST_H * 3600
        events = fetch_span_events(t_lo, t_hi)
        picks_utc = [(mqs_name(ev), pt) for ev in events
                     if (pt := best_pick(ev, t_lo, t_hi)) is not None]
        if not any(n_ == name for n_, _ in picks_utc):
            print(f"{name}: no usable pick — skip")
            continue
        if any(n_ in known for n_, _ in picks_utc):
            print(f"{name}: span overlaps an existing split event — skip")
            continue
        try:
            st = client.get_waveforms("XB", "ELYSE", "02", "BHV", t_lo, t_hi)
        except Exception as e:
            print(f"{name}: waveform fetch failed ({e}) — skip")
            continue
        st.merge(method=1, fill_value=0)
        tr = st[0]
        if tr.stats.endtime - tr.stats.starttime < 2 * 3600:
            print(f"{name}: too gappy — skip")
            continue
        data = np.nan_to_num(np.asarray(tr.data, dtype=np.float64))
        proc, rate = preprocess(data, float(tr.stats.sampling_rate), CFG.preproc)
        rel = [float(pt - tr.stats.starttime) for _, pt in picks_utc]
        rel = [r for r in rel if 0 <= r <= len(proc) / rate]
        stem = f"XB.ELYSE.02.BHV.{name}"
        out = DATA_CACHE / "mars_ext" / "continuous" / "test" / f"{stem}.npz"
        np.savez_compressed(out, trace=proc.astype(np.float32), rate=rate,
                            picks=np.array(rel))
        added.append(stem)
        for n_, _ in picks_utc:
            known.add(n_)
        print(f"{name}: M{row['Magnitude']:.1f} -> test "
              f"({len(rel)} pick(s), {len(proc)/rate/3600:.1f} h)")

    splits["test"] = sorted(set(splits["test"]) | set(added))
    splits["provenance"] += (f"; test extended with {len(added)} additional "
                             "MQS v14 spans (fetch_mqs_test_events.py) — "
                             "never used for tuning")
    splits_path.write_text(json.dumps(splits, indent=2))
    print(f"\nadded {len(added)} test spans; test now {len(splits['test'])} files")


if __name__ == "__main__":
    main()

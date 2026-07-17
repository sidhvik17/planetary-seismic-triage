"""Grow the mars_ext TRAIN/VAL sets with additional MQS v14 events.

The packet-derived mars_ext training set holds only 7 events — too few
templates for injection training to generalize (train F1 1.0, test F1 0.2:
pure template overfit). The MQS catalog has hundreds more. This script picks
the strongest catalogued events that are (a) not one of the frozen test
events, (b) not inside any frozen test file's UTC span, fetches ~4 h of
XB.ELYSE.02.BHV around each P pick from EarthScope, and caches them as extra
mars_ext train/val files. The newest fetched events become the VAL split
(mars_ext previously had none), so the operating point is no longer tuned on
the model's own template sources.

Usage: python scripts/fetch_mqs_train_events.py [--n 24]
Updates benchmark/mars_ext_splits.json (train/val only — test is frozen).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import obspy
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import DATA_CACHE, DEFAULT as CFG, PROJECT_ROOT
from planetseis.preprocessing import preprocess

SERVICE = "https://service.iris.edu/irisws/mars-event/1/query"
PHASE_PREF = ["P", "Pg", "PP", "start", "P_spectral_start"]
PRE_H, POST_H = 2.5, 1.5
N_VAL = 5


def fetch_span_events(t0, t1):
    url = (f"{SERVICE}?starttime={(t0 - 3600).isoformat()}"
           f"&endtime={t1.isoformat()}&includearrivals=true")
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        return obspy.read_events(io.BytesIO(data)) if data.strip() else []
    except Exception as e:
        print(f"  arrivals fetch failed: {e}")
        return []


def best_pick(event, t0, t1):
    phase_of = {}
    for origin in event.origins:
        for arr in origin.arrivals:
            phase_of[arr.pick_id] = arr.phase
    picks = {}
    for p in event.picks:
        ph = phase_of.get(p.resource_id)
        if ph and p.time and t0 <= p.time <= t1:
            picks.setdefault(str(ph), p.time)
    for ph in PHASE_PREF:
        if ph in picks:
            return picks[ph]
    return None


def mqs_name(event):
    for d in event.event_descriptions:
        if d.type == "earthquake name":
            return d.text
    return str(event.resource_id).split("/")[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()

    cat = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "mqs_catalog.txt",
                      sep="|")
    cat["Magnitude"] = pd.to_numeric(cat["Magnitude"], errors="coerce")
    cat["utc"] = cat["Time"].map(UTCDateTime)

    # exclusion: frozen test events + test file spans
    arrivals = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "mqs_arrivals.csv")
    splits = json.loads(
        (PROJECT_ROOT / "benchmark" / "mars_ext_splits.json").read_text())
    known_names = set(arrivals["mqs_name"])
    test_spans = []
    for stem in splits["test"]:
        rows = arrivals[arrivals.filename == stem]
        t0 = UTCDateTime(str(rows.iloc[0]["file_start_utc"]))
        test_spans.append((t0 - PRE_H * 3600 - 7200,
                           t0 + 3600 + POST_H * 3600 + 7200))

    def excluded(row):
        if row["ContributorID"] in known_names:
            return True
        return any(lo <= row["utc"] <= hi for lo, hi in test_spans)

    cand = cat[~cat.apply(excluded, axis=1)].sort_values(
        "Magnitude", ascending=False).head(args.n * 2)

    client = Client("EARTHSCOPE", timeout=300)
    fetched = []
    for _, row in cand.iterrows():
        if len(fetched) >= args.n:
            break
        name = row["ContributorID"]
        t_ev = row["utc"]
        t_lo, t_hi = t_ev - PRE_H * 3600, t_ev + POST_H * 3600
        events = fetch_span_events(t_lo, t_hi)
        picks_utc = []
        for ev in events:
            pt = best_pick(ev, t_lo, t_hi)
            if pt is not None:
                picks_utc.append((mqs_name(ev), pt))
        if not any(n == name for n, _ in picks_utc):
            print(f"{name}: no usable pick — skip")
            continue
        try:
            st = client.get_waveforms("XB", "ELYSE", "02", "BHV", t_lo, t_hi)
        except Exception as e:
            print(f"{name}: waveform fetch failed ({e}) — skip")
            continue
        st.merge(method=1, fill_value=0)
        tr = st[0]
        if tr.stats.endtime - tr.stats.starttime < 2 * 3600:
            print(f"{name}: span too gappy — skip")
            continue
        data = np.nan_to_num(np.asarray(tr.data, dtype=np.float64))
        proc, rate = preprocess(data, float(tr.stats.sampling_rate), CFG.preproc)
        rel = [float(pt - tr.stats.starttime) for _, pt in picks_utc]
        rel = [r for r in rel if 0 <= r <= len(proc) / rate]
        stem = f"XB.ELYSE.02.BHV.{name}"
        fetched.append((stem, proc.astype(np.float32), rate, rel,
                        float(row["Magnitude"])))
        print(f"{name}: M{row['Magnitude']:.1f} cached "
              f"{len(proc)/rate/3600:.1f} h, {len(rel)} pick(s)")

    # newest N_VAL fetched events -> val; rest -> train
    fetched.sort(key=lambda f: f[0])
    val_set = {f[0] for f in fetched[-N_VAL:]}
    for stem, proc, rate, rel, _ in fetched:
        split = "val" if stem in val_set else "train"
        out = DATA_CACHE / "mars_ext" / "continuous" / split / f"{stem}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, trace=proc, rate=rate, picks=np.array(rel))

    splits["train"] = sorted(set(splits["train"])
                             | {f[0] for f in fetched if f[0] not in val_set})
    splits["val"] = sorted(val_set)
    splits["provenance"] += ("; train/val extended with top-magnitude MQS v14 "
                             "events fetched from EarthScope (fetch_mqs_train_"
                             "events.py), test spans excluded")
    (PROJECT_ROOT / "benchmark" / "mars_ext_splits.json").write_text(
        json.dumps(splits, indent=2))
    print(f"\n{len(fetched)} events added: "
          f"{len(fetched) - len(val_set)} train / {len(val_set)} val")


if __name__ == "__main__":
    main()

"""Fetch and cache the continuous Apollo PSE archive (FDSN XA) for B1.

    python scripts/fetch_apollo_archive.py --stations S12 S14 S15 S16
    python scripts/fetch_apollo_archive.py --stations S12 --probe   # 3 days only

Why this exists: the Space Apps packet is 183 curated single-event snippets,
so the frozen benchmark has 19 test events and every confidence interval spans
+/-0.25 F1. The archive is ~9,500 station-days of continuous recording, which
is what turns a leaderboard comparison into a catalogue result — recall
against Nakamura's 5,358 S12-detected events, stratified by event type, with
false alarms per station-day as the denominator.

Requests are month-chunked (~320 requests for all four stations rather than
one per day), cleaned once per chunk so bad runs spanning midnight are handled
coherently, bandpassed once per chunk so no per-day detrend discontinuity
enters the cache, then sliced into day files carrying PAD_SEC of context on
each side (the detector zeroes EDGE_GUARD_BINS at every trace edge, and
without the pad every event near midnight would be blanked).

Output per day: data/cache/archive/{station}/{YYYY-MM-DD}.npz with
  trace   float32, bandpassed, 6.625 Hz, day +/- PAD_SEC
  valid   uint8 bit-packed per-sample validity (see archive.clean_samples)
  pad_sec, loc, rate, starttime, plus the cleaning stats
Re-running skips days already cached, so an interrupted pull resumes.
A manifest with retrieval provenance is written per station for the data
section (SRL Data Mine and ESS both require it).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.archive import (MAX_INTERP_SEC, clean_samples, pick_location,
                                stream_to_samples)
from planetseis.config import ARCHIVE_CACHE, DEFAULT as CFG
from planetseis.preprocessing import preprocess

warnings.filterwarnings("ignore")

NET, CHAN = "XA", "MHZ"
RATE = 6.625                 # archive nominal rate == our target rate exactly
PAD_SEC = 1800.0             # context each side of a day file
ARCHIVE = ARCHIVE_CACHE      # set PLANETSEIS_ARCHIVE to keep this off OneDrive
RETRIES, BACKOFF = 4, 5.0


def month_edges(t0: UTCDateTime, t1: UTCDateTime):
    """[start, stop) month boundaries covering [t0, t1)."""
    out, cur = [], UTCDateTime(t0.year, t0.month, 1)
    while cur < t1:
        nxt = UTCDateTime(cur.year + (cur.month == 12),
                          cur.month % 12 + 1, 1)
        out.append((max(cur, t0), min(nxt, t1)))
        cur = nxt
    return out


def fetch(client, station, t0, t1):
    """One waveform request with backoff. None means 'no data', not failure."""
    for attempt in range(RETRIES):
        try:
            return client.get_waveforms(NET, station, "*", CHAN, t0, t1)
        except Exception as exc:                       # noqa: BLE001
            msg = str(exc).lower()
            if "no data" in msg or "204" in msg:
                return None
            if attempt == RETRIES - 1:
                print(f"    ! give up {t0.date} {exc}")
                return None
            time.sleep(BACKOFF * (attempt + 1))
    return None


def station_span(client, station):
    inv = client.get_stations(network=NET, station=station,
                              starttime=UTCDateTime("1969-01-01"),
                              endtime=UTCDateTime("1978-01-01"), level="station")
    s = inv[0][0]
    return UTCDateTime(s.start_date), UTCDateTime(s.end_date)


def day_path(station, day):
    return ARCHIVE / station / f"{day.strftime('%Y-%m-%d')}.npz"


def process_month(client, station, m0, m1, force=False):
    """Fetch, clean and bandpass one month; write its day files."""
    days = [m0 + i * 86400 for i in range(int((m1 - m0) // 86400) + 1)]
    days = [UTCDateTime(d.year, d.month, d.day) for d in days if d < m1]
    if not force and all(day_path(station, d).exists() for d in days):
        return 0, 0

    q0, q1 = m0 - PAD_SEC, m1 + PAD_SEC
    st = fetch(client, station, q0, q1)
    if st is None or not len(st):
        return 0, 0
    loc = pick_location(st)
    raw, gap = stream_to_samples(st, loc, q0, q1, RATE)
    clean, valid, _ = clean_samples(raw, RATE, gap_mask=gap,
                                    max_interp_sec=MAX_INTERP_SEC)
    # one detrend+bandpass for the whole chunk: per-day filtering would leave a
    # discontinuity at every midnight, which is exactly what the scan reads as
    # an onset
    proc, _ = preprocess(clean, RATE, CFG.preproc)

    written = kept = 0
    out_dir = ARCHIVE / station
    out_dir.mkdir(parents=True, exist_ok=True)
    for d in days:
        p = day_path(station, d)
        if p.exists() and not force:
            continue
        lo = int(round((d - PAD_SEC - q0) * RATE))
        hi = int(round((d + 86400 + PAD_SEC - q0) * RATE))
        lo_c, hi_c = max(lo, 0), min(hi, len(proc))
        if hi_c - lo_c < int(600 * RATE):        # <10 min of anything: skip
            continue
        seg, seg_valid = proc[lo_c:hi_c], valid[lo_c:hi_c]
        if not seg_valid.any():
            continue
        # the slice is clipped whenever the requested pad runs past the chunk
        # (a station's first day), so the stored start must be derived from the
        # ACTUAL first sample — labelling it d - PAD_SEC would offset every
        # detection in that file by up to the clip, silently poisoning the
        # Nakamura match and any multi-station moveout
        seg_start = q0 + lo_c / RATE
        # re-derive per-day stats from the sliced validity mask
        stats = dict(n_samples=int(len(seg)),
                     n_invalid=int((~seg_valid).sum()),
                     valid_frac=round(float(seg_valid.mean()), 6))
        np.savez_compressed(
            p, trace=seg.astype(np.float32),
            valid=np.packbits(seg_valid), n_valid_bits=len(seg_valid),
            rate=RATE, loc=loc, pad_sec=PAD_SEC,
            starttime=str(seg_start), day=d.strftime("%Y-%m-%d"),
            stats=json.dumps(stats))
        written += 1
        kept += int(seg_valid.sum())
    return written, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", nargs="+", default=["S12", "S14", "S15", "S16"])
    ap.add_argument("--probe", action="store_true",
                    help="first 3 days per station only — sizing run")
    ap.add_argument("--force", action="store_true", help="rewrite existing days")
    args = ap.parse_args()

    client = Client("IRIS", timeout=180)
    grand_days = grand_valid = 0
    for station in args.stations:
        t0, t1 = station_span(client, station)
        if args.probe:
            t1 = t0 + 3 * 86400
        print(f"\n=== {station}  {t0.date} -> {t1.date} "
              f"({int((t1 - t0) // 86400)} days) ===")
        n_days = n_valid = 0
        for m0, m1 in month_edges(t0, t1):
            w, k = process_month(client, station, m0, m1, force=args.force)
            n_days += w
            n_valid += k
            if w:
                print(f"  {m0.strftime('%Y-%m')}: {w:2d} days, "
                      f"{k / RATE / 86400:.2f} valid station-days")
        manifest = ARCHIVE / station / "manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({
            "network": NET, "station": station, "channel": CHAN,
            "nominal_rate_hz": RATE, "pad_sec": PAD_SEC,
            "max_interp_sec": MAX_INTERP_SEC,
            "span": [str(t0), str(t1)],
            "day_files": n_days,
            "valid_station_days": round(n_valid / RATE / 86400, 3),
            "service": "http://service.iris.edu/fdsnws/dataselect/1/",
            "retrieved_utc": str(UTCDateTime()),
            "bandpass_hz": list(CFG.preproc.band_hz),
            "note": "day files carry pad_sec of context each side; `valid` is "
                    "bit-packed, unpack with np.unpackbits(...)[:n_valid_bits]",
        }, indent=2))
        print(f"  -> {n_days} day files, "
              f"{n_valid / RATE / 86400:.1f} valid station-days")
        grand_days += n_days
        grand_valid += n_valid
    print(f"\nTOTAL {grand_days} day files, "
          f"{grand_valid / RATE / 86400:.1f} valid station-days")


if __name__ == "__main__":
    main()

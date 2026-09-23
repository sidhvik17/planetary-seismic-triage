"""Apollo PSE continuous-archive ingest (FDSN network XA).

The Space Apps packet ships 183 curated single-event snippets, already
cleaned. The full archive does not: it carries the raw instrument record with
every documented Apollo data pathology intact (Nunn et al. 2022, PSJ 3:219).
This module is the ONLY place archive bytes become model-ready traces, so the
continuous scan and the packet benchmark share one preprocessing path
(the F-16 contract that `preprocessing.py` exists to enforce).

Three pathologies matter enough to handle explicitly:

1. Missing samples are flagged with the literal value -1, not masked — the
   archive stores them inline for file-size reasons. On a trace whose valid
   centerline is ~495 counts and whose signal range spans ~15 counts, each
   flag is a 496-count negative spike: 33x the entire dynamic range. Measured
   on 1973-03-01 S12 MHZ, 464 flags in one day (0.081%) in 5 runs, longest 396
   samples (60 s). Passed through `preprocess` untouched they inflate the
   day's std 7.7x and its peak 48.6x, and a 60 s run is a rectangular pulse —
   event-shaped in a spectrogram, exactly the 'stripe = event' failure mode
   `injection.despike` was written to prevent.

2. True transmission gaps (damaged Barker codes, ~0.3% of the archive) arrive
   as stream discontinuities. Zero-filling them — `preprocessing.load_trace`'s
   behaviour, harmless on pre-cleaned snippets — manufactures a step edge at
   every dropout, which bandpasses into a broadband transient.

3. Gain state is carried in the location code: '00' is peaked mode, '01' is
   flat, and peaked-mode sensitivity is 5.6x flat. Merging across location
   codes would splice a 5.6x amplitude step into the middle of a trace. We
   never do: one location code per day file, recorded in the output.

Gap policy (chosen deliberately, see `clean_samples`): interpolate ALL bad
runs so the waveform stays continuous and no synthetic edges enter the
filter, but mark runs longer than `max_interp_sec` invalid in a per-sample
validity mask. Detections overlapping invalid samples are rejected at scan
time, and the valid-sample total is the denominator for false alarms per
station-day — the convention that makes the number comparable to prior work.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# The archive's inline missing-sample flag (Nunn et al. 2022).
MISSING_FLAG = -1

# Bad runs at or below this are interpolated and still counted as valid data;
# longer runs stay interpolated (no step edge) but are marked invalid so they
# can never source a detection or inflate the false-alarm denominator.
MAX_INTERP_SEC = 10.0

# Peaked ('00') vs flat ('01') gain state. Sensitivity differs 5.6x, so a
# single day file is only ever built from ONE of them.
LOC_PEAKED, LOC_FLAT = "00", "01"


@dataclass
class CleanStats:
    """Per-day accounting, written into the cache for the data section."""
    n_samples: int = 0
    n_flagged: int = 0          # samples equal to MISSING_FLAG
    n_gap: int = 0              # samples absent from the stream entirely
    n_interpolated: int = 0     # bad samples repaired and kept valid
    n_invalid: int = 0          # bad samples repaired but marked unusable
    n_runs: int = 0
    longest_run_sec: float = 0.0

    @property
    def valid_frac(self) -> float:
        return (self.n_samples - self.n_invalid) / self.n_samples if self.n_samples else 0.0

    def as_dict(self) -> dict:
        return {"n_samples": self.n_samples, "n_flagged": self.n_flagged,
                "n_gap": self.n_gap, "n_interpolated": self.n_interpolated,
                "n_invalid": self.n_invalid, "n_runs": self.n_runs,
                "longest_run_sec": round(self.longest_run_sec, 2),
                "valid_frac": round(self.valid_frac, 6)}


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True spans of `mask` as [start, stop) index pairs."""
    if not mask.any():
        return []
    idx = np.flatnonzero(mask)
    brk = np.flatnonzero(np.diff(idx) != 1)
    starts = np.concatenate(([idx[0]], idx[brk + 1]))
    stops = np.concatenate((idx[brk], [idx[-1]])) + 1
    return list(zip(starts.tolist(), stops.tolist()))


def clean_samples(
    data: np.ndarray,
    rate_hz: float,
    gap_mask: np.ndarray | None = None,
    max_interp_sec: float = MAX_INTERP_SEC,
) -> tuple[np.ndarray, np.ndarray, CleanStats]:
    """Repair archive pathologies and report what was repaired.

    `data` is the raw counts trace; `gap_mask` marks samples the stream did
    not supply at all (from an unfilled ObsPy merge). Returns
    (repaired float64 samples, per-sample validity mask, stats).

    Every bad sample is linearly interpolated from the nearest valid
    neighbours — a smooth bridge rather than a step, so nothing broadband
    enters the bandpass. Validity is tracked separately: bad runs longer than
    `max_interp_sec` are interpolated for continuity but flagged invalid,
    because a minute of invented signal must not be allowed to source a
    detection or count toward observed station-time.
    """
    x = np.asarray(data, dtype=np.float64).copy()
    n = len(x)
    stats = CleanStats(n_samples=n)
    if n == 0:
        return x, np.zeros(0, dtype=bool), stats

    flagged = x == MISSING_FLAG
    gaps = (np.zeros(n, dtype=bool) if gap_mask is None
            else np.asarray(gap_mask, dtype=bool))
    bad = flagged | gaps | ~np.isfinite(x)
    stats.n_flagged = int(flagged.sum())
    stats.n_gap = int(gaps.sum())

    valid = ~bad
    if not valid.any():
        # nothing recoverable — hand back zeros, everything invalid
        stats.n_invalid = n
        stats.n_runs = 1
        stats.longest_run_sec = n / rate_hz
        return np.zeros(n), np.zeros(n, dtype=bool), stats
    if not bad.any():
        return x, np.ones(n, dtype=bool), stats

    # bridge every bad sample; np.interp clamps at the ends, so leading and
    # trailing bad runs become a flat hold of the nearest valid value
    good_idx = np.flatnonzero(valid)
    x[bad] = np.interp(np.flatnonzero(bad), good_idx, x[good_idx])

    validity = np.ones(n, dtype=bool)
    max_run = int(round(max_interp_sec * rate_hz))
    runs = _runs(bad)
    stats.n_runs = len(runs)
    stats.longest_run_sec = max(stop - start for start, stop in runs) / rate_hz
    for start, stop in runs:
        if stop - start > max_run:
            validity[start:stop] = False
    stats.n_invalid = int((~validity).sum())
    stats.n_interpolated = int(bad.sum()) - stats.n_invalid
    return x, validity, stats


def pick_location(stream) -> str | None:
    """Choose one gain state for a day: whichever location code supplies more
    samples. Never merge across them — peaked is 5.6x flat, and splicing the
    two puts an amplitude step mid-trace that looks like an onset."""
    totals: dict[str, int] = {}
    for tr in stream:
        totals[tr.stats.location] = totals.get(tr.stats.location, 0) + tr.stats.npts
    return max(totals, key=totals.get) if totals else None


def stream_to_samples(
    stream, location: str, starttime, endtime, rate_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize one location code's traces onto a fixed grid for [start, end).

    Samples the stream never supplied come back marked in the gap mask, so
    `clean_samples` can treat them exactly like inline -1 flags instead of
    silently becoming zeros.
    """
    n = int(round((endtime - starttime) * rate_hz))
    out = np.zeros(n, dtype=np.float64)
    have = np.zeros(n, dtype=bool)
    for tr in stream:
        if tr.stats.location != location or tr.stats.npts == 0:
            continue
        off = int(round((tr.stats.starttime - starttime) * rate_hz))
        d = np.asarray(tr.data, dtype=np.float64)
        lo, hi = max(off, 0), min(off + len(d), n)
        if hi <= lo:
            continue
        out[lo:hi] = d[lo - off : hi - off]
        have[lo:hi] = True
    return out, ~have

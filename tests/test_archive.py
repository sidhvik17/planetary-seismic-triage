"""Contracts for Apollo archive ingest.

These guard the pathologies measured on real XA data (see archive.py): the
-1 missing-sample flag must never reach the filter, long dropouts must never
count as observed station-time, and gain states must never be spliced.
"""
import numpy as np
import pytest

from planetseis.archive import (MISSING_FLAG, CleanStats, clean_samples,
                                pick_location, stream_to_samples)

RATE = 6.625


def test_clean_trace_untouched():
    x = np.sin(np.arange(1000) / 10.0) + 495.0
    out, valid, st = clean_samples(x, RATE)
    assert np.allclose(out, x)
    assert valid.all()
    assert st.n_flagged == 0 and st.n_invalid == 0
    assert st.valid_frac == 1.0


def test_short_flag_run_interpolated_and_stays_valid():
    x = np.full(1000, 495.0)
    x[500:504] = MISSING_FLAG           # 4 samples = 0.6 s, well under 10 s
    out, valid, st = clean_samples(x, RATE)
    assert not np.any(out == MISSING_FLAG)
    assert np.allclose(out, 495.0)      # bridged from equal neighbours
    assert valid.all()                  # short repair still counts as observed
    assert st.n_flagged == 4 and st.n_interpolated == 4 and st.n_invalid == 0


def test_long_flag_run_interpolated_but_marked_invalid():
    x = np.full(2000, 495.0)
    bad = slice(500, 500 + int(60 * RATE))    # 60 s, the real S12 worst case
    x[bad] = MISSING_FLAG
    out, valid, st = clean_samples(x, RATE)
    assert not np.any(out == MISSING_FLAG)
    assert not valid[bad].any()               # cannot source a detection
    assert valid[:500].all() and valid[500 + int(60 * RATE):].all()
    assert st.n_invalid == int(60 * RATE)
    assert st.n_interpolated == 0
    assert st.longest_run_sec == pytest.approx(60.0, abs=0.2)


def test_flag_spike_never_reaches_the_filter():
    """The whole point: a -1 on a ~495-count centreline is a 33x-range spike."""
    rng = np.random.default_rng(0)
    x = 495.0 + rng.normal(0, 3.0, 5000)
    clean_peak = np.abs(x - x.mean()).max()
    x[[100, 2500, 4000]] = MISSING_FLAG
    out, valid, st = clean_samples(x, RATE)
    assert np.abs(out - out.mean()).max() <= clean_peak * 1.1
    assert st.n_flagged == 3
    assert valid.all()


def test_stream_gaps_treated_like_flags():
    x = np.full(1000, 495.0)
    gap = np.zeros(1000, dtype=bool)
    gap[200:900] = True                       # 105 s absent from the stream
    out, valid, st = clean_samples(x, RATE, gap_mask=gap)
    assert st.n_gap == 700
    assert not valid[200:900].any()
    assert np.isfinite(out).all()             # bridged, not zero-filled


def test_all_bad_day_is_fully_invalid():
    x = np.full(500, MISSING_FLAG, dtype=float)
    out, valid, st = clean_samples(x, RATE)
    assert not valid.any()
    assert st.valid_frac == 0.0
    assert np.isfinite(out).all()


def test_leading_and_trailing_runs_do_not_crash():
    x = np.full(1000, 495.0)
    x[:5] = MISSING_FLAG
    x[-5:] = MISSING_FLAG
    out, valid, st = clean_samples(x, RATE)
    assert np.isfinite(out).all()
    assert st.n_runs == 2


class _Tr:
    def __init__(self, loc, npts, start=0.0, data=None):
        self.stats = type("S", (), {})()
        self.stats.location = loc
        self.stats.npts = npts
        self.stats.starttime = start
        self.data = np.arange(npts, dtype=float) if data is None else data


def test_pick_location_prefers_the_fuller_gain_state():
    assert pick_location([_Tr("00", 900), _Tr("01", 100)]) == "00"
    assert pick_location([_Tr("00", 10), _Tr("01", 999)]) == "01"
    assert pick_location([]) is None


def test_stream_to_samples_marks_unsupplied_span():
    # one trace covering the middle third of a 300-sample grid
    tr = _Tr("00", 100, start=100 / RATE, data=np.full(100, 7.0))
    out, gap = stream_to_samples([tr], "00", 0.0, 300 / RATE, RATE)
    assert len(out) == 300
    assert not gap[100:200].any()
    assert gap[:100].all() and gap[200:].all()
    assert np.allclose(out[100:200], 7.0)


def test_stream_to_samples_ignores_other_location():
    tr = _Tr("01", 100, start=0.0, data=np.full(100, 7.0))
    out, gap = stream_to_samples([tr], "00", 0.0, 100 / RATE, RATE)
    assert gap.all()          # nothing from the requested gain state

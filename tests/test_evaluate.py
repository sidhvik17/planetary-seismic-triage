"""Contracts for the detection scorer.

`evaluate.score_trace` produces every number in the headline table, the
paired bootstrap and the Nakamura crosscheck, so its behaviour is pinned
here — including one documented suboptimality that is currently harmless and
must not become harmful silently.
"""
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from planetseis.evaluate import Scores, score_trace

TOL = 120.0


def optimal_match(dets, truth, tol):
    """Reference scorer: globally optimal one-to-one assignment."""
    s = Scores()
    if not dets or not truth:
        s.fp, s.fn = len(dets), len(truth)
        return s
    C = np.abs(np.asarray(dets, float)[:, None] - np.asarray(truth, float)[None, :])
    BIG = 1e6
    Cm = np.where(C <= tol, C, BIG)
    for i, j in zip(*linear_sum_assignment(Cm)):
        if Cm[i, j] < BIG:
            s.tp += 1
            s.abs_errors_sec.append(float(C[i, j]))
    s.fp, s.fn = len(dets) - s.tp, len(truth) - s.tp
    return s


def test_exact_hit():
    s = score_trace([100.0], [100.0], TOL)
    assert (s.tp, s.fp, s.fn) == (1, 0, 0)
    assert s.mae_sec == 0.0
    assert s.precision == s.recall == s.f1 == 1.0


def test_miss_beyond_tolerance_is_fp_plus_fn():
    s = score_trace([500.0], [100.0], TOL)
    assert (s.tp, s.fp, s.fn) == (0, 1, 1)
    assert np.isnan(s.mae_sec)


def test_empty_inputs():
    assert (score_trace([], [], TOL).tp, score_trace([], [], TOL).fn) == (0, 0)
    assert score_trace([], [100.0], TOL).fn == 1
    assert score_trace([100.0], [], TOL).fp == 1


def test_one_to_one_no_double_counting():
    """Two detections on one pick: one TP, one FP — never two TPs."""
    s = score_trace([100.0, 110.0], [100.0], TOL)
    assert (s.tp, s.fp, s.fn) == (1, 1, 0)


def test_tolerance_boundary_is_inclusive():
    assert score_trace([100.0 + TOL], [100.0], TOL).tp == 1
    assert score_trace([100.0 + TOL + 1e-6], [100.0], TOL).tp == 0


def test_scores_merge_accumulates():
    a, b = score_trace([100.0], [100.0], TOL), score_trace([500.0], [100.0], TOL)
    a.merge(b)
    assert (a.tp, a.fp, a.fn) == (1, 1, 1)


def test_as_dict_reports_none_mae_when_no_true_positives():
    d = score_trace([500.0], [100.0], TOL).as_dict()
    assert d["mae_sec"] is None and d["median_ae_sec"] is None


# --- the documented suboptimality -------------------------------------------
# Matching walks detections in TIME order and takes the nearest unmatched
# pick, rather than solving the assignment globally. An early sloppy
# detection can therefore consume a pick and leave the accurate one counted
# as a false positive. Verified 2026-07-25 against Hungarian assignment: this
# changes NO published number (lunar and mars_ext test sets score
# identically) because dead-time suppression keeps detections sparse. Pinned
# so that if suppression is ever relaxed, the divergence surfaces in CI
# rather than in a headline table.

def test_greedy_inflates_mae_when_detections_collide():
    dets, truth = [0.0, 105.0], [100.0]
    greedy, best = score_trace(dets, truth, TOL), optimal_match(dets, truth, TOL)
    assert (greedy.tp, greedy.fp) == (best.tp, best.fp) == (1, 1)
    assert greedy.mae_sec == pytest.approx(100.0)   # early detection wins
    assert best.mae_sec == pytest.approx(5.0)       # the accurate one should


def test_greedy_is_order_invariant():
    a = score_trace([0.0, 105.0], [100.0], TOL).as_dict()
    b = score_trace([105.0, 0.0], [100.0], TOL).as_dict()
    assert a == b


@pytest.mark.parametrize("dets,truth", [
    ([100.0, 300.0], [110.0, 290.0]),
    ([100.0], [90.0, 300.0]),
    ([50.0, 200.0, 400.0], [60.0, 210.0, 390.0]),
    ([1000.0], [100.0]),
    ([100.0, 160.0, 220.0], [150.0]),
])
def test_greedy_matches_optimal_counts_on_sparse_detections(dets, truth):
    """Sparse detections — the regime dead-time suppression enforces."""
    g, o = score_trace(dets, truth, TOL), optimal_match(dets, truth, TOL)
    assert (g.tp, g.fp, g.fn) == (o.tp, o.fp, o.fn)


def test_greedy_matches_optimal_on_random_sparse_traces():
    """Randomised: picks separated by more than the tolerance, one detection
    near each. Counts must agree with the optimal assignment every time."""
    rng = np.random.default_rng(0)
    for _ in range(300):
        n = int(rng.integers(1, 6))
        truth = np.cumsum(rng.uniform(4 * TOL, 8 * TOL, n)).tolist()
        dets = [t + rng.uniform(-TOL * 0.9, TOL * 0.9) for t in truth
                if rng.random() < 0.8]
        dets += [t + rng.uniform(3 * TOL, 4 * TOL) for t in truth
                 if rng.random() < 0.3]
        g, o = score_trace(dets, truth, TOL), optimal_match(dets, truth, TOL)
        assert (g.tp, g.fp, g.fn) == (o.tp, o.fp, o.fn)

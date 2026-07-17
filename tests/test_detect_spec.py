"""End-to-end detection contract: a perfect mask must yield the right arrival."""
import numpy as np
import torch

from planetseis.config import DEFAULT as CFG
from planetseis.detect_spec import (
    _smooth,
    _window_starts,
    curve_to_detections,
    detect_events_spec,
)
from planetseis.spectral import N_SAMPLES, SEC_PER_BIN
from planetseis.unet import SpecUNet


def test_window_starts_cover_tail():
    starts = _window_starts(N_SAMPLES * 3 + 100)
    assert starts[0] == 0
    assert starts[-1] == N_SAMPLES * 3 + 100 - N_SAMPLES
    starts_short = _window_starts(100)
    assert starts_short == [0]


def test_curve_to_detections_onset_and_suppression():
    curve = np.zeros(1000)
    curve[100:140] = 0.8          # event A
    curve[150:170] = 0.5          # aftershock inside dead time, weaker
    curve[600:640] = 0.9          # event B
    curve[700] = 0.99             # 1-bin speckle: must be discarded
    dead = 200 * SEC_PER_BIN
    dets = curve_to_detections(curve, threshold=0.3, suppress_sec=dead)
    assert len(dets) == 2
    assert dets[0].time_sec == 100 * SEC_PER_BIN
    assert dets[0].confidence == 0.8
    assert dets[1].time_sec == 600 * SEC_PER_BIN


def test_curve_to_detections_keeps_stronger_in_dead_time():
    curve = np.zeros(1000)
    curve[100:120] = 0.4
    curve[150:180] = 0.9          # stronger event inside the dead-time slot
    dets = curve_to_detections(curve, 0.3, suppress_sec=100 * SEC_PER_BIN)
    assert len(dets) == 1
    assert dets[0].confidence == 0.9


def test_smooth_preserves_length():
    c = np.random.rand(500)
    assert _smooth(c).shape == c.shape


def test_detect_events_spec_runs_on_untrained_model():
    """Pipeline plumbing: untrained model on a short trace, no crash, sane types."""
    model = SpecUNet(base=8)
    trace = np.random.randn(N_SAMPLES * 2).astype(np.float32)
    dets, curve = detect_events_spec(model, trace, 6.625, CFG,
                                     threshold=0.99, device="cpu")
    assert isinstance(dets, list)
    assert curve.ndim == 1 and np.isfinite(curve).all()

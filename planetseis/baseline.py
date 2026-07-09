"""Classical STA/LTA baseline (milestone 3) — the number the CNN must beat."""
from __future__ import annotations

import numpy as np
from obspy.signal.trigger import classic_sta_lta, trigger_onset


def sta_lta_detect(
    trace: np.ndarray,
    rate_hz: float,
    sta_sec: float = 60.0,
    lta_sec: float = 600.0,
    thr_on: float = 3.0,
    thr_off: float = 1.5,
) -> list[float]:
    """Returns trigger-on times in seconds from trace start."""
    nsta = max(1, int(sta_sec * rate_hz))
    nlta = max(nsta + 1, int(lta_sec * rate_hz))
    if len(trace) <= nlta:
        return []
    cft = classic_sta_lta(trace.astype(np.float64), nsta, nlta)
    onsets = trigger_onset(cft, thr_on, thr_off)
    return [float(on / rate_hz) for on, _ in onsets]

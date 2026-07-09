"""Planetary seismic event detector — web demo (FR-12..FR-16).

Run locally:  streamlit run app/streamlit_app.py
Deploy: Hugging Face Spaces (streamlit SDK) or Streamlit Community Cloud.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import CODA_SEC, DEFAULT as CFG, PROJECT_ROOT, RUNS_DIR
from planetseis.detect import detect_events
from planetseis.model import SeisCNN, count_params
from planetseis.preprocessing import load_trace, preprocess

MAX_UPLOAD_MB = 200
MAX_SAMPLES = 12_000_000  # cap trace length (F-15)

st.set_page_config(page_title="Planetary Seismic Event Detector", layout="wide")
st.title("Planetary Seismic Event Detection")
st.caption(
    "Lightweight 1D CNN trained on Apollo (Moon) and InSight (Mars) data. "
    "Upload a single-channel trace (miniSEED / SAC / CSV) to see detected "
    "events, arrival times, and confidence."
)


@st.cache_resource
def get_model(body: str):
    # local training output first, then the checkpoint shipped in the repo
    # (what free hosting uses — runs/ is gitignored)
    path = RUNS_DIR / body / "best.pt"
    if not path.exists():
        path = PROJECT_ROOT / "models" / f"{body}_best.pt"
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = SeisCNN()
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


col_a, col_b = st.columns(2)
body = col_a.selectbox("Model", ["lunar", "mars"],
                       help="Which body the detector was trained on")
threshold = col_b.slider("Detection threshold", 0.1, 0.95, 0.5, 0.05)

model = get_model(body)
if model is None:
    st.error(f"No trained checkpoint at runs/{body}/best.pt — train the model first.")
    st.stop()
st.caption(f"Model: {count_params(model):,} parameters, CPU inference.")

up = st.file_uploader("Seismic trace", type=["mseed", "sac", "csv"])
if up is None:
    st.info("Upload a trace to begin. Apollo/InSight samples are in the "
            "Space Apps 2024 packet.")
    st.stop()

if up.size > MAX_UPLOAD_MB * 1024 * 1024:
    st.error(f"File too large (>{MAX_UPLOAD_MB} MB).")
    st.stop()

try:
    raw, rate, _ = load_trace(up, filename=up.name)
except Exception as e:
    st.error(f"Could not read this file as a seismic trace: {e}")
    st.stop()

if len(raw) > MAX_SAMPLES:
    st.warning(f"Trace truncated to first {MAX_SAMPLES:,} samples.")
    raw = raw[:MAX_SAMPLES]
if len(raw) < 100:
    st.error("Trace too short.")
    st.stop()

with st.spinner("Preprocessing + inference..."):
    proc, prate = preprocess(raw, rate, CFG.preproc)
    dets, win_starts, win_probs = detect_events(model, proc, prate, CFG, threshold,
                                                suppress_sec=CODA_SEC[body])

t = np.arange(len(proc)) / prate
# downsample plot to keep the browser responsive
step = max(1, len(proc) // 200_000)
fig = go.Figure()
fig.add_trace(go.Scattergl(x=t[::step], y=proc[::step], mode="lines",
                           name="trace", line=dict(width=0.7, color="#4a7ebb")))
for d in dets:
    fig.add_vline(x=d.time_sec, line_color="red", line_dash="dash")
    fig.add_annotation(x=d.time_sec, y=1.02, yref="paper", showarrow=False,
                       text=f"{d.confidence:.2f}", font=dict(color="red", size=11))
fig.update_layout(height=420, xaxis_title="time (s)",
                  yaxis_title="filtered amplitude",
                  margin=dict(l=40, r=20, t=30, b=40))
st.plotly_chart(fig, use_container_width=True)

if dets:
    st.subheader(f"{len(dets)} event(s) detected")
    st.dataframe(
        [{"arrival time (s)": round(d.time_sec, 1),
          "arrival (h:m:s)": f"{int(d.time_sec // 3600)}:{int(d.time_sec % 3600 // 60):02d}:{int(d.time_sec % 60):02d}",
          "confidence": round(d.confidence, 3)} for d in dets],
        use_container_width=True,
    )
else:
    st.success("No events detected in this trace at the current threshold. "
               "Lower the threshold to search for weaker candidates.")

with st.expander("Window-level probabilities"):
    pfig = go.Figure(go.Scatter(x=win_starts, y=win_probs, mode="lines+markers",
                                line=dict(color="#e07a30")))
    pfig.add_hline(y=threshold, line_dash="dot")
    pfig.update_layout(height=220, xaxis_title="window start (s)",
                       yaxis_title="P(event)", yaxis_range=[0, 1],
                       margin=dict(l=40, r=20, t=10, b=40))
    st.plotly_chart(pfig, use_container_width=True)

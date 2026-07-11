"""Planetary seismic event detection — analysis + on-lander triage demo.

Run locally:  streamlit run app/streamlit_app.py
Theme: .streamlit/config.toml (cream + orange). Palette validated for CVD
and contrast: events #C74E00, review queue #2E6FB8, waveform ink #3D3833.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import CODA_SEC, DEFAULT as CFG, PROJECT_ROOT, RUNS_DIR
from planetseis.detect import (Detection, cluster_detections, detect_events,
                               detect_events_mc)
from planetseis.model import ARCHS, SeisCNN, count_params
from planetseis.preprocessing import load_trace, normalize_window, preprocess

# validated palette (cream surface)
INK = "#3D3833"
EVENT = "#C74E00"      # accepted detection
REVIEW = "#2E6FB8"     # routed to human review
GRID = "#E4D5B0"

MAX_UPLOAD_MB = 200
MAX_SAMPLES = 12_000_000
REVIEW_LOW, STD_REVIEW, MC_PASSES = 0.5, 0.15, 15

st.set_page_config(page_title="Planetary Seismic Triage", page_icon="🌒", layout="wide")

st.markdown(
    f"""<div style="border-left: 6px solid {EVENT}; padding: 0.2rem 1rem; margin-bottom: 1rem;">
    <h1 style="margin-bottom:0.1rem;">Planetary Seismic Event Detection</h1>
    <p style="color:#6B5E4A; margin:0;">Lightweight 1D CNN · Apollo (Moon) + InSight (Mars) ·
    uncertainty-aware triage for bandwidth-limited landers</p></div>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model(body: str):
    path = RUNS_DIR / body / "best.pt"
    if not path.exists():
        path = PROJECT_ROOT / "models" / f"{body}_best.pt"
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = SeisCNN(channels=ARCHS[ckpt.get("arch", "base")])
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@st.cache_data
def catalog_lookup(stem: str) -> list[dict]:
    """NASA catalog rows for an uploaded file, if it is a catalogued trace.

    This is the provenance proof: the arrival times were picked by human
    seismologists and published by NASA — the model never sees them.
    """
    import pandas as pd
    rows = []
    packet = PROJECT_ROOT / "data" / "raw" / "space_apps_2024_seismic_detection" / "data"
    for cat_name, raw_path in (
        ("apollo12_catalog_GradeA_final.csv",
         packet / "lunar" / "training" / "catalogs"),
        ("Mars_InSight_training_catalog_final.csv",
         packet / "mars" / "training" / "catalogs"),
    ):
        for base in (PROJECT_ROOT / "demo_data" / cat_name, raw_path / cat_name):
            if not base.exists():
                continue
            cat = pd.read_csv(base)
            fcol = next(c for c in cat.columns if "filename" in c.lower())
            rcol = next(c for c in cat.columns if "rel" in c.lower())
            acol = next((c for c in cat.columns if "abs" in c.lower()), None)
            tcol = next((c for c in cat.columns if "type" in c.lower()), None)
            for _, r in cat.iterrows():
                cstem = str(r[fcol]).replace(".csv", "").replace(".mseed", "")
                if cstem == stem:
                    rows.append({
                        "time_rel": float(r[rcol]),
                        "time_abs": str(r[acol]) if acol else "",
                        "type": str(r[tcol]) if tcol else "",
                        "evid": str(r.get("evid", "")),
                    })
            break
    return rows


def waveform_fig(t, y, dets: list[Detection], title="", truth: list[dict] | None = None):
    step = max(1, len(y) // 30_000)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t[::step], y=y[::step], mode="lines", name="trace",
                             line=dict(width=0.8, color=INK), hoverinfo="skip"))
    for tr in truth or []:
        fig.add_vline(x=tr["time_rel"], line_color=INK, line_dash="dot", line_width=2)
        fig.add_annotation(x=tr["time_rel"], y=-0.06, yref="paper", showarrow=False,
                           text="NASA catalog", font=dict(color=INK, size=10))
    for d in dets:
        color = REVIEW if d.needs_review else EVENT
        dash = "dot" if d.needs_review else "dash"
        fig.add_vline(x=d.time_sec, line_color=color, line_dash=dash, line_width=2)
        label = f"{d.confidence:.2f}" + (f"±{d.uncertainty:.2f}" if d.uncertainty else "")
        fig.add_annotation(x=d.time_sec, y=1.04, yref="paper", showarrow=False,
                           text=label, font=dict(color=color, size=11))
    fig.update_layout(
        title=title, height=380, xaxis_title="time (s)", yaxis_title="filtered amplitude",
        margin=dict(l=40, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID),
        showlegend=False,
    )
    return fig


def detections_table(dets: list[Detection]):
    return [{
        "arrival (s)": round(d.time_sec, 1),
        "arrival (h:m:s)": f"{int(d.time_sec // 3600)}:{int(d.time_sec % 3600 // 60):02d}:{int(d.time_sec % 60):02d}",
        "confidence": round(d.confidence, 3),
        "uncertainty (±)": round(d.uncertainty, 3) if d.uncertainty else None,
        "status": "🔎 review" if d.needs_review else "✅ event",
    } for d in dets]


tab_analyze, tab_triage, tab_about = st.tabs(
    ["📈 Analyze a trace", "🛰️ On-lander triage simulation", "ℹ️ Model & results"])

# ---------------------------------------------------------------- analyze tab
with tab_analyze:
    c1, c2, c3 = st.columns([1, 1, 1])
    body = c1.selectbox("Model", ["lunar", "mars"], help="Body the detector was trained on")
    threshold = c2.slider("Accept threshold", 0.1, 0.99, 0.9, 0.01)
    use_mc = c3.toggle("Uncertainty mode (MC-Dropout)", value=False,
                       help=f"{MC_PASSES} stochastic passes; borderline detections "
                            "are routed to a review queue instead of accepted/dropped")

    model = get_model(body)
    if model is None:
        st.error(f"No checkpoint for '{body}' — train first or add models/{body}_best.pt")
        st.stop()
    st.caption(f"{count_params(model):,} parameters · CPU inference · "
               f"window ≈ {CFG.window.n_samples / 6.625 / 60:.0f} min")

    up = st.file_uploader("Seismic trace (miniSEED / SAC / CSV)", type=["mseed", "sac", "csv"])
    if up is None:
        st.info("Upload a single-channel trace to begin — Apollo and InSight files "
                "from the Space Apps 2024 packet work as-is.")
    elif up.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"File too large (>{MAX_UPLOAD_MB} MB).")
    else:
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
            if use_mc:
                dets, ws, wp, wstd = detect_events_mc(
                    model, proc, prate, CFG, threshold, suppress_sec=CODA_SEC[body],
                    n_passes=MC_PASSES, review_band=(REVIEW_LOW, None), std_review=STD_REVIEW)
            else:
                dets, ws, wp = detect_events(model, proc, prate, CFG, threshold,
                                             suppress_sec=CODA_SEC[body])
                wstd = None

        stem = Path(up.name).stem
        truth = catalog_lookup(stem)
        t = np.arange(len(proc)) / prate
        st.plotly_chart(waveform_fig(t, proc, dets, truth=truth), use_container_width=True)

        if truth:
            for tr in truth:
                nearest = min((d for d in dets), key=lambda d: abs(d.time_sec - tr["time_rel"]),
                              default=None)
                err = f" — model detected {abs(nearest.time_sec - tr['time_rel']):.0f} s away" if nearest else ""
                st.info(f"**Ground truth (NASA catalog):** event `{tr['evid']}` "
                        f"({tr['type']}) at **{tr['time_rel']:.0f} s** "
                        f"({tr['time_abs']} UTC){err}. The dotted black line is the "
                        "human pick; the model never sees the catalog.")

        accepted = [d for d in dets if not d.needs_review]
        review = [d for d in dets if d.needs_review]
        if dets:
            st.subheader(f"{len(accepted)} event(s)" +
                         (f" · {len(review)} for review" if review else ""))
            st.dataframe(detections_table(dets), use_container_width=True)
        else:
            st.success("No events detected at this threshold — lower it to search "
                       "for weaker candidates.")

        with st.expander("Window-level probabilities"):
            pfig = go.Figure()
            if wstd is not None:
                pfig.add_trace(go.Scatter(
                    x=np.concatenate([ws, ws[::-1]]),
                    y=np.concatenate([np.clip(wp + wstd, 0, 1),
                                      np.clip(wp - wstd, 0, 1)[::-1]]),
                    fill="toself", fillcolor="rgba(199,78,0,0.18)",
                    line=dict(width=0), hoverinfo="skip", name="±1σ (MC)"))
            pfig.add_trace(go.Scatter(x=ws, y=wp, mode="lines", name="P(event)",
                                      line=dict(color=EVENT, width=2)))
            pfig.add_hline(y=threshold, line_dash="dot", line_color=INK)
            pfig.update_layout(height=240, xaxis_title="window start (s)",
                               yaxis_title="P(event)", yaxis_range=[0, 1.02],
                               margin=dict(l=40, r=20, t=10, b=40),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID))
            st.plotly_chart(pfig, use_container_width=True)

# ----------------------------------------------------------------- triage tab
with tab_triage:
    st.markdown(
        "Simulates the mission scenario: the seismometer streams continuously, the "
        "detector screens every window **on the lander**, and only flagged windows "
        "are queued for downlink. Confident detections are auto-accepted; borderline "
        "ones light up the **review queue** instead of being silently kept or dropped.")

    demo_files = sorted((PROJECT_ROOT / "demo_data").glob("*.npz"))
    if not demo_files:
        st.error("demo_data/ is empty — run scripts/build_windows.py and copy a "
                 "cached trace in.")
        st.stop()

    names = {p.name: p for p in demo_files}
    d1, d2 = st.columns([3, 1])
    pick_name = d1.selectbox("Demo stream (held-out day, never trained on)", list(names))
    speed = d2.select_slider("Playback", ["1×", "2×", "4×", "8×"], value="4×")
    sim_body = "mars" if pick_name.lower().startswith("xb") else "lunar"
    model = get_model(sim_body)

    @st.cache_data(show_spinner="Scoring stream (one-time)...")
    def score_stream(name: str):
        z = np.load(names[name])
        trace, rate = z["trace"], float(z["rate"])
        picks = list(z["picks"])
        n, hop = CFG.window.n_samples, CFG.window.hop
        m = get_model("mars" if name.lower().startswith("xb") else "lunar")
        starts = list(range(0, max(len(trace) - n + 1, 1), hop))
        # cheap deterministic screen on every window
        with torch.no_grad():
            m.eval()
            probs, offs = [], []
            for b in range(0, len(starts), 256):
                x = torch.from_numpy(np.stack([
                    normalize_window(trace[s:s + n]) for s in starts[b:b + 256]
                ])).unsqueeze(1)
                logit, off, _ = m(x)
                probs.append(torch.sigmoid(logit).numpy())
                offs.append(off.numpy())
        probs, offs = np.concatenate(probs), np.concatenate(offs)
        # expensive MC uncertainty only on candidate windows (on-lander budget)
        stds = np.zeros_like(probs)
        cand = np.where(probs >= REVIEW_LOW)[0]
        if len(cand):
            from planetseis.detect import _enable_mc_dropout
            _enable_mc_dropout(m)
            reps = []
            with torch.no_grad():
                x = torch.from_numpy(np.stack([
                    normalize_window(trace[starts[i]:starts[i] + n]) for i in cand
                ])).unsqueeze(1)
                for _ in range(MC_PASSES):
                    logit, _, _ = m(x)
                    reps.append(torch.sigmoid(logit).numpy())
            m.eval()
            reps = np.stack(reps)
            probs[cand] = reps.mean(0)
            stds[cand] = reps.std(0)
        return trace, rate, picks, np.array(starts), probs, offs, stds

    trace, rate, picks, starts, probs, offs, stds = score_stream(pick_name)
    n = CFG.window.n_samples
    win_sec = n / rate
    start_secs = starts / rate
    total_windows = len(starts)

    if st.button("▶ Start stream", type="primary"):
        chunk = max(1, total_windows // 60)
        delay = {"1×": 0.6, "2×": 0.3, "4×": 0.15, "8×": 0.05}[speed]
        metrics_box = st.empty()
        plot_box = st.empty()
        queue_box = st.empty()
        for upto in range(chunk, total_windows + chunk, chunk):
            upto = min(upto, total_windows)
            p, o, s_ = probs[:upto], offs[:upto], start_secs[:upto]
            dets = cluster_detections(p, o, s_, win_sec, REVIEW_LOW,
                                      suppress_sec=CODA_SEC[sim_body], stds=stds[:upto])
            for d in dets:
                d.needs_review = not (d.confidence >= 0.9 and d.uncertainty <= STD_REVIEW)
            accepted = [d for d in dets if not d.needs_review]
            review = [d for d in dets if d.needs_review]
            flagged = int((p >= REVIEW_LOW).sum())
            reduction = 100 * (1 - flagged / upto)
            hours = (s_[-1] + win_sec) / 3600

            with metrics_box.container():
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Stream scanned", f"{hours:.1f} h")
                m2.metric("Events (auto-accept)", len(accepted))
                m3.metric("Review queue", len(review))
                m4.metric("Downlink reduction", f"{reduction:.1f} %",
                          help="Only flagged windows are transmitted to Earth")
            t_end = int((s_[-1] + win_sec) * rate)
            t = np.arange(t_end) / rate
            plot_box.plotly_chart(
                waveform_fig(t, trace[:t_end], dets, title=f"{pick_name} — live"),
                use_container_width=True)
            if review:
                queue_box.dataframe(detections_table(review), use_container_width=True)
            time.sleep(delay)

        st.success(
            f"Stream complete: {total_windows} windows screened, "
            f"{len(accepted)} auto-accepted event(s), {len(review)} sent to review, "
            f"{100 * (1 - int((probs >= REVIEW_LOW).sum()) / total_windows):.1f}% of the "
            f"stream never needed downlink.")
        if picks:
            st.caption("Catalog ground truth for this file: pick(s) at " +
                       ", ".join(f"{p:.0f} s" for p in picks))

# ------------------------------------------------------------------ about tab
with tab_about:
    st.markdown(f"""
**Model** — dual-head 1D CNN, {count_params(get_model('lunar')):,} parameters
(0.49 MB), 11 ms CPU inference per ~21-minute window. Detection head +
soft-argmax arrival head. Trained on Apollo 12 Grade-A catalogued events
(NASA Space Apps 2024 packet), hard-negative mining, coda-aware labeling.

**Headline results** (continuous held-out traces, ±120 s tolerance)

| Experiment | Precision | Recall | F1 | MAE |
|---|---|---|---|---|
| Lunar → Lunar (this CNN, 118K) | 0.556 | 0.526 | 0.541 | 40 s |
| Lunar → Lunar (STA/LTA, tuned) | 0.116 | 0.421 | 0.182 | 76 s |
| Lunar → Lunar (PhaseNet 268K, zero-shot) | 0.000 | 0.000 | 0.000 | — |
| Lunar → Lunar (EQTransformer 376K, zero-shot) | 0.006 | 0.053 | 0.011 | 106 s |
| Lunar → Mars (transfer) | 0.000 | 0.000 | 0.000 | — |
| Mars → Lunar (transfer) | 0.006 | 0.080 | 0.012 | 62 s |

Cross-body transfer **collapses in both directions** — reported as a finding:
compact detectors do not cross planetary noise regimes without adaptation.

**Uncertainty** — MC-Dropout ({MC_PASSES} passes) attaches an epistemic ±σ to every
detection; borderline ones are routed to the review queue rather than silently
accepted or dropped — built for Mars-scale label scarcity.
""")

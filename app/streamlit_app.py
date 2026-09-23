"""Planetary seismic event detection — analysis + on-lander triage demo.

Run locally:  streamlit run app/streamlit_app.py
Theme: .streamlit/config.toml (cream + orange). Palette validated for CVD
and contrast: events #C74E00, review queue #2E6FB8, waveform ink #3D3833.
"""
from __future__ import annotations

import sys
import time
import hashlib
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import CODA_SEC, DEFAULT as CFG, PROJECT_ROOT, RUNS_DIR
from planetseis.detect import (Detection, cluster_detections, detect_events,
                               detect_events_mc, _window_starts)
from planetseis.detect_spec import (denoise_trace, detect_events_spec,
                                    detect_events_spec_mc)
from planetseis.model import ARCHS, SeisCNN, count_params
from planetseis.preprocessing import load_trace, normalize_window, preprocess
from planetseis.unet import UNET_ARCHS, SpecUNet

# validated palette (cream surface)
INK = "#3D3833"
EVENT = "#C74E00"      # accepted detection
REVIEW = "#2E6FB8"     # routed to human review
GRID = "#E4D5B0"

MAX_UPLOAD_MB = 200
MAX_SAMPLES = 12_000_000
# Gap-filled span limit for uploads; longer valid traces are still truncated
# to MAX_SAMPLES below, but a tiny file cannot request a multi-GB gap fill.
MAX_MERGED_SAMPLES = 4 * MAX_SAMPLES
REVIEW_LOW, STD_REVIEW, MC_PASSES = 0.5, 0.15, 15
UNET_MC_PASSES = 10
# Bound CPU memory use for long uploads and small hosting instances.
CNN_BATCH_SIZE, UNET_BATCH_SIZE = 64, 8
OPERATING_POINTS = {
    ("SpecUNet", "lunar"): (0.25, 430.0),
    ("SpecUNet", "mars"): (0.30, 240.0),
    ("SeisCNN", "lunar"): (0.97, None),
    ("SeisCNN", "mars"): (0.50, None),
}
LUNAR_CHECKPOINTS = {
    "SeisCNN": ("lunar_grouped_v1", "lunar_grouped_v1_seed42.pt"),
    "SpecUNet": ("unet_lunar_grouped_v1", "unet_lunar_grouped_v1_seed42.pt"),
}

st.set_page_config(page_title="Planetary Seismic Triage", page_icon="🌒", layout="wide")

st.markdown(
    f"""<div style="border-left: 6px solid {EVENT}; padding: 0.2rem 1rem; margin-bottom: 1rem;">
    <h1 style="margin-bottom:0.1rem;">Planetary Seismic Event Detection</h1>
    <p style="color:#6B5E4A; margin:0;">SeisCNN + SpecUNet · Apollo (Moon) + InSight (Mars) ·
    event detection, waveform denoising, and on-lander triage</p></div>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model(body: str):
    if body == "lunar":
        return get_lunar_model("SeisCNN")
    path = RUNS_DIR / body / "best.pt"
    if not path.exists():
        path = PROJECT_ROOT / "models" / f"{body}_best.pt"
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = SeisCNN(channels=ARCHS[ckpt.get("arch", "base")])
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.deployment_metadata = historical_model_metadata(ckpt, path, "SeisCNN")
    return model


@st.cache_resource
def get_unet(body: str):
    """Spectrogram U-Net checkpoint. Mars uses the mars_ext-trained model
    (MQS v14 labels); lunar uses the injection-trained lunar model."""
    if body == "lunar":
        return get_lunar_model("SpecUNet")
    names = ["unet_mars_ext", "unet_mars"] if body == "mars" else [f"unet_{body}"]
    for name in names:
        for path in (RUNS_DIR / name / "best.pt",
                     PROJECT_ROOT / "models" / f"{name}_best.pt"):
            if path.exists():
                ckpt = torch.load(path, map_location="cpu", weights_only=True)
                model = SpecUNet(base=UNET_ARCHS[ckpt.get("arch", "base")])
                model.load_state_dict(ckpt["model"])
                model.eval()
                model.deployment_metadata = historical_model_metadata(ckpt, path, "SpecUNet")
                return model
    return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_lunar_model(detector: str):
    """Load only the designated corrected seed and its verified operating point."""
    run_name, filename = LUNAR_CHECKPOINTS[detector]
    path = next((p for p in (RUNS_DIR / run_name / "best.pt", PROJECT_ROOT / "models" / filename)
                 if p.is_file()), None)
    if path is None:
        return None
    metadata_path = PROJECT_ROOT / "models" / "lunar_grouped_v1_seed42.json"
    if not metadata_path.is_file():
        raise FileNotFoundError("Corrected lunar checkpoint metadata is missing from models/.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    entry = metadata["checkpoints"][detector]
    threshold, min_duration = OPERATING_POINTS[(detector, "lunar")]
    point = entry["operating_point"]
    if (metadata.get("benchmark_id") != "lunar_grouped_v1"
            or metadata.get("seed") != 42 or entry.get("seed") != 42
            or entry.get("filename") != filename
            or point.get("threshold") != threshold
            or point.get("min_dur_sec") != min_duration):
        raise ValueError("Corrected lunar model metadata does not match the designated seed and settings.")
    if file_sha256(path) != entry["sha256"]:
        raise ValueError("Corrected lunar checkpoint hash does not match its recorded evaluation.")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if (ckpt.get("benchmark_id") != metadata["benchmark_id"]
            or ckpt.get("data_manifest_sha256") != metadata["data_manifest_sha256"]
            or ckpt.get("seed") != 42 or ckpt.get("initialized_from_scratch") is not True):
        raise ValueError("Corrected lunar checkpoint provenance does not match its metadata.")
    model = (SeisCNN(channels=ARCHS[ckpt.get("arch", "base")]) if detector == "SeisCNN"
             else SpecUNet(base=UNET_ARCHS[ckpt.get("arch", "base")]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.deployment_metadata = {
        "benchmark_id": metadata["benchmark_id"], "seed": 42,
        "checkpoint_filename": filename, "checkpoint_sha256": entry["sha256"],
        "data_manifest_sha256": metadata["data_manifest_sha256"],
        "epoch": ckpt.get("epoch"), "operating_point": dict(point),
        "selection_sha256": entry["selection_sha256"],
        "evaluation_sha256": entry["evaluation_sha256"],
        "checkpoint_selection": metadata["checkpoint_selection"],
    }
    return model


def historical_model_metadata(ckpt, path: Path, detector: str):
    return {
        "benchmark_id": "historical_mars_ext" if detector == "SpecUNet" else "historical_mars_packet",
        "seed": ckpt.get("seed", ckpt.get("config", {}).get("seed")),
        "checkpoint_filename": path.name, "checkpoint_sha256": file_sha256(path),
    }


@st.cache_data
def catalog_lookup(stem: str) -> list[dict]:
    """NASA catalog rows for an uploaded file, if it is a catalogued trace.

    Arrival times are used for the displayed comparison, not passed into
    inference. Training uses catalog labels, as described in Model & results.
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


def detections_table(dets: list[Detection], is_unet: bool = False):
    score_label = "mask-energy score" if is_unet else "confidence"
    return [{
        "arrival (s)": round(d.time_sec, 1),
        "arrival (h:m:s)": f"{int(d.time_sec // 3600)}:{int(d.time_sec % 3600 // 60):02d}:{int(d.time_sec % 60):02d}",
        score_label: round(d.confidence, 3),
        "uncertainty (±)": round(d.uncertainty, 3) if d.uncertainty else None,
        "status": "🔎 review" if d.needs_review else "✅ event",
    } for d in dets]


def demo_body(name: str) -> str:
    return "mars" if name.lower().startswith("xb") else "lunar"


def load_demo(path: Path):
    """Demo NPZ traces are already filtered and resampled by build_windows."""
    with np.load(path, allow_pickle=False) as z:
        trace = np.asarray(z["trace"], dtype=np.float32)
        rate = float(z["rate"])
        picks = np.asarray(z["picks"], dtype=float).tolist()
    validate_trace(trace, rate)
    if not np.isclose(rate, CFG.preproc.target_rate_hz):
        raise ValueError("Demo sample rate does not match the preprocessed model input rate.")
    return trace, rate, picks


def validate_trace(trace: np.ndarray, rate: float):
    if trace.ndim != 1:
        raise ValueError("Choose a single-channel, one-dimensional trace.")
    if len(trace) < 100:
        raise ValueError("Trace too short: at least 100 samples are required.")
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("The sample rate must be a positive, finite number.")
    if not np.isfinite(trace).all():
        raise ValueError("The trace contains non-finite samples.")


def analyze_trace(source, upload, demo_path, body, detector, threshold,
                  min_dur, use_mc, show_denoised):
    """Perform one explicitly requested analysis and retain its exact settings."""
    is_unet = detector == "SpecUNet"
    notes = []
    if source == "Built-in demo":
        raw, rate, picks = load_demo(demo_path)
        name = demo_path.name
        # Cached demos have already passed the shared preprocessing pipeline.
        proc, prate = raw, rate
    else:
        if upload is None:
            raise ValueError("Upload a seismic trace before selecting Analyze.")
        if upload.size > MAX_UPLOAD_MB * 1024 * 1024:
            raise ValueError(f"File too large (>{MAX_UPLOAD_MB} MB).")
        upload.seek(0)
        raw, rate, _ = load_trace(upload, filename=upload.name,
                                  max_samples=MAX_MERGED_SAMPLES)
        validate_trace(raw, rate)
        if len(raw) > MAX_SAMPLES:
            notes.append(f"Trace truncated to the first {MAX_SAMPLES:,} input samples.")
            raw = raw[:MAX_SAMPLES]
        name, picks = upload.name, []
        proc, prate = preprocess(raw, rate, CFG.preproc)
    validate_trace(proc, prate)
    if len(proc) > MAX_SAMPLES:
        notes.append(f"Analysis limited to the first {MAX_SAMPLES:,} processed samples.")
        proc = proc[:MAX_SAMPLES]
    if is_unet and len(proc) < CFG.window.n_samples:
        required_minutes = CFG.window.n_samples / prate / 60
        raise ValueError(
            f"SpecUNet requires at least {CFG.window.n_samples:,} processed samples "
            f"(about {required_minutes:.1f} minutes). Upload a longer trace or use SeisCNN.")

    model = get_unet(body) if is_unet else get_model(body)
    if model is None:
        raise FileNotFoundError(
            f"No checkpoint for {detector} / {body}. Add its checkpoint under models/ "
            "or select an available detector.")
    if is_unet:
        from planetseis.spectral import SEC_PER_BIN
        if use_mc:
            dets, wp, wstd = detect_events_spec_mc(
                model, proc, prate, CFG, threshold,
                suppress_sec=CODA_SEC[body], n_passes=UNET_MC_PASSES,
                min_dur_sec=min_dur, batch_size=UNET_BATCH_SIZE)
        else:
            dets, wp = detect_events_spec(
                model, proc, prate, CFG, threshold,
                suppress_sec=CODA_SEC[body], min_dur_sec=min_dur,
                batch_size=UNET_BATCH_SIZE)
            wstd = None
        ws = np.arange(len(wp)) * SEC_PER_BIN
    elif use_mc:
        dets, ws, wp, wstd = detect_events_mc(
            model, proc, prate, CFG, threshold, suppress_sec=CODA_SEC[body],
            n_passes=MC_PASSES, review_band=(min(REVIEW_LOW, threshold), None),
            std_review=STD_REVIEW, batch_size=CNN_BATCH_SIZE)
    else:
        dets, ws, wp = detect_events(
            model, proc, prate, CFG, threshold, suppress_sec=CODA_SEC[body],
            batch_size=CNN_BATCH_SIZE)
        wstd = None
    den = (denoise_trace(model, proc, prate, batch_size=UNET_BATCH_SIZE)
           if is_unet and show_denoised else None)
    try:
        truth = catalog_lookup(Path(name).stem)
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        truth = []
        notes.append(f"Catalog overlay unavailable: {exc}")
    if source == "Built-in demo" and picks:
        # Bundled lunar picks use catalog UTC minus the actual trace start.
        # Keep these corrected references even when the older CSV also matches.
        truth = [{**next((row for row in truth if abs(row["time_rel"] - p) < 1.0),
                        {"time_abs": "", "type": "", "evid": ""}), "time_rel": p}
                 for p in picks]
    duration = len(proc) / prate
    truth = [row for row in truth if 0 <= row["time_rel"] < duration]
    default_thr, default_dur = OPERATING_POINTS[(detector, body)]
    exploratory = (use_mc or not np.isclose(threshold, default_thr)
                   or min_dur != default_dur
                   or (source == "Built-in demo" and demo_body(name) != body))
    return dict(name=name, body=body, detector=detector, threshold=threshold,
                min_dur=min_dur, use_mc=use_mc, exploratory=exploratory,
                proc=proc, prate=prate, dets=dets, ws=ws, wp=wp, wstd=wstd,
                den=den, truth=truth, notes=notes, params=count_params(model),
                model_provenance=dict(model.deployment_metadata))


def render_analysis_result(result):
    r = result
    is_unet = r["detector"] == "SpecUNet"
    duration_text = f' · minimum event duration {r["min_dur"]:g} s' if is_unet else ""
    st.subheader("Last completed analysis")
    st.caption(
        f'{r["name"]} · {r["detector"]} / {r["body"]} · threshold {r["threshold"]:.2f}'
        f'{duration_text} · {"MC-Dropout" if r["use_mc"] else "deterministic"} · '
        f'{r["params"]:,} parameters · CPU inference')
    provenance = r["model_provenance"]
    seed_text = f' · seed {provenance["seed"]}' if provenance.get("seed") is not None else ""
    st.caption(f'Active model: {provenance["benchmark_id"]}{seed_text} · '
               f'{provenance["checkpoint_filename"]}')
    with st.expander("Model provenance"):
        st.json(provenance)
    if r["exploratory"]:
        st.info("Exploratory settings: these results do not use the deterministic "
                "operating point recorded for this model and benchmark.")
    for note in r["notes"]:
        st.warning(note)
    if is_unet:
        st.caption("Mask-energy scores measure the predicted event mask; they are "
                   "not calibrated probabilities of a real seismic event.")
        if r["use_mc"]:
            st.warning("SpecUNet MC-Dropout is exploratory. In the stored lunar "
                       "experiment, uncertainty did not distinguish false alarms "
                       "reliably; its review flags have not been validated for triage.")
    t = np.arange(len(r["proc"])) / r["prate"]
    st.plotly_chart(waveform_fig(t, r["proc"], r["dets"], truth=r["truth"]))
    for tr in r["truth"]:
        st.info(f'NASA catalog pick: **{tr["time_rel"]:.0f} s** from trace start. '
                "The dotted line is the human pick and is used only for comparison.")
    accepted = [d for d in r["dets"] if not d.needs_review]
    review = [d for d in r["dets"] if d.needs_review]
    if r["dets"]:
        st.subheader(f"{len(accepted)} event(s)" +
                     (f" · {len(review)} for review" if review else ""))
        st.dataframe(detections_table(r["dets"], is_unet))
    else:
        st.info("No events detected with these settings. A lower threshold is an "
                "exploratory search for weaker candidates.")
    label = "event-mask energy (band-integrated)" if is_unet else "P(event)"
    with st.expander("Event-mask detection curve" if is_unet else "Window-level probabilities"):
        pfig = go.Figure()
        ws, wp, wstd = r["ws"], r["wp"], r["wstd"]
        if wstd is not None:
            pfig.add_trace(go.Scatter(
                x=np.concatenate([ws, ws[::-1]]),
                y=np.concatenate([np.clip(wp + wstd, 0, 1),
                                  np.clip(wp - wstd, 0, 1)[::-1]]),
                fill="toself", fillcolor="rgba(199,78,0,0.18)",
                line=dict(width=0), hoverinfo="skip", name="±1σ (MC)"))
        pfig.add_trace(go.Scatter(x=ws, y=wp, mode="lines", name=label,
                                  line=dict(color=EVENT, width=2)))
        pfig.add_hline(y=r["threshold"], line_dash="dot", line_color=INK)
        pfig.update_layout(
            height=240, xaxis_title="time (s)" if is_unet else "window start (s)",
            yaxis_title=label, yaxis_range=[0, 1.02],
            margin=dict(l=40, r=20, t=10, b=40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID))
        st.plotly_chart(pfig)
    if r["den"] is not None:
        with st.expander("Denoised trace (event mask × spectrogram)", expanded=True):
            step = max(1, len(t) // 30_000)
            dfig = go.Figure()
            for y, name, color in ((r["proc"], "input", GRID),
                                   (r["den"], "denoised", EVENT)):
                dfig.add_trace(go.Scatter(x=t[::step], y=y[::step], mode="lines",
                                          name=name, line=dict(width=0.9, color=color)))
            dfig.update_layout(
                height=300, xaxis_title="time (s)", yaxis_title="amplitude",
                margin=dict(l=40, r=20, t=10, b=40),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(dfig)


def render_analyze():
    source = st.radio("Trace source", ["Built-in demo", "Upload trace"],
                      horizontal=True, key="analysis_source")
    demo_path = None
    hint_body = "lunar"
    if source == "Built-in demo":
        demo_files = sorted((PROJECT_ROOT / "demo_data").glob("*.npz"))
        if not demo_files:
            st.info("No built-in demos are available. Choose Upload trace to analyze a file.")
        else:
            names = {p.name: p for p in demo_files}
            name = st.selectbox("Demo trace", list(names), key="analysis_demo")
            demo_path = names[name]
            hint_body = demo_body(name)
            st.caption("Built-in traces are already preprocessed. Select Analyze to run detection.")
    else:
        st.caption("CSV files need numeric samples and preferably a relative-time column. "
                   "A sample-only CSV is assumed to be 6.625 Hz; use a relative-time "
                   "column when its actual sample rate differs.")
    c0, c1 = st.columns([2, 1])
    detector = c0.selectbox(
        "Detector", ["SpecUNet", "SeisCNN"], key="analysis_detector",
        help="SpecUNet surveys event-mask energy and denoises waveforms. "
             "SeisCNN is the supervised 1D detector used by the triage simulation.")
    body = c1.selectbox(
        "Model body", ["lunar", "mars"], index=int(hint_body == "mars"),
        key=f"analysis_body_{hint_body}",
        help="Select the body the model was trained on. Cross-body transfer performed poorly.")
    if demo_path and body != hint_body:
        st.warning(f"This is a {hint_body} demo with the {body} model selected. "
                   "Cross-body analysis is exploratory.")
    default_thr, default_dur = OPERATING_POINTS[(detector, body)]
    if body == "lunar":
        st.caption(f"Active lunar model: {detector} · lunar_grouped_v1 · seed 42. "
                   "Seed 42 is designated for the app; operating settings were selected on validation.")
    else:
        st.caption("Active Mars model: original InSight checkpoint and recorded Mars settings.")
    is_unet = detector == "SpecUNet"
    settings_key = f"{detector}_{body}"
    with st.form("analysis_form"):
        threshold = st.slider(
            "Accept threshold" if not is_unet else "Mask-energy threshold",
            0.05 if is_unet else 0.1, 0.80 if is_unet else 0.99,
            default_thr, 0.01, key=f"threshold_{settings_key}")
        min_dur = st.number_input(
            "Minimum event duration (seconds)", min_value=1.0, max_value=3600.0,
            value=default_dur, step=30.0, key=f"duration_{settings_key}",
            help="Minimum time the detection curve stays above threshold. "
                 "Recorded defaults: 430 s for corrected lunar seed 42, 240 s for Mars.") if is_unet else None
        passes = UNET_MC_PASSES if is_unet else MC_PASSES
        use_mc = st.checkbox(
            "Uncertainty mode (MC-Dropout, exploratory)", value=False,
            help=f"{passes} stochastic passes; requires additional CPU inference.",
            key=f"mc_{settings_key}")
        show_denoised = st.checkbox(
            "Also generate the denoised trace", value=False,
            help="Runs an additional U-Net pass only when Analyze is selected.",
            key="include_denoised") if is_unet else False
        upload = st.file_uploader(
            "Seismic trace (miniSEED / SAC / CSV)", type=["mseed", "sac", "csv"]
        ) if source == "Upload trace" else None
        submitted = st.form_submit_button(
            "Analyze trace", type="primary", disabled=source == "Built-in demo" and demo_path is None)
    if submitted:
        st.session_state.pop("analysis_result", None)
        try:
            with st.spinner("Analyzing trace..."):
                st.session_state["analysis_result"] = analyze_trace(
                    source, upload, demo_path, body, detector, threshold,
                    min_dur, use_mc, show_denoised)
        except Exception as exc:
            st.error(f"Could not analyze this trace: {exc}")
    if "analysis_result" in st.session_state:
        render_analysis_result(st.session_state["analysis_result"])


@st.cache_data(show_spinner="Scoring stream...")
def score_stream(path: str):
    trace, rate, picks = load_demo(Path(path))
    model = get_model(demo_body(Path(path).name))
    if model is None:
        raise FileNotFoundError("No SeisCNN checkpoint is available for this demo's body.")
    n, hop = CFG.window.n_samples, CFG.window.hop
    padded, starts = _window_starts(trace, n, hop)
    with torch.no_grad():
        model.eval()
        probs, offs = [], []
        for b in range(0, len(starts), CNN_BATCH_SIZE):
            x = torch.from_numpy(np.stack([
                normalize_window(padded[s:s + n]) for s in starts[b:b + CNN_BATCH_SIZE]
            ])).unsqueeze(1)
            logit, off, _ = model(x)
            probs.append(torch.sigmoid(logit).numpy())
            offs.append(off.numpy())
    probs, offs = np.concatenate(probs), np.concatenate(offs)
    stds = np.zeros_like(probs)
    cand = np.where(probs >= REVIEW_LOW)[0]
    if len(cand):
        from planetseis.detect import _enable_mc_dropout
        _enable_mc_dropout(model)
        try:
            with torch.no_grad():
                for b in range(0, len(cand), CNN_BATCH_SIZE):
                    indices = cand[b:b + CNN_BATCH_SIZE]
                    x = torch.from_numpy(np.stack([
                        normalize_window(padded[starts[i]:starts[i] + n]) for i in indices
                    ])).unsqueeze(1)
                    reps = []
                    for _ in range(MC_PASSES):
                        logit, _, _ = model(x)
                        reps.append(torch.sigmoid(logit).numpy())
                    reps = np.stack(reps)
                    probs[indices] = reps.mean(0)
                    stds[indices] = reps.std(0)
        finally:
            model.eval()
    return trace, rate, picks, np.array(starts), probs, offs, stds, dict(model.deployment_metadata)


def render_triage():
    st.markdown(
        "The SeisCNN detector screens the stream on the lander. Flagged windows "
        "are queued for downlink; borderline detections enter the review queue. "
        "This simulation uses its own exploratory MC-Dropout triage policy.")
    demo_files = sorted((PROJECT_ROOT / "demo_data").glob("*.npz"))
    if not demo_files:
        st.info("No demo streams are available. Upload analysis and model results remain available.")
        return
    names = {p.name: p for p in demo_files}
    d1, d2 = st.columns([3, 1])
    name = d1.selectbox("Demo stream", list(names), key="triage_demo")
    if demo_body(name) == "lunar":
        st.caption("Active triage model: SeisCNN · lunar_grouped_v1 · seed 42.")
    else:
        st.caption("Active triage model: original Mars SeisCNN checkpoint.")
    speed = d2.select_slider("Playback", ["1×", "2×", "4×", "8×"], value="4×")
    st.caption("Demo playback; scoring starts only when Start stream is selected. "
               "Auto-accept requires mean score ≥0.90 and uncertainty ≤0.15.")
    if not st.button("▶ Start stream", type="primary"):
        return
    try:
        trace, rate, picks, starts, probs, offs, stds, provenance = score_stream(str(names[name]))
    except Exception as exc:
        st.error(f"Could not start this stream: {exc}")
        return
    st.session_state["triage_model_provenance"] = provenance
    with st.expander("Triage model provenance"):
        st.json(provenance)
    n = CFG.window.n_samples
    win_sec = n / rate
    start_secs = starts / rate
    total_windows = len(starts)
    chunk = max(1, total_windows // 60)
    delay = {"1×": 0.6, "2×": 0.3, "4×": 0.15, "8×": 0.05}[speed]
    metrics_box, plot_box, queue_box = st.empty(), st.empty(), st.empty()
    for upto in range(chunk, total_windows + chunk, chunk):
        upto = min(upto, total_windows)
        p, o, s_ = probs[:upto], offs[:upto], start_secs[:upto]
        dets = cluster_detections(
            p, o, s_, win_sec, REVIEW_LOW,
            suppress_sec=CODA_SEC[demo_body(name)], stds=stds[:upto])
        for d in dets:
            d.needs_review = not (d.confidence >= 0.9 and d.uncertainty <= STD_REVIEW)
        accepted = [d for d in dets if not d.needs_review]
        review = [d for d in dets if d.needs_review]
        flagged = int((p >= REVIEW_LOW).sum())
        reduction = 100 * (1 - flagged / upto)
        t_end = min(len(trace), int((s_[-1] + win_sec) * rate))
        with metrics_box.container():
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Stream scanned", f"{t_end / rate / 3600:.1f} h")
            m2.metric("Events (auto-accept)", len(accepted))
            m3.metric("Review queue", len(review))
            m4.metric("Windows skipped for downlink", f"{reduction:.1f} %",
                      help="Fraction of overlapping windows not queued. "
                           "This is a window-count estimate, not measured bytes saved.")
        plot_box.plotly_chart(
            waveform_fig(np.arange(t_end) / rate, trace[:t_end], dets,
                         title=f"{name} — live"), key=f"stream_frame_{upto}")
        if review:
            queue_box.dataframe(detections_table(review))
        else:
            queue_box.empty()
        time.sleep(delay)
    st.success(
        f"Stream complete: {total_windows} windows screened, "
        f"{len(accepted)} auto-accepted event(s), {len(review)} sent to review, "
        f"{reduction:.1f}% of windows not queued for downlink.")
    if picks:
        st.caption("Catalog pick(s): " + ", ".join(f"{p:.0f} s" for p in picks))


def render_about():
    st.markdown("""
### Models and recorded results

**SeisCNN** — a supervised dual-head 1D CNN with approximately 118K parameters.
It classifies windows and estimates arrival times. **SpecUNet** — a 1.9M-parameter
spectrogram U-Net used for event surveys and waveform denoising.

Both use the shared preprocessing pipeline and approximately 21-minute windows
at 6.625 Hz. SpecUNet builds synthetic training mixtures from **catalog-derived
event templates** and noise, with synthetic masks as targets. It does not train
on real positive windows, but its template selection still uses catalog labels.

**Corrected lunar benchmark** (`lunar_grouped_v1`: acquisition-grouped split,
unioned picks, 21 test spans / 23 events, ±120 s tolerance). Both models were
retrained from scratch over five seeds, each at its own validation-selected
operating point:

| Detector | Seeds | F1 mean ± SD | Mean precision | Mean recall |
|---|---|---|---|---|
| SeisCNN, supervised | 5 | 0.531 ± 0.031 | 0.448 | 0.661 |
| SpecUNet, injection training | 5 | 0.408 ± 0.043 | 0.344 | 0.530 |
| Matched filter, validation-tuned | — | 0.200 | 0.429 | 0.130 |
| STA/LTA, validation-tuned | — | 0.168 | 0.111 | 0.348 |

SeisCNN is better at the seed level (Welch p = 0.0012). This app uses the
**corrected seed-42 checkpoints** for both lunar models. Seed 42 is designated
for deployment without ranking seeds by test score. Its validation-selected
settings are SeisCNN **0.97** and SpecUNet **0.25 / 430 s**. The single-seed test
F1 values are 0.561 and 0.412, respectively; the five-seed summary above reports
the variation across training runs. Mars retains its original checkpoints.

**Historical frozen benchmark** (continuous traces, ±120 s tolerance)

Split audit found two lunar test waveforms duplicated in training; the
corrected benchmark above supersedes these scores for performance claims.

| Experiment | Precision | Recall | F1 | MAE |
|---|---|---|---|---|
| Lunar → Lunar (SeisCNN, supervised) | 0.556 | 0.526 | 0.541 | 40 s |
| Lunar → Lunar (SpecUNet, injection training) | 0.355 | 0.579 | 0.440 | 68 s |
| Lunar → Lunar (matched filter, validation-tuned) | 0.333 | 0.158 | 0.214 | 56 s |
| Lunar → Lunar (STA/LTA, tuned) | 0.116 | 0.421 | 0.182 | 76 s |
| Mars_ext → Mars_ext (SpecUNet) | 1.000 | 0.233 | 0.378 | 34 s |

The historical lunar checkpoints remain archived for reproduction. Mars app
defaults remain SpecUNet threshold 0.30 / minimum duration 240 s and SeisCNN
threshold 0.50. Changing any recorded setting or enabling MC-Dropout creates
an exploratory analysis.

**Across training seeds**, recorded mean F1 was **0.498 ± 0.043** for SeisCNN
(3 seeds) and **0.372 ± 0.076** for screened SpecUNet (5 seeds). These are mean ±
sample standard deviation. The reported Welch test gave **p = 0.024**;
the earlier claim of statistical parity is not supported by this seed analysis.
These comparisons inherit the frozen split limitation above.

On the corrected split, across five seeds, 43% ± 13% of SpecUNet's benchmark
false positives lie within ±300 s of events in the full Nakamura catalog that
the Grade-A subset omits (1.5% expected by chance; historical split: 9 of 20).
These matches show that the small benchmark catalog omits real events. The
separate continuous-archive scan found poor recall at acceptable false-alarm
rates; survey deployment is not validated. Mask-energy scores are not calibrated
event probabilities. Cross-body transfer performed poorly in both directions.

**Uncertainty** — SeisCNN uses MC-Dropout to demonstrate a human-review queue.
On the corrected split its window uncertainty is 7.9 ± 3.5× higher on false
alarms than on true events (five seeds), but the review queue rarely held a
real event. SpecUNet uncertainty does not reliably separate false alarms (FP/TP
ratio 0.86 ± 0.43 across seeds); its MC mode is exploratory and should not be
read as validated triage confidence.

Sources within the project: `results/lunar_grouped_v1_seed_summary.json`,
`results/lunar_grouped_v1_secondary_summary.json`,
`results/seed_level_comparison.json`, `results/uncertainty_unet.json`, `results/unet_lunar_to_lunar.json`,
`results/unet_mars_ext_to_mars_ext.json`, and `benchmark/README.md`.
""")


tab_analyze, tab_triage, tab_about = st.tabs(
    ["📈 Analyze a trace", "🛰️ On-lander triage simulation", "ℹ️ Model & results"])
with tab_analyze:
    render_analyze()
with tab_triage:
    render_triage()
with tab_about:
    render_about()

"""App interaction contracts, using tiny local inputs and mocked inference.

The real Streamlit script executes in AppTest. No bundled checkpoint is read
and no neural-network inference runs in these tests.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import streamlit as st
import torch
from streamlit.testing.v1 import AppTest

from planetseis import config, detect, detect_spec, model, preprocessing, unet


APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
LUNAR_DEMO = "xa.test.lunar.npz"
MARS_DEMO = "XB.test.mars.npz"
MANIFEST = "a" * 64
LUNAR_FILES = {"SeisCNN": ("lunar_grouped_v1_seed42.pt", {"threshold": 0.97}),
               "SpecUNet": ("unet_lunar_grouped_v1_seed42.pt",
                            {"threshold": 0.25, "min_dur_sec": 430})}


def write_lunar_metadata(model_dir):
    """Corrected seed-42 metadata whose hashes match the mock checkpoint bytes."""
    checkpoints = {}
    for detector, (filename, point) in LUNAR_FILES.items():
        digest = hashlib.sha256((model_dir / filename).read_bytes()).hexdigest()
        checkpoints[detector] = {"filename": filename, "sha256": digest, "seed": 42,
                                 "operating_point": point, "selection_sha256": "s" * 64,
                                 "evaluation_sha256": "e" * 64}
    (model_dir / "lunar_grouped_v1_seed42.json").write_text(json.dumps({
        "benchmark_id": "lunar_grouped_v1", "seed": 42, "data_manifest_sha256": MANIFEST,
        "checkpoint_selection": "designated seed", "checkpoints": checkpoints}))


class Upload(io.BytesIO):
    def __init__(self, data: bytes, name: str = "trace.csv"):
        super().__init__(data)
        self.name = name
        self.size = len(data)


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    """Isolate file discovery, resource caches, and every inference entry point."""
    demo_dir = tmp_path / "demo_data"
    model_dir = tmp_path / "models"
    demo_dir.mkdir()
    model_dir.mkdir()
    samples = np.sin(np.arange(8192) * 0.1).astype(np.float32)
    for name in (LUNAR_DEMO, MARS_DEMO):
        np.savez(demo_dir / name, trace=samples, rate=6.625, picks=[20.0])
    for name in ("mars_best.pt", "unet_mars_ext_best.pt",
                 *(filename for filename, _ in LUNAR_FILES.values())):
        (model_dir / name).write_bytes(b"mock checkpoint " + name.encode())
    write_lunar_metadata(model_dir)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    checkpoint_load = Mock(return_value={
        "arch": "base", "model": {}, "benchmark_id": "lunar_grouped_v1",
        "data_manifest_sha256": MANIFEST, "seed": 42,
        "initialized_from_scratch": True, "epoch": 27})
    monkeypatch.setattr(torch, "load", checkpoint_load)
    forward = Mock()

    class TinyModel(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def load_state_dict(self, state_dict, strict=True):
            pass

        def forward(self, x):
            forward(x)
            # A deterministic all-negative stream avoids MC candidate forwards.
            return torch.full((len(x),), -4.0), torch.full((len(x),), 0.5), x[:, 0, :1]

    monkeypatch.setattr(model, "SeisCNN", TinyModel)
    monkeypatch.setattr(unet, "SpecUNet", TinyModel)
    detections = [detect.Detection(time_sec=50.0, confidence=0.7)]
    curve = np.linspace(0.1, 0.7, 16)
    spec = Mock(return_value=(detections, curve))
    spec_mc = Mock(return_value=(detections, curve, np.zeros_like(curve)))
    cnn = Mock(return_value=(detections, np.array([0.0]), np.array([0.7])))
    cnn_mc = Mock(return_value=(detections, np.array([0.0]), np.array([0.7]),
                               np.array([0.02])))
    denoise = Mock(side_effect=lambda _model, trace, rate, **kwargs: trace.copy())
    preproc = Mock(side_effect=AssertionError("A cached demo must not be preprocessed again"))
    monkeypatch.setattr(detect_spec, "detect_events_spec", spec)
    monkeypatch.setattr(detect_spec, "detect_events_spec_mc", spec_mc)
    monkeypatch.setattr(detect_spec, "denoise_trace", denoise)
    monkeypatch.setattr(detect, "detect_events", cnn)
    monkeypatch.setattr(detect, "detect_events_mc", cnn_mc)
    monkeypatch.setattr(preprocessing, "preprocess", preproc)
    st.cache_resource.clear()
    st.cache_data.clear()
    env = SimpleNamespace(
        root=tmp_path, demo_dir=demo_dir, model_dir=model_dir, samples=samples,
        spec=spec, spec_mc=spec_mc, cnn=cnn, cnn_mc=cnn_mc, denoise=denoise,
        preproc=preproc, forward=forward, checkpoint_load=checkpoint_load,
        run=lambda: AppTest.from_file(str(APP_PATH), default_timeout=60).run(),
    )
    yield env
    st.cache_resource.clear()
    st.cache_data.clear()


def click(app, label):
    return next(button for button in app.button if button.label == label).click().run()


def assert_about_available(app):
    assert not app.exception
    assert any("Models and recorded results" in element.value for element in app.markdown)


def select_analysis(app, detector="SpecUNet", body="lunar"):
    name = LUNAR_DEMO if body == "lunar" else MARS_DEMO
    app.selectbox(key="analysis_demo").select(name).run()
    app.selectbox(key="analysis_detector").select(detector).run()
    assert not app.exception
    return app


def test_idle_and_setting_changes_do_not_load_models_or_run_inference(app_env):
    app = app_env.run()
    assert_about_available(app)
    select_analysis(app, "SeisCNN", "lunar")
    app.select_slider[0].set_value("8×").run()
    for operation in (app_env.checkpoint_load, app_env.forward, app_env.spec,
                      app_env.spec_mc, app_env.cnn, app_env.cnn_mc,
                      app_env.denoise, app_env.preproc):
        operation.assert_not_called()


@pytest.mark.parametrize("detector,body,threshold,min_duration", [
    ("SpecUNet", "lunar", 0.25, 430.0),
    ("SpecUNet", "mars", 0.30, 240.0),
    ("SeisCNN", "lunar", 0.97, None),
    ("SeisCNN", "mars", 0.50, None),
])
def test_analyze_uses_published_defaults_and_does_not_reprocess_demo(
        app_env, detector, body, threshold, min_duration):
    app = select_analysis(app_env.run(), detector, body)
    assert app.slider(key=f"threshold_{detector}_{body}").value == pytest.approx(threshold)
    if min_duration is not None:
        assert app.number_input(key=f"duration_{detector}_{body}").value == min_duration
    click(app, "Analyze trace")
    assert_about_available(app)
    assert not app.error
    inference = app_env.spec if detector == "SpecUNet" else app_env.cnn
    inference.assert_called_once()
    args, kwargs = inference.call_args
    np.testing.assert_array_equal(args[1], app_env.samples)
    assert args[2] == 6.625
    assert args[4] == pytest.approx(threshold)
    assert kwargs["suppress_sec"] == config.CODA_SEC[body]
    if min_duration is not None:
        assert kwargs["min_dur_sec"] == min_duration
    app_env.preproc.assert_not_called()
    app_env.denoise.assert_not_called()
    app_env.forward.assert_not_called()  # hidden triage never scores the stream
    assert app.session_state["analysis_result"]["body"] == body
    assert not app.session_state["analysis_result"]["exploratory"]


def test_completed_result_survives_settings_reruns_without_relabeling_or_inference(app_env):
    app = select_analysis(app_env.run())
    click(app, "Analyze trace")
    app.selectbox(key="analysis_detector").select("SeisCNN").run()
    app.select_slider[0].set_value("8×").run()
    assert_about_available(app)
    app_env.spec.assert_called_once()
    app_env.cnn.assert_not_called()
    app_env.forward.assert_not_called()
    saved = app.session_state["analysis_result"]
    assert saved["detector"] == "SpecUNet"
    assert saved["threshold"] == 0.25
    assert saved["model_provenance"]["benchmark_id"] == "lunar_grouped_v1"
    assert saved["model_provenance"]["seed"] == 42
    assert any("SpecUNet / lunar" in element.value for element in app.caption)


def test_denoising_and_mc_only_run_when_submitted_and_are_marked_exploratory(app_env):
    app = select_analysis(app_env.run())
    app.checkbox(key="include_denoised").check().run()
    app.checkbox(key="mc_SpecUNet_lunar").check().run()
    app_env.denoise.assert_not_called()
    app_env.spec_mc.assert_not_called()
    click(app, "Analyze trace")
    assert_about_available(app)
    assert not app.error
    app_env.spec.assert_not_called()
    app_env.spec_mc.assert_called_once()
    app_env.denoise.assert_called_once()
    assert app_env.spec_mc.call_args.kwargs["min_dur_sec"] == 430.0
    assert app_env.spec_mc.call_args.kwargs["n_passes"] == 10
    assert app.session_state["analysis_result"]["exploratory"]
    assert any("Exploratory settings" in element.value for element in app.info)
    assert any("not been validated for triage" in element.value for element in app.warning)


@pytest.mark.parametrize("problem,message", [
    ("short", "Trace too short"),
    ("model_short", "SpecUNet requires"),
    ("malformed", "trace"),
])
def test_bad_demo_fails_locally_without_hiding_about(app_env, problem, message):
    path = app_env.demo_dir / LUNAR_DEMO
    if problem in ("short", "model_short"):
        length = 50 if problem == "short" else 1024
        np.savez(path, trace=np.zeros(length), rate=6.625, picks=[])
    else:
        np.savez(path, unexpected=np.zeros(100))
    app = select_analysis(app_env.run())
    click(app, "Analyze trace")
    assert_about_available(app)
    assert any(message in error.value for error in app.error)
    app_env.checkpoint_load.assert_not_called()
    app_env.spec.assert_not_called()


@pytest.mark.parametrize("data,message", [
    (b"samples\n1\n2\n3\n", "Trace too short"),
    (b"samples\nnot-a-number\n", "numeric"),
])
def test_invalid_upload_reports_error_and_keeps_about(app_env, monkeypatch, data, message):
    monkeypatch.setattr(st, "file_uploader", Mock(return_value=Upload(data)))
    app = app_env.run()
    app.radio(key="analysis_source").set_value("Upload trace").run()
    click(app, "Analyze trace")
    assert_about_available(app)
    assert any(message in error.value for error in app.error)
    app_env.preproc.assert_not_called()
    app_env.spec.assert_not_called()
    assert any("assumed to be 6.625 Hz" in element.value for element in app.caption)


def test_missing_checkpoints_fail_only_after_action_and_leave_about_available(app_env):
    for path in app_env.model_dir.glob("*.pt"):
        path.unlink()
    app = select_analysis(app_env.run())
    assert not app.error
    click(app, "Analyze trace")
    assert_about_available(app)
    assert any("No checkpoint" in error.value for error in app.error)
    click(app, "▶ Start stream")
    assert_about_available(app)
    assert any("No SeisCNN checkpoint" in error.value for error in app.error)
    app_env.spec.assert_not_called()
    app_env.forward.assert_not_called()


def test_missing_demos_keeps_upload_and_about_available(app_env):
    for path in app_env.demo_dir.glob("*.npz"):
        path.unlink()
    app = app_env.run()
    assert_about_available(app)
    assert next(button for button in app.button if button.label == "Analyze trace").disabled
    assert any("No demo streams" in element.value for element in app.info)
    app.radio(key="analysis_source").set_value("Upload trace").run()
    assert not next(button for button in app.button if button.label == "Analyze trace").disabled
    click(app, "Analyze trace")
    assert_about_available(app)
    assert any("Upload a seismic trace" in error.value for error in app.error)
    app_env.checkpoint_load.assert_not_called()


def test_start_stream_runs_only_the_requested_triage(app_env):
    app = app_env.run()
    app_env.forward.assert_not_called()
    click(app, "▶ Start stream")
    assert_about_available(app)
    assert not app.error
    app_env.forward.assert_called_once()
    app_env.spec.assert_not_called()
    assert any("Stream complete" in element.value for element in app.success)


@pytest.mark.parametrize("detector", ["SeisCNN", "SpecUNet"])
def test_tampered_corrected_lunar_checkpoint_is_refused(app_env, detector):
    filename, _ = LUNAR_FILES[detector]
    (app_env.model_dir / filename).write_bytes(b"replaced checkpoint")
    app = select_analysis(app_env.run(), detector, "lunar")
    click(app, "Analyze trace")
    assert_about_available(app)
    assert any("hash does not match" in error.value for error in app.error)
    app_env.spec.assert_not_called()
    app_env.cnn.assert_not_called()

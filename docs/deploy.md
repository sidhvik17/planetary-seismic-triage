# Deploying the demo (free tier)

The app and its PyTorch checkpoints ship in `models/`. The lunar demo loads
the corrected seed-42 pair (`lunar_grouped_v1_seed42.pt`, 0.49 MB, and
`unet_lunar_grouped_v1_seed42.pt`, 7.79 MB) and verifies them against
`lunar_grouped_v1_seed42.json`; Mars uses `mars_best.pt` and
`unet_mars_ext_best.pt`. The historical lunar files stay for reproduction. The bundled
traces in `demo_data/` support analysis and the triage simulation, so
deployment is just pointing a free host at it. Needs your account — one-time,
~10 minutes.

## Option A — Hugging Face Spaces (recommended)

1. Create a free account at huggingface.co, then **New Space** → SDK:
   **Streamlit**, hardware: free CPU.
2. Push this repo to the Space (or upload files via the web UI). The Space
   needs this YAML at the top of its `README.md`:

   ```yaml
   ---
   title: Planetary Seismic Event Detector
   emoji: 🌒
   sdk: streamlit
   app_file: app/streamlit_app.py
   ---
   ```

3. HF installs `requirements.txt` automatically (CPU torch — fine, inference
   is 11 ms/window). First build takes a few minutes; cold starts after idle
   are expected on the free tier (PRD F-14, accepted).

Files needed by the Space: `app/`, `planetseis/`, `models/`, `demo_data/`,
`.streamlit/` (theme), `requirements.txt`. `data/`, `runs/`, `results/` are
NOT needed. Note: seisbench is only used by offline baseline scripts — you can
delete it from requirements.txt on the Space to slim the build.

## Option B — Streamlit Community Cloud

1. Push this repo to GitHub (public).
2. share.streamlit.io → New app → pick the repo, main branch,
   `app/streamlit_app.py`.
3. Done — same requirements.txt applies.

## Sanity check after deploy (F-16/F-17)

In **Analyze a trace**, select a bundled demo and press **Analyze trace**.
Confirm the waveform and detections render for both detector families. For
SpecUNet, request denoising and submit again to inspect the reconstructed trace.
The default mask duration is 430 s for lunar (corrected seed-42 model) and
240 s for Mars; these gates
are part of the published operating points.

Check that the app opens without scoring a stream, **Start stream** begins
the triage simulation, and a malformed CSV shows a readable error while
**Model & results** remains available. A no-detections result should render
normally; arbitrary noise is not guaranteed to produce zero detections.

Cite lunar performance from the corrected acquisition-grouped benchmark in
`README.md` (five seeds per model). The demo's seed-42 models are one run
each; their single-seed scores are not the benchmark summary. The historical
lunar scores come from a split with two train/test duplicate waveforms.

## Security notes

* Checkpoints load with `torch.load(..., weights_only=True)`, so a replaced
  `.pt` file cannot run pickled code. Keep `torch>=2.6` (CVE-2025-32434).
* Uploads are capped at 200 MB and 12 M analysed samples. A miniSEED whose
  records would expand past 48 M samples once gaps are zero-filled is
  rejected before allocation (a tiny file can otherwise request many GB).
* `scripts/deploy_hf.py` builds a Docker Space that runs as non-root uid 1000,
  and reads `HF_TOKEN` only from the environment. Never commit the token.
* Keep `streamlit>=1.37` (CVE-2024-42474, static-file path traversal on
  Windows). The app enables no static serving and renders no user HTML.

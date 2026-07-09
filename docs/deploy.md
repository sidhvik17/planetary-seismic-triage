# Deploying the demo (free tier)

The app + both checkpoints (0.5 MB each, in `models/`) ship in this repo, so
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

Files needed by the Space: `app/`, `planetseis/`, `models/`, `requirements.txt`.
`data/`, `runs/`, `results/` are NOT needed.

## Option B — Streamlit Community Cloud

1. Push this repo to GitHub (public).
2. share.streamlit.io → New app → pick the repo, main branch,
   `app/streamlit_app.py`.
3. Done — same requirements.txt applies.

## Sanity check after deploy (F-16/F-17)

Upload `data/raw/.../lunar/training/data/S12_GradeA/
xa.s12.00.mhz.1970-12-11HR00_evid00017.mseed` at threshold 0.95+:
expect one detection near t=26571 s, confidence ≈ 0.999 (matches the local
result — identical preprocessing module, so served output must equal
notebook output).

Also verify the "no events" path: upload any CSV of pure noise, expect the
green "No events detected" message, not an error.

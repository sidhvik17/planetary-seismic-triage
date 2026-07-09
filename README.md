# Planetary Seismic Event Detection

Automated detection and localization of planetary seismic events using a
lightweight dual-head 1D CNN, trained on Apollo (lunar) and InSight (Martian)
data, with a cross-body transfer study. B.Tech major project — see the PRD for
full requirements. $0 stack: ObsPy + PyTorch + Streamlit, free data, free hosting.

## Layout

```
planetseis/            core package (shared by training AND the web app)
  config.py            all hyperparameters in one place
  preprocessing.py     load + detrend + bandpass + resample (single source of truth)
  data.py              Space Apps 2024 packet parsing, file-level splits
  windows.py           overlapping windows + labels + shift augmentation
  dataset.py           torch Dataset with on-the-fly augmentation
  model.py             SeisCNN: conv backbone + detection & arrival heads
  baseline.py          STA/LTA classical detector
  detect.py            sliding-window inference + detection merging
  evaluate.py          precision / recall / MAE scorer (tolerance-matched)
  train.py             training loop
scripts/
  build_windows.py     raw packet -> cached npz datasets
  run_eval.py          checkpoint -> metrics json (incl. transfer + baseline)
app/streamlit_app.py   web demo
results/               metrics json per experiment
runs/                  checkpoints + training logs
```

## Reproduce

```powershell
# 1. environment
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# 2. data (~2.15 GB, NASA Space Apps 2024 packet: Apollo 12 + InSight, catalogued)
curl.exe -L -o data\raw\space_apps_2024.zip https://wufs.wustl.edu/SpaceApps/data/space_apps_2024_seismic_detection.zip
Expand-Archive data\raw\space_apps_2024.zip data\raw

# 3. build window caches
.venv\Scripts\python scripts\build_windows.py

# 4. train (per body)
.venv\Scripts\python -m planetseis.train --body lunar
.venv\Scripts\python -m planetseis.train --body mars

# 5. evaluate: same-body, transfer, baseline
.venv\Scripts\python scripts\run_eval.py --model runs\lunar\best.pt --eval-body lunar
.venv\Scripts\python scripts\run_eval.py --model runs\lunar\best.pt --eval-body mars
.venv\Scripts\python scripts\run_eval.py --model runs\mars\best.pt --eval-body mars
.venv\Scripts\python scripts\run_eval.py --model runs\mars\best.pt --eval-body lunar

# ablations + extensions
.venv\Scripts\python -m planetseis.train --body lunar --no-augment --tag noaug
.venv\Scripts\python -m planetseis.train --body lunar --arch tiny --tag tiny   # Pareto sweep
.venv\Scripts\python -m planetseis.train --body lunar --arch large --tag large
.venv\Scripts\python scripts\mine_hard_negatives.py --body lunar               # then retrain
.venv\Scripts\python scripts\seisbench_baseline.py --eval-body lunar           # PhaseNet/EQT zero-shot
.venv\Scripts\python scripts\uncertainty_eval.py                               # MC-Dropout + calibration
.venv\Scripts\python scripts\make_figures.py                                   # report figures

# 6. web app (analysis + on-lander triage simulation)
.venv\Scripts\streamlit run app\streamlit_app.py
```

## Key protocol decisions

| Decision | Value | Why |
|---|---|---|
| Common sampling rate | 6.625 Hz | Apollo LP native; InSight downsampled — one input distribution for cross-body |
| Bandpass | 0.5–3.0 Hz | under common Nyquist; keeps Apollo 0.5–1 Hz energy and Mars 2.4 Hz band |
| Window | 4096 samples (~618 s), 50% overlap | lunar events are long/emergent |
| Normalization | per-window z-score | no global statistics → no train/test leakage |
| Match tolerance | ±60 s | catalog picks are themselves approximate (emergent onsets) |
| Splits | by file, seeded | events never leak across train/val/test |
| Threshold | tuned on val, never test | honest precision/recall |
| Metrics | on continuous traces | balanced window accuracy would be meaningless |

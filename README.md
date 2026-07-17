# Planetary Seismic Event Detection

Two detector families for planetary seismic data, built on a $0 stack
(ObsPy + PyTorch + Streamlit, free data, free hosting):

1. **SeisCNN** — lightweight dual-head 1D CNN (117,842 params), supervised
   on the packet's window labels, with MC-Dropout uncertainty and an
   on-lander downlink-triage demo.
2. **SpecUNet** — MarsQuakeNet-style spectrogram U-Net (Dahmen et al. 2022)
   trained **without any real labeled positives**: real event templates are
   spectrally gated, injected into event-free noise at random SNRs, and the
   exact time-frequency energy-ratio masks supervise a per-pixel event/noise
   segmentation. Detection = frequency-integrated mask energy; the same mask
   denoises the trace (mask × complex STFT → inverse).

B.Tech major project. **Repo:** https://github.com/sidhvik17/planetary-seismic-triage ·
**Report:** [docs/report.md](docs/report.md) · **Demo:** `streamlit run app/streamlit_app.py`

## Headline results (continuous held-out traces, ±120 s tolerance)

| Experiment | P | R | F1 | MAE |
|---|---|---|---|---|
| Lunar→Lunar — **SeisCNN (118K, supervised)** | **0.556** | **0.526** | **0.541** | **40 s** |
| Lunar→Lunar — SpecUNet (1.9M, injection-only) | 0.355 | **0.579** | 0.440 | 68 s |
| Lunar→Lunar — STA/LTA (tuned) | 0.116 | 0.421 | 0.182 | 76 s |
| Lunar→Lunar — PhaseNet 268K, zero-shot | 0.000 | 0.000 | 0.000 | — |
| Lunar→Lunar — EQTransformer 376K, zero-shot | 0.006 | 0.053 | 0.011 | 106 s |
| Mars_ext→Mars_ext — SpecUNet (injection-only) | **1.000** | 0.600 | **0.750** | **25 s** |
| Lunar→Mars / Mars→Lunar transfer (SeisCNN) | — | — | ~0 | — |

Paired bootstrap ΔF1 (SpecUNet − SeisCNN) = −0.098 [−0.345, +0.152]: the
injection-trained model is statistically indistinguishable from the
supervised one **while never seeing a real labeled positive window**
(`results/statistics_unet.json`).

**Mars, on official MQS labels:** the packet's Martian labels stop at 2
files; cross-referencing its unlabeled files against MQS catalog v14 (IRIS
mars-event service) plus fetching top-magnitude MQS events from the open
XB.ELYSE archive grows the set to 34 files / 5 frozen test spans
(`benchmark/mars_ext_splits.json`). The injection-trained SpecUNet scores
P=1.0 (zero false positives across 25 h), R=0.6, MAE 25 s on the frozen
test — n=5 events, so treat as promising, not definitive.

**Catalog extension (the MarsQuakeNet result, reproduced on the Moon):** of
the SpecUNet's 20 benchmark "false positives" on the lunar test split, **9
match events in the full Nakamura Apollo catalog** (13,058 events; ±5 min)
that the benchmark's 76-label Grade-A subset simply omits — a 45% match rate
against a 1.7% chance rate. Survey-mode precision is 0.645
(`scripts/crosscheck_nakamura.py`, `results/nakamura_crosscheck.json`).

**Arrival refinement on denoised waveforms** (Dahmen & Stott, GJI 2024):
onset-picking each detection on the mask-denoised segment cuts Martian
arrival MAE **24.5 s → 18.7 s** at identical F1; on lunar emergent onsets
it does not help (67.7 → 75.4 s) — both stored in the results json, use
`refine_arrivals` for Mars only.

**Denoising quality** (known clean event on injection val samples, median):
at the hardest SNR bin (0.4–1.0×noise) the mask-denoiser reaches CC 0.59 /
SDR +1.6 dB on lunar and CC 0.69 / +2.5 dB on Mars, versus CC 0.42-0.44 /
−6.8 dB for plain bandpass — an ~8-9 dB SDR gain exactly where events are
hardest, converging to parity at high SNR
(`scripts/denoise_metrics.py`, `results/denoise_metrics_*.json`).

**Cross-station generalization (weak labels):** on the packet's
uncatalogued station sets (each file curated around one real event, no
arrival shipped), SpecUNet fires in 95/96 files — S12 Grade-B 98.4%,
S15/S16 (different instruments, never trained on) 100% — where SeisCNN
managed 60–93% (`results/catalog_extension/summary_lunar.json` vs
`results/station_transfer.json`). Injected-mask training learns event
morphology, not station fingerprints.

Cross-body transfer collapses in both directions — reported as a finding.
MC-Dropout σ separates false alarms from true events 5.9×; the human-review
queue costs 2.6 items/day. Capacity beyond ~120K params *lowers* SeisCNN F1
under 45-event label scarcity (see docs/figures/).

## Layout

```
planetseis/            core package (shared by training AND the web app)
  config.py            all hyperparameters in one place
  preprocessing.py     load + detrend + bandpass + resample (single source of truth)
  data.py              Space Apps 2024 packet parsing, file-level splits
  windows.py           overlapping windows + labels + shift augmentation
  dataset.py           torch Dataset with on-the-fly augmentation
  model.py             SeisCNN: conv backbone + detection & arrival heads
  spectral.py          fixed 128x128 STFT grid, normalization, ratio masks
  injection.py         MQNet synthetic-injection engine (templates, noise
                       pool, glitch negatives, on-the-fly torch Dataset)
  unet.py              SpecUNet: spectrogram U-Net, MC-Dropout bottleneck
  detect_spec.py       stitched mask-curve detection + MQNet-style denoising
  baseline.py          STA/LTA classical detector
  detect.py            sliding-window inference + detection merging
  evaluate.py          precision / recall / MAE scorer (tolerance-matched)
  train.py             SeisCNN training loop
scripts/
  build_windows.py     raw packet -> cached npz datasets
  run_eval.py          SeisCNN checkpoint -> metrics json
  train_unet.py        SpecUNet training on injection data
  eval_unet.py         SpecUNet eval: (threshold, duration) tuned on val only
  crosscheck_nakamura.py  benchmark FPs vs full 13,058-event Apollo catalog
  scan_catalog_extension.py  uncatalogued station sets -> candidate events
  fetch_mqs_labels.py  MQS v14 picks (IRIS) -> mars_ext labels
  fetch_insight_context.py / fetch_mqs_train_events.py  InSight waveforms
  statistics_unet.py   paired bootstrap SpecUNet vs SeisCNN
tests/                 unit tests (STFT grid, injection, detection contracts)
app/streamlit_app.py   web demo (both detectors + denoising view)
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

# SpecUNet (MQNet-style): injection training + benchmark eval + catalog checks
.venv\Scripts\python scripts\train_unet.py --body lunar --epochs 30
.venv\Scripts\python scripts\eval_unet.py --model runs\unet_lunar\best.pt --eval-body lunar
.venv\Scripts\python scripts\crosscheck_nakamura.py --model runs\unet_lunar\best.pt --threshold 0.3 --min-dur 600
.venv\Scripts\python scripts\statistics_unet.py --unet-thr 0.3
# Mars extended track (fetches MQS v14 picks + InSight waveforms; network)
.venv\Scripts\python scripts\fetch_mqs_labels.py
.venv\Scripts\python scripts\fetch_insight_context.py
.venv\Scripts\python scripts\fetch_mqs_train_events.py
.venv\Scripts\python scripts\train_unet.py --body mars_ext --epochs 30
.venv\Scripts\python scripts\eval_unet.py --model runs\unet_mars_ext\best.pt --eval-body mars_ext

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

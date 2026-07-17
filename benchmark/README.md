# PlanetSeis-Bench: planetary seismic event detection benchmark

A fixed, reproducible benchmark for event detection on planetary
single-channel seismic data, built on the public NASA Space Apps 2024 packet
(Apollo 12/15/16 PSE + InSight SEIS). Tiny-n is the *point*: this is the
label regime planetary science actually lives in, and every design choice
below exists to make small-sample results honest.

## Task

Given a continuous single-channel trace, emit event arrival times (seconds
from trace start). A prediction is a true positive if within **±120 s** of an
unmatched catalog pick (greedy one-to-one matching). Report precision,
recall, F1, and arrival MAE, **with file-level bootstrap 95% CIs and a
permutation test vs chance** (`scripts/statistics_rigor.py`).

## Data and fixed splits

* Source: https://wufs.wustl.edu/SpaceApps/data/space_apps_2024_seismic_detection.zip
  (2.15 GB; provenance IRIS/NASA PDS). Fetch + cache: see repo README.
* **Lunar (Apollo 12, Grade A):** 75 labeled day-files, split 45/11/19
  train/val/test **by file** — `lunar_splits.json` (seeded, frozen).
* **Martian (InSight):** 2 labeled files, 1 train / 1 test —
  `mars_splits.json`. n=1 test: anecdotal by definition, never report a bare
  percentage from it.
* **Martian extended track (`mars_ext`)** — `mars_ext_splits.json`: the
  packet's 9 unlabeled test files cross-referenced against the official MQS
  catalog v14 (IRIS mars-event web service, P/start picks), plus
  top-magnitude MQS events fetched from the open XB.ELYSE archive
  (EarthScope FDSN) with ~4-5 h of context each. Test = the 5 packet-derived
  spans (frozen; includes S1222a's family sans S1222a itself, which is
  train); train/val = packet train files + fetched events, with duplicate-
  hour packet files collapsed into one split so no span leaks. Rebuild:
  `scripts/fetch_mqs_labels.py`, `scripts/fetch_insight_context.py`,
  `scripts/fetch_mqs_train_events.py`.
* **Weak-label station sets:** S12 Grade B (64 files), S15 (15), S16 (17) —
  uncatalogued but each curated around one event; used for detection-rate
  generalization only.

## Frozen protocol

| Constant | Value |
|---|---|
| Common sampling rate | 6.625 Hz (InSight downsampled) |
| Bandpass | 0.5–3.0 Hz, 4th-order zero-phase Butterworth |
| Window / hop | 8192 samples (~20.6 min) / 4096 |
| Normalization | per-window z-score (no global stats) |
| Match tolerance | ±120 s (lunar picks are minute-quantized) |
| Dead time after detection | 1800 s lunar / 300 s Mars |
| Decision threshold | tuned on val only, never test |
| Metrics | continuous traces only — never balanced window accuracy |
| Strata to report | SNR (split at median 12.2 dB), event type, per-station |

## Baselines (lunar test, this repo's runs)

| Method | Params | P | R | F1 [95% CI] |
|---|---|---|---|---|
| STA/LTA (tuned) | — | 0.116 | 0.421 | 0.182 [0.099, 0.270] |
| PhaseNet (STEAD, zero-shot) | 268K | 0.000 | 0.000 | 0.000 |
| EQTransformer (STEAD, zero-shot) | 376K | 0.006 | 0.053 | 0.011 |
| SeisCNN (this repo, scratch) | 118K | 0.556 | 0.526 | 0.541 [0.324, 0.757] |
| SeisCNN + planetary SSL (n=10 labels) | 118K | — | — | 0.492 ± 0.077 |
| SpecUNet (MQNet-style injection, no real positives) | 1.9M | 0.355 | 0.579 | 0.440 [0.269, 0.612] |

SpecUNet's operating point is (mask threshold, min event duration) tuned on
val; its benchmark FPs are dominated by real events outside the Grade-A
subset — 9/20 match the full Nakamura catalog (±5 min; chance 1.7%), giving
survey-mode precision 0.645. See `results/nakamura_crosscheck.json`.

## mars_ext baselines (frozen 5-span test, MQS v14 picks)

| Method | P | R | F1 | MAE |
|---|---|---|---|---|
| SpecUNet (injection-only, val-tuned thr=0.30/dur=240s) | 1.000 | 0.600 | 0.750 | 25 s |

n=5 test events: report alongside, never instead of, the lunar numbers.

Reproduce any row: `scripts/run_eval.py`, `scripts/seisbench_baseline.py`,
`scripts/sample_efficiency.py`; error bars via `scripts/statistics_rigor.py`.

## Rules for new entries

1. Never touch the test split during development — threshold and all
   hyperparameters tune on val.
2. Report seed variance (≥3 seeds) for anything trained.
3. Report the SNR-stratified recall and the PR curve, not just the operating
   point.
4. Mars single-event results are anecdotal; label them so.

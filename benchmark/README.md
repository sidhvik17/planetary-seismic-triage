# PlanetSeis-Bench: planetary seismic event detection benchmark

A fixed, reproducible benchmark for event detection on planetary
single-channel seismic data, built on the public NASA Space Apps 2024 packet
(Apollo 12/15/16 PSE + InSight SEIS). Tiny-n is the *point*: this is the
label regime planetary science actually lives in, and every design choice
below exists to make small-sample results honest.

**2026-09-15 correction:** the frozen lunar manifest separates filenames but
does not separate all acquisition spans. Two test waveforms are identical to
training waveforms with different event IDs (1972-07-17 and 1974-07-06).
Treat the existing lunar rows as historical, contaminated-split measurements.
They require grouped splitting, unioned picks, and retraining before they can
support independent-test claims. Run `python scripts/audit_splits.py`; see
`results/split_integrity_audit.json` and `docs/PROJECT_REVIEW.md`.

**2026-09-23 corrected lunar track: `lunar_grouped_v1.json`.** Built by
`scripts/build_grouped_lunar.py` (metadata-only policy, fixed before any
corrected training or test scoring):

* Sources = all 76 Grade-A catalog rows. `evid00029`'s waveform is filed
  under HR02, not the catalog's HR00; it is recovered by a unique same-channel
  event-ID match and its pick recomputed from catalog UTC (36,731 s, not the
  catalog's 46,500 s). It was never in a historical split, so it goes to train.
* Group = connected components of (same channel AND overlapping half-open UTC
  interval) OR identical raw samples/rate. Split precedence per group:
  historical test > val > train.
* Identical copies collapse to one trace; picks are unioned per span from
  catalog UTC minus the actual trace start. Partially overlapping spans
  (adjacent-day files, ~2 s overlap) stay separate within their shared split.
* Counts: 38/11/18 groups, 39/11/21 traces, 42/11/23 events. The audit fails
  closed on any cross-split interval overlap, raw or processed duplicate,
  lost or double-counted event, or cache/manifest hash mismatch.
* Manifest SHA256
  `da0d85966c1090b1b088e42ab0f1957fcb73ec4c2c287505a5aa7245facb46cf`; the
  cache is `data/cache/lunar_grouped_v1/` (not in Git; rebuild with the
  script, which refuses to overwrite a non-empty cache).

| lunar_grouped_v1 test (21 spans / 23 events) | Seeds | F1 mean ± SD | P / R (mean) |
|---|---|---|---|
| SeisCNN, from scratch | 5 | **0.531 ± 0.031** | 0.448 / 0.661 |
| SpecUNet, from scratch, unscreened | 5 | 0.408 ± 0.043 | 0.344 / 0.530 |
| Matched filter (val-tuned) | — | 0.200 | 0.429 / 0.130 |
| STA/LTA (val-tuned) | — | 0.168 | 0.111 / 0.348 |

Welch p = 0.0012 (SeisCNN > SpecUNet); acquisition-group bootstrap ΔF1 95 %
CI [0.009, 0.233]. Summary: `results/lunar_grouped_v1_seed_summary.json`.
Rule 3 on this split (SeisCNN, five seeds, each at its validation point):
recall 0.62 ± 0.10 below and 0.70 ± 0.11 above the 14.4 dB median event SNR
proxy (descriptive strata using the test median);
per-seed PR sweeps (descriptive) in `results/pr_curve_lunar_grouped_v1_seed<N>.json`,
summary in `results/lunar_grouped_v1_secondary_summary.json`. STA/LTA's
validation pick (7.0) is also the optimum on an extended 2–50 grid
(`results/sta_lta_extended_lunar_grouped_v1.json`): above 7.0 it detects no
validation event.
Checkpoints qualify only if they record `benchmark_id`, this manifest hash and
random initialization (`scripts/evaluate_grouped.py` enforces it).

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
  (EarthScope FDSN) with ~4-5 h of context each. Test = 17 spans / 30 events:
  the original 5 packet-derived spans plus 12 tuning-blind additions.
  S1222a itself is in train; train/val = packet train files + fetched events, with duplicate-
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

## Historical baselines (filename split with two duplicate test waveforms)

| Method | Params | P | R | F1 [95% CI] |
|---|---|---|---|---|
| STA/LTA (tuned) | — | 0.116 | 0.421 | 0.182 [0.099, 0.270] |
| PhaseNet (STEAD, zero-shot) | 268K | 0.000 | 0.000 | 0.000 |
| EQTransformer (STEAD, zero-shot) | 376K | 0.006 | 0.053 | 0.011 |
| SeisCNN (this repo, scratch) | 118K | 0.556 | 0.526 | 0.541 [0.324, 0.757] |
| SeisCNN + planetary SSL (n=10 labels) | 118K | — | — | 0.492 ± 0.077 |
| Matched filter (45 train templates, val-tuned) | — | 0.333 | 0.158 | 0.214 |
| Matched filter (*oracle*, operating point tuned on test) | — | 0.261 | 0.316 | 0.286 |
| SpecUNet (synthetic masks from catalog-selected templates) | 1.9M | 0.355 | 0.579 | 0.440 [0.269, 0.612] — **max of 3 seeds**; 5-seed mean 0.372 ± 0.076 |

SpecUNet's operating point is (mask threshold, min event duration) tuned on
val. The historical 9/20 FP matches to the Nakamura catalog used ±300 s,
not the benchmark's ±120 s. They cannot establish distinct additional events
or validated survey precision. In the corrected five-seed identity audit,
45/47 pooled catalog matches refer to already labeled Grade-A events; the
other two refer to one additional catalog event. Keep the primary benchmark
precision unchanged. See `results/nakamura_crosscheck.json` for the preserved
historical counts and the current handoff for the corrected audit.

## mars_ext baselines (expanded 17-span test, MQS v14 picks)

| Method | P | R | F1 | MAE |
|---|---|---|---|---|
| SpecUNet (val-tuned thr=0.30/dur=240s, n=30 events) | 1.000 | 0.233 | 0.378 | 34 s |

Test = 17 spans / 30 events (12 spans added post-freeze, tuning-blind). PR curve: results/mars_ext_test_pr_curve.json (peak F1 0.50 at survey point).

Reproduce any row: `scripts/run_eval.py`, `scripts/seisbench_baseline.py`,
`scripts/sample_efficiency.py`; error bars via `scripts/statistics_rigor.py`.

## Rules for new entries

1. Never touch the test split during development — threshold and all
   hyperparameters tune on val.
2. Report seed variance (**≥5 seeds**) for anything trained, and compare
   models on the seed MEAN, not a single checkpoint. Raised from 3 after a
   3-seed reading supported a "statistically indistinguishable" claim that
   5 seeds refuted (Welch p = 0.024); single-checkpoint scores here are
   dominated by seed variance. See `results/seed_level_comparison.json`.
3. Report the SNR-stratified recall and the PR curve, not just the operating
   point.
4. Mars single-event results are anecdotal; label them so.

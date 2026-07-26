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
| Lunar→Lunar — **matched filter** (45 train templates, val-tuned) | 0.333 | 0.158 | 0.214 | 56 s |
| Lunar→Lunar — matched filter, *oracle* test-tuned upper bound | 0.261 | 0.316 | 0.286 | 59 s |
| Lunar→Lunar — STA/LTA (tuned) | 0.116 | 0.421 | 0.182 | 76 s |
| Lunar→Lunar — PhaseNet 268K, zero-shot | 0.000 | 0.000 | 0.000 | — |
| Lunar→Lunar — EQTransformer 376K, zero-shot | 0.006 | 0.053 | 0.011 | 106 s |

**Matched filtering is the baseline that matters** — Nakamura (2003) and
Bulow et al. (2005, 2007) extended the Apollo catalog by waveform
cross-correlation, so it, not a terrestrial picker, is the method this work
must beat. Templates come from the *same* 45 Grade-A train events the
injection engine uses, so the labelled-information budget is identical;
template length and MAD threshold are both selected on val. It reaches
F1 0.214, and even an *oracle* allowed to pick its operating point on the
test set reaches only 0.286 — below every learned detector here
(`results/matched_filter_lunar.json`). The reason is structural: template
matching wins on *repeating* sources, and the Grade-A benchmark events are
heterogeneous. That same property is why it remains the right tool for
deep-moonquake nests, which the archive scan confirms independently
(`results/archive_scan/B3_FINDINGS.md`). PhaseNet/EQTransformer scoring ~0
is a domain mismatch — 100 Hz terrestrial P/S pickers on 6.625 Hz emergent,
scattered signals — and is reported as context, not as a competitive
baseline.
| Mars_ext→Mars_ext — SpecUNet (injection-only, n=30 events) | **1.000** | 0.233 | 0.378 | 34 s |
| Lunar→Mars / Mars→Lunar transfer (SeisCNN) | — | — | ~0 | — |

**Seed-level comparison (5 seeds SpecUNet, 3 seeds SeisCNN) — the supervised
detector is significantly better.** Averaged over seeds rather than read off
one checkpoint:

| detector | seeds | mean F1 | sd |
|---|---|---|---|
| SeisCNN (supervised) | 3 | **0.498** | 0.043 |
| SpecUNet (injection-only, screened) | 5 | 0.372 | 0.076 |
| SpecUNet (injection-only, unscreened) | 3 | 0.379 | 0.086 |

Welch t-test, SeisCNN vs SpecUNet: **p = 0.024**. The earlier "statistically
indistinguishable" claim does **not** survive proper seed accounting — it came
from a single-seed paired bootstrap that resampled test files while holding
the trained model fixed, so it never saw training stochasticity, which for
injection training (fresh synthetic events every epoch) is a first-order
effect. The defensible claim is that **injection-only training reaches 74.7%
of supervised mean F1 while using zero labeled positive windows**. That is
not parity and is not described as such
(`results/seed_level_comparison.json`).

Note also that the frozen v1.0 SpecUNet figure, F1 0.440, is the **maximum**
of its three seeds; the seed mean is 0.379. Single-checkpoint numbers in this
README should be read against the seed distribution.

The two detectors are complementary operating regimes, not competitors:
SeisCNN learned the Grade-A catalog's selection function (benchmark
precision), SpecUNet learned event morphology (recall + survey mode).
Hard-negative fine-tuning with the negative pool screened against the full
Nakamura catalog (23 of 64 mined "negatives" were real moonquakes and were
excluded) preserves recall but does not raise benchmark precision —
consistent with the remaining "false positives" being dominated by real
uncatalogued events rather than learnable noise
(`results/unet_lunar_ft2_to_lunar.json`).

**Contaminated negatives (positive-unlabeled ablation).** The injection
engine harvests "event-free" noise by guarding the 76 Grade-A picks — but
those are a strict *subset* of Nakamura (all 75 checked picks fall within
±60 s of a catalog entry), so the ~5,300 remaining S12 events are harvested
as noise and trained against a zero mask. Measured contamination: **8.38%
of eligible noise windows (2,009,601 of 23,977,989)** sit inside the guard
radius of a real catalogued event — textbook case-control contamination.
Retraining with those windows screened out (`--screen-nakamura`) and
comparing **across 5 seeds vs 3**: mean F1 0.372 (screened) vs 0.379
(unscreened), Welch **p = 0.923**. **Training-negative contamination has no
measurable effect on benchmark F1.** The single-seed 0.440 → 0.407 drop seen
before the seed experiment was noise, not an effect — a useful reminder of
how misleading one checkpoint is here.

At a fixed seed the screened model did leave Grade-A recall identical (11 TP)
while raising real catalogued events found from 20 to 22 — the metric
penalizing a model for finding events its label subset omits — but that too
is a 2-event difference on 19 files and is not claimed as significant
(`results/ablation_noise_screen.json`, `results/seed_level_comparison.json`).
The frozen headline therefore keeps the unscreened checkpoint: with p = 0.92
there is no basis for re-freezing.

**Mars, on official MQS labels:** the packet's Martian labels stop at 2
files; cross-referencing its unlabeled files against MQS catalog v14 (IRIS
mars-event service) plus fetching MQS events from the open XB.ELYSE archive
grows the set to 46 files / **17 frozen test spans, 30 test events**
(`benchmark/mars_ext_splits.json` — the expansion was added to test only,
never used for tuning). At the val-tuned operating point the
injection-trained SpecUNet holds **precision 1.000 — zero false positives
across ~85 hours** — with recall 0.233 against a test set dominated by
M2.9–3.0 events (MAE 34 s; the initial n=5 evaluation had read R=0.6, a
small-sample artifact now corrected). The full test PR curve peaks at
F1 0.50 (P 0.667 / R 0.40) at the survey point
(`results/mars_ext_test_pr_curve.json`). Miss autopsy
(`results/`): S0173a fires above threshold but fails the duration gate
(operating-point boundary); S1022a sits in amplitude-degraded data — and a
5000-count glitch in the same span stays *below* threshold, the synthetic
glitch training holding up.

**Catalog extension (the MarsQuakeNet result, reproduced on the Moon):** of
the SpecUNet's 20 benchmark "false positives" on the lunar test split, **9
match events in the full Nakamura Apollo catalog** (13,058 events) that the
benchmark's 76-label Grade-A subset simply omits — 45% match rate vs 1.7%
chance, permutation test p < 10⁻⁴ (0 of 10,000 random placements reach 9).
Match tolerance ±5 min, set by **detector** arrival error, not catalog
coarseness: SpecUNet's true-positive arrivals carry MAE 68 s / median 77 s,
so genuine matches routinely land 2–4 min from a catalog time. (The catalog
itself is tighter than it looks — all 75 checked Grade-A picks fall within
±60 s of their Nakamura entry — which is why the earlier "Nakamura times are
too coarse" justification was wrong.) Sensitivity is reported in full: 0
matches at ±60 s, 1 at ±120 s, 5 at ±180 s, 9 at ±300 s. The permutation test
is evaluated at the same tolerance, so the p-value is unaffected by the
choice. Survey-mode precision 0.645. Side-by-side waveform
evidence in `docs/figures/nakamura_match_*.png` — match #1 is an
unambiguous meteoroid impact with an hour of coda, 160 s from its catalog
entry (`scripts/crosscheck_nakamura.py`,
`results/nakamura_crosscheck.json`).

**Denoise-then-detect chaining** (`scripts/eval_chain.py`,
`scripts/build_denoised_windows.py`): feeding SpecUNet-denoised traces to
the raw-trained SeisCNN collapses precision (0.556 → 0.067 — distribution
shift); retraining SeisCNN on denoised windows restores it (F1 0.512,
recall 0.526 → 0.579, one false negative recovered); probability fusion of
raw+denoised streams reaches the **highest recall of any configuration,
0.684**, at survey precision — the denoiser demonstrably surfaces the
low-SNR events the supervised detector misses, and the recall/precision
trade is reported rather than hidden.

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
MC-Dropout σ separates false alarms from true events 5.9× **for the
supervised SeisCNN**; the human-review queue costs 2.6 items/day. Ported to
the injection-trained SpecUNet the separation *inverts* (FP σ / TP σ =
0.55) and curve-height confidence is not calibrated as a probability — the
mid-confidence bin is 100% real events (incl. Nakamura) while the top bin
holds the artifacts (`results/uncertainty_unet.json`). A model never taught
the catalog's selection function cannot rank catalog membership by
epistemic uncertainty: triage remains SeisCNN's role; SpecUNet is the
survey and denoising instrument. Capacity beyond ~120K params *lowers*
SeisCNN F1 under 45-event label scarcity (see docs/figures/).

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

# one-command headline-table reproduction (frozen checkpoints, no tuning)
.venv\Scripts\python scripts\reproduce_headline.py

# 6. web app (analysis + on-lander triage simulation)
.venv\Scripts\streamlit run app\streamlit_app.py
```

## Key protocol decisions

| Decision | Value | Why |
|---|---|---|
| Common sampling rate | 6.625 Hz | Apollo LP native; InSight downsampled — one input distribution for cross-body |
| Bandpass | 0.5–3.0 Hz | under common Nyquist; keeps Apollo 0.5–1 Hz energy and Mars 2.4 Hz band |
| Window | 8192 samples (~1236 s), hop 4096 (50% overlap) | lunar events are long/emergent |
| Normalization | per-window z-score | no global statistics → no train/test leakage |
| Match tolerance | ±120 s | lunar picks are minute-quantized and onsets emergent, so tolerance must exceed the 60 s label resolution |
| Splits | by file, seeded | events never leak across train/val/test |
| Threshold | tuned on val, never test | honest precision/recall |
| Metrics | on continuous traces | balanced window accuracy would be meaningless |

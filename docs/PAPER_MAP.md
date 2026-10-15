# Paper map: where every number in the manuscript comes from

Manuscript: *Injection-Trained Planetary Seismic Event Detection Under Label
Scarcity: A Leakage-Corrected Lunar Benchmark and Its Limits* (Springer Nature
proceedings, SVProc class), as submitted. Its LaTeX source lives in a separate
private repository. This page maps each table, figure and number in it to the
record in `results/` (or `benchmark/`) that holds it and the script that wrote
that record, and lists the places where the submitted text and the code
differ. The companion review paper quotes a subset of these numbers.

Paths are relative to the repository root. "seed summary" is
`results/lunar_grouped_v1_seed_summary.json`; "secondary summary" is
`results/lunar_grouped_v1_secondary_summary.json`.

## Where the submitted text and the code differ

A read-only audit (2026-10-15) checked every number and method statement of
the submitted manuscript against this repository. Every headline result
matches its record. The differences below are in method descriptions, two
derived numbers and wording; none changes a conclusion of the paper. A
corrected text is kept for a camera-ready version.

| Where | Submitted text | Code and records |
|---|---|---|
| Abstract; Sect. 6 | false-positive coincidences are "late detections" of labelled events | 45 labelled events detected 120–300 s from the pick: 31 after it, 14 before (signed `detection_utc` − `nearest_catalog_utc` in the tolerance audit; the stored offsets are absolute). Sect. 5.2's "120–300 s from the pick" is exact |
| Sect. 1–2 | MarsQuakeNet trains on "real event templates" injected into noise | its abstract (doi 10.1029/2022JE007503) says synthetic event waveforms combined with recorded noise; APSIS injects real templates |
| Sect. 3.1 | 16.2 GB of day files | 16.2 GiB = 17.4 GB (`results/archive_scan/archive_inventory.json` → `totals.npz_bytes`) |
| Sect. 3.2 | historical files grouped "by UTC interval and channel" | `scripts/audit_splits.py` groups by channel and start hour in the filename. The corrected split's interval rule also links two train/test day-boundary neighbours, evid00003/00004 and evid00023/00024 (1.5 s and 3.5 s overlap), which the corrected split already keeps together |
| Sect. 3.3 | manifest hash recorded in every checkpoint and evaluation | true for every checkpoint and model evaluation; the template-matching record stores only `benchmark_id` |
| Sect. 4.1 | a window is positive if it contains a catalog pick | the pick must lie in the window's central 90 % (`edge_margin_frac` 0.05) |
| Sect. 4.2 | each training sample adds one template | 60 % of samples get one template, 10 % two, 30 % stay pure noise with a zero mask (`InjectionDataset` defaults) |
| Sect. 5.1 | EQTransformer F1 0.011 | 1 TP, 171 FP, 18 FN: 2/191 = 0.0105 → 0.010; the record stores 0.0105, which was rounded a second time |
| Sect. 5.1 | seed-level testing reversed the bootstrap ($p=0.024$ historically) | 0.024 compares SeisCNN (3 seeds) with the five *screened* SpecUNet seeds; against the three unscreened seeds, the family the bootstrap tested, Welch p = 0.12. The corrected split (p = 0.0012, five seeds each) is the clear result |
| Sect. 5.3 | screening removed 1,614,243 window starts | that count is the training pool only; the validation pool is screened as well |
| Sect. 5.5 | 7,850 valid station-days | 7,850.44 also counts each day file's 30-minute context pads, so every day boundary twice; inside each file's own day the archive holds 7,536.43 (the scan's 6,848 for S12, S15 and S16 already uses that rule). `archive_inventory.json` records both |
| Sect. 5.5 | S14 lacks its vertical component for about 4.5 years | 1,710 days, 1972-03-12 to 1976-11-17 = 4.7 years |
| Sect. 5.5 | "long dropouts" are marked invalid | runs longer than 10 s (`MAX_INTERP_SEC`); shorter ones are interpolated and stay valid |
| Sect. 5.6 | SpecUNet uncertainty on "all benchmark false positives" | false and true positives come from the 10-pass mean-curve detections at each seed's locked point (`scripts/uncertainty_unet.py`), which differ slightly from the deterministic benchmark detections |
| Sect. 5.7 | (not stated) | the Mars SpecUNet is one seed (42) with 55 training templates |

The review paper's APSIS section shares the "late", "p = 0.024" and
"7,850" points above, and in addition: its Table 1 row applies "42 events,
five seeds" to Mars as well (Mars: 55 templates, one seed); its Fig. 2 draws
the archive through the split and training and says the catalog is used only
at evaluation time (the screened-noise ablation uses it during training); it
says gaps and flags are marked invalid (only runs over 10 s are).

## Data and benchmark (Sect. 3)

| Claim | Value in the records | Record → key | Written by |
|---|---|---|---|
| Historical split | 75 files, 45/11/19; 70 acquisitions, 5 duplicate groups, 2 cross train/test (evid00067/68, evid00150/51) | `results/split_integrity_audit.json` | `scripts/audit_splits.py` |
| Corrected split | 38/11/18 groups, 39/11/21 traces, 42/11/23 events; 76 Grade-A recordings | `benchmark/lunar_grouped_v1.json` → `audit` | `scripts/build_grouped_lunar.py` (`planetseis/grouped_data.py`) |
| Day-boundary neighbours linked by the corrected rule | evid00003/00004, evid00023/00024 (historically split across train/test) | `benchmark/lunar_grouped_v1.json` → `groups`, `grouping_links` | same |
| Manifest hash in checkpoints and model evaluations | `da0d8596…46cf` | every `results/lunar_grouped_v1_*.json` → `data_manifest_sha256` | `planetseis/training_state.py`, `scripts/evaluate_grouped.py` |
| Continuous archive | 7,777 day files; 7,536.43 valid station-days inside each day (7,850.44 with the pads); 17.36 GB | `results/archive_scan/archive_inventory.json` → `totals` | `scripts/archive_inventory.py` |
| S14 vertical gap | 1,710 days (4.68 years), 1972-03-12 to 1976-11-17 | same → `stations.S14.longest_gap` | same |
| Mars test | 17 spans, 30 events | `benchmark/mars_ext_splits.json`; `results/unet_mars_ext_to_mars_ext.json` | `scripts/fetch_mqs_labels.py`, `scripts/eval_unet.py` |

## Methods (Sect. 4)

Every constant is in code, not in a result file:

| Claim | Where |
|---|---|
| Band-pass 0.5–3.0 Hz, 6.625 Hz, windows 8,192 / hop 4,096, ±120 s, 1,800 s dead time | `planetseis/config.py`, `planetseis/preprocessing.py`, `planetseis/detect.py` |
| Window positive when a pick lies in its central 90 % | `planetseis/windows.py`, `edge_margin_frac` 0.05 in `planetseis/config.py` |
| SeisCNN 117,842 parameters; 60 epochs, AdamW 1e-3, batch 64, 4 negatives per positive | `planetseis/model.py`, `planetseis/train.py`, `scripts/run_grouped_seeds.sh` |
| SpecUNet 1,942,433 parameters; 128 × 128 STFT grid | `planetseis/unet.py`, `planetseis/spectral.py` |
| Injection mixture: 60 % one template, 10 % two, 30 % pure noise; SNR log-uniform 0.4–12; glitches with p = 0.4 | `planetseis/injection.py` (`InjectionDataset` defaults) |
| Ratio mask with noise floor 0.05 × median in-band noise, ε = 1e-20 | `planetseis/spectral.py` |
| 30 epochs × 6,400 samples, batch 32, patience 8 | `scripts/run_grouped_seeds.sh`, `scripts/train_unet.py` |
| Validation grids, tie rules, exclusive hashed selection files | `scripts/evaluate_grouped.py` |
| Welch test, 10,000-replicate bootstrap over 18 test groups | `scripts/aggregate_grouped_seeds.py` |

## Results (Sect. 5)

| Claim | Value in the records | Record → key | Written by |
|---|---|---|---|
| Table 1, SeisCNN / SpecUNet | 0.531 ± 0.031 / 0.408 ± 0.043; P 0.448 / 0.344; R 0.661 / 0.530 | seed summary → `detectors` | `scripts/aggregate_grouped_seeds.py` |
| Welch test, ratio | t = 5.15, p = 0.0012, ΔF1 0.123, 76.8 % | seed summary → `tests` | same |
| Group bootstrap | [0.437, 0.619], [0.319, 0.493], [0.009, 0.233] | seed summary → `bootstrap` | same |
| Table 1, screened SpecUNet | 0.286 ± 0.133; P 0.403, R 0.287; t = −1.95, p = 0.11 | `results/lunar_grouped_v1_screening_ablation.json` | `scripts/run_screened_grouped.sh`, `scripts/compare_screened_grouped.py` |
| Template matching, oracle | 0.200 (P 0.429, R 0.130); oracle 0.276 | `results/matched_filter_lunar_grouped_v1.json` | `scripts/matched_filter_baseline.py` |
| STA/LTA, extended grid | 0.168 (P 0.111, R 0.348); 7.0 selected on 2–50 | seed summary → `baselines`; `results/sta_lta_extended_lunar_grouped_v1.json` | `scripts/sta_lta_grouped.py` |
| Zero-shot pickers (historical split) | PhaseNet 0.000 (310 FP); EQTransformer 1 TP, 171 FP, 18 FN (F1 0.0105) | `results/seisbench_zeroshot_lunar.json` | `scripts/seisbench_baseline.py` |
| Historical seeds | SeisCNN 0.498 (3); SpecUNet 0.379 unscreened (3), 0.372 screened (5); 0.440 = best of the 3 unscreened | `results/seed_level_comparison.json` → `detectors` | — (assembled from the per-seed results) |
| Historical seed tests | p = 0.024 vs screened (stored); p = 0.12 vs unscreened (Welch on the stored `per_seed_f1`) | same → `tests`, `detectors` | — |
| Catalog coincidences | 43 % ± 13 % (47 of 120), chance 1.5 %, plus-one p ≈ 1e-4 | secondary summary → `nakamura` | `scripts/run_grouped_secondary.py`, `scripts/aggregate_grouped_secondary.py` |
| Identity audit | 45 of 47 are Grade-A events 120–300 s from the pick (31 after, 14 before); 2 detections → 1 event (1975-06-26) | `results/lunar_grouped_v1_catalog_tolerance_audit.json` | `scripts/analyze_catalog_tolerance.py` |
| Historical noise contamination | 8.38 % (2,009,601 of 23,977,989) | `results/ablation_noise_screen.json` → `setup` | `scripts/train_unet.py --screen-nakamura` |
| Corrected screen | 142 event times in 50 spans; 1,614,243 training-pool window starts removed | screen SHA256 in the ablation record → `nakamura_screen_sha256`; counts printed by `scripts/train_unet.py` into the (git-ignored) training log | `scripts/build_nakamura_screen.py --data-dir` |
| Denoising SDR (historical checkpoint) | 1.6 vs −6.8 dB (SNR 0.4–1.0); 15.6 vs 15.7 dB (6–12) | `results/denoise_metrics_lunar.json` → `by_snr` | `scripts/denoise_metrics.py` |
| Table 2, denoise-then-detect | raw 0.561, denoised 0.222, fusion 0.200; 16 TP / 18 FP / 7 FN raw | `results/chain_lunar_grouped_v1_seed42.json` (+ `.selection.json`) | `scripts/chain_grouped.py` |
| Archive scan, locked point (0.25, 430 s) | recall 1.4 / 0.5 / 0.3 %; 0.05 / 0.04 / 0.04 unmatched per station-day; 6,848 station-days | `results/archive_scan/grouped_v1_seed42/nakamura_score.json`, `scan_summary.json` | `scripts/run_archive_grouped.sh` |
| Archive sweep | S12 0.557 at 18.8/day; 0.060 at ≤ 0.2/day; deep 0.56 / 0.36 / 0.29; shallow 38 of 61 | `results/archive_scan/grouped_v1_seed42/sweep.json` | `scripts/sweep_archive.py` |
| Historical archive scan | S12 0.525 at 15.4/day; 0.071 at 0.19/day | `results/archive_scan/screened/sweep.json` | same |
| SeisCNN MC dropout (20 passes) | 7.9 ± 3.5 (3.7–12.9); queue 29–42; a real event in the queue for one seed | secondary summary → `seiscnn_uncertainty`; `results/uncertainty_lunar_grouped_v1_seed*.json` | `scripts/uncertainty_eval.py` |
| SpecUNet MC dropout (10 passes) | all-FP / TP 1.06 ± 0.38 (0.68–1.66); FP and TP from the 10-pass mean-curve detections at each seed's locked point | secondary summary → `specunet_uncertainty.fp_over_tp` | `scripts/uncertainty_unet.py` |
| Signal-strength strata | median 14.4 dB; recall 0.62 ± 0.10 / 0.70 ± 0.11 | secondary summary → `seiscnn_snr_recall` | `scripts/extra_analysis.py` |
| Mars | TP 7, FP 0; P 1.000, R 0.233, F1 0.378 at (0.30, 240 s); MAE 33.9 → 34.2 s; one seed, 55 training templates | `results/unet_mars_ext_to_mars_ext.json` | `scripts/eval_unet.py` |

Do not cite `sigma_separation_fp_over_tp` from the per-seed
`uncertainty_unet_*.json` files: despite its name it holds catalog-unmatched
FP over TP. The all-FP ratio is the secondary summary's `fp_over_tp`.

## Figures

`scripts/make_springer_figures.py` redraws Fig. 1–3 from the records above into
`docs/figures/paper/` (git-ignored, like every `paper/` path); the output is
pixel-identical to the figures printed in the manuscript.
`scripts/make_paper_figures.py` is older: it draws figures for the July
workshop draft from the historical checkpoints.

## Not held in a result file

- The −1 missing-sample flag measurement (centreline near 495 counts, signal
  range about 15 counts, 33× spike, standard deviation inflated 7.7-fold) was
  made on the raw 1973-03-01 S12 MHZ record during ingest and is documented in
  the `planetseis/archive.py` docstring. The cache holds cleaned traces, so it
  cannot be recomputed from the cache.
- The Nakamura catalog file `data/raw/levent.1008.dat` (2008 revision) is not
  redistributed here; the scripts expect it at that path.

## Caveats for anyone re-verifying hashes

- Hashes were recorded on Windows. The published manifest hash `da0d8596…`
  is the hash of the CRLF-encoded `benchmark/lunar_grouped_v1.json`; git
  stores it with LF line endings and Git for Windows restores CRLF on
  checkout. Result and selection files were written with LF.
- A cache rebuilt on another platform can differ in bytes (line endings,
  floating point), so `scripts/evaluate_grouped.py` may reject the shipped
  checkpoints there. Compare metrics, not hashes, in that case.
- Some historical records keep interpretive text that later audits withdrew
  (`seed_level_comparison.json` → `revised_claim`; `survey_precision` in
  `nakamura_crosscheck.json` and `ablation_noise_screen.json`). The values
  stay as records; the README and this page give the current reading.

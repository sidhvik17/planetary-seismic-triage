# APSIS — accumulated changes and research notes

**Updated:** 2026-09-23  
**Comparison:** current working tree against Git HEAD
`ecf07efe6c95eb026870cede5cb06dcef3769d25` (`docs: name the project APSIS`)  
**Branch:** `archive-scan-and-seed-variance`

**Latest status (2026-09-23):** the corrected benchmark and five-seed training
are complete; see sections 7–8. Sections 1–6 preserve the earlier audit and
must be read as history. The current continuation record is
[`tasks/RESEARCH_HANDOFF.md`](../tasks/RESEARCH_HANDOFF.md).

The manuscript has its own nested Git repository at `paper/`, on branch
`main`. Its comparison baseline is
`2e219841c05f5785c12613e2c6f46851277c7613` (`ML4PS 2026 workshop paper:
corrected claims, matched-filter baseline, archive negative`). The two
repositories are reviewed separately below; the paper is already tracked
in its own repository.

## 1. Summary

This report covers the earlier improvement pass and the resumed verification
and manuscript work. APSIS retains its purpose: planetary seismic detection,
arrival estimation, denoising and on-lander triage simulation. Python, ObsPy,
PyTorch, Streamlit, NumPy/SciPy and Plotly remain the stack. SeisCNN and
SpecUNet were already present at the Git baseline; they were not added by
this pass.

The improvements make the software easier to run and errors easier to
understand. The audit also identified a scientific problem: **two of nineteen
lunar test waveforms occur identically in training**. Software improvements
and exact score reproduction do not repair that problem.

## 2. Changes since the previous Git version

| Area / files | Before | After | Why it matters |
|---|---|---|---|
| `app/streamlit_app.py` — execution | Upload/control changes triggered inference; hidden triage and collapsed denoising could compute eagerly | Explicit Analyze / Start actions, optional denoising, results stored with submitted settings | Avoids unnecessary work and mislabeling retained results |
| App — demo access | Analyze required a user-provided raw file | Bundled Moon/Mars NPZ examples available; already-preprocessed arrays are reused directly | Demo works without downloading the full packet or filtering twice |
| App — operating settings | SpecUNet silently used a 30 s duration gate; CNN default differed from recorded evaluation | Mask threshold 0.30, lunar/Mars duration 600/240 s; CNN threshold 0.99/0.50 | Default deterministic settings now match the historical experiments |
| App — failure isolation | `st.stop()` or absent models could prevent unrelated views from rendering | Local errors for missing models/data or invalid uploads; About stays available | One unavailable resource does not disable the whole demonstration |
| App — memory | Large default inference batches | CNN batches 64; SpecUNet batches 8, including optional denoising | Bounds memory use on the CPU without changing architectures |
| App — meaning of scores | Generic confidence and shared uncertainty interpretation | Mask-energy scores identified as uncalibrated; MC/changed parameters exploratory | Helps prevent misuse of model scores in the paper or demo |
| App — triage metric | Broad downlink-reduction wording | Fraction of overlapping windows not queued; not measured bytes | Keeps a simulation measure distinct from radio savings |
| `planetseis/preprocessing.py` | Invalid CSV timing could fall back to a guessed rate; non-finite values and multiple channels could be mishandled | Validates amplitude shape/finiteness, timing monotonicity/spacing, rates, band limits, and single-channel input | Invalid data cannot silently become a plausible no-event result |
| CSV loading | Headerless single-column files lost their first value | Preserves the first sample; documents 6.625 Hz assumption for sample-only CSV | Fixes a concrete input loss and makes rate assumptions visible |
| `scripts/audit_splits.py` | No repeatable check for duplicate acquisition filenames across splits | Candidate grouping plus optional exact array/rate and SHA256 confirmation; nonzero on overlap | Produces inspectable evidence for the scientific correction |
| `planetseis/data.py` | Split docstring asserted no leakage | Docstring identifies the historical filename splitter and known overlap | Removes a false guarantee without silently changing the frozen protocol |
| Tests and `.github/workflows/ci.yml` | 44 tests, no app/input/audit regressions | 115 tests; CI installs existing Streamlit/Plotly dependencies for app tests | Covers the new behavior and keeps changes reviewable |
| README, report, benchmark, walkthrough, deploy guide | Conflicting supervision/seed/Mars claims and outdated demo instructions | Explicit historical-score limitations and corrected claims/settings | Reduces contradictions between implementation, demonstration and paper |
| Separate `paper/` repository | Existing manuscript tracked separately; project-root Git intentionally ignores the nested repository | Revised `ml4ps_2026.md`, added `REVISION_NOTES.md`, ignored the local pre-revision backup in `paper/.gitignore` | Preserves the existing repository layout and makes the manuscript diff reviewable with `git -C paper diff` |

New regression files: `tests/test_preprocessing.py` (49 cases),
`tests/test_app.py` (15 cases), and `tests/test_split_audit.py` (7 cases).
New audit evidence: `results/split_integrity_audit.json`.

## 3. Research corrections and how to write them

### 3.1 The label budget

Use: **“SpecUNet learns from synthetic mask targets constructed by injecting
catalog-derived event templates into candidate noise.”**

The training does not directly supervise real positive windows, but catalog
picks select the templates. Do not describe this as zero-label learning.
The source is `planetseis/injection.py`, especially `build_template_bank`.

### 3.2 Split integrity

The frozen lunar manifest contains 75 filenames and 70 acquisition groups.
Five groups repeat; two cross train/test:

| Date / S12 MHZ acquisition | Train ID | Test ID | Samples | Confirmation |
|---|---|---|---|---|
| 1972-07-17 00:00 | evid00068 | evid00067 | 572,399 | Equal arrays/rates and matching SHA256 |
| 1974-07-06 00:00 | evid00150 | evid00151 | 572,411 | Equal arrays/rates and matching SHA256 |

Use: **“We reproduced the historical scores, but an acquisition audit found
two test waveforms duplicated in training. Independent-test claims require
grouped splitting and retraining.”**

Do not merely drop those files and label the remaining score a corrected
benchmark. A valid correction needs a versioned split, merged picks, training
on that split, and fresh validation-only operating-point selection.

### 3.3 Seed comparison

`results/seed_level_comparison.json` records:

| Detector | Seeds | Mean F1 ± sample SD |
|---|---:|---:|
| SeisCNN | 3 | 0.4983 ± 0.0430 |
| SpecUNet, screened | 5 | 0.3724 ± 0.0761 |
| SpecUNet, unscreened | 3 | 0.3786 ± 0.0857 |

The stored supervised-versus-screened Welch p-value is 0.0244. The earlier
parity claim is unsupported; the ratio 74.7% describes these stored means and
inherits the split limitation. Screening p = 0.9231 means no difference was
detected, not that equivalence was proved. The frozen SpecUNet F1 0.4400 is
the maximum of its three unscreened seeds.

### 3.4 Mars and uncertainty

The expanded Mars artifact has 17 spans and 30 events: TP 7, FP 0, FN 23,
precision 1.0000, recall 0.2333, F1 0.3784. Arrival refinement changes MAE
**33.93 → 34.20 s** at identical detection counts; the older 24.5 → 18.7 s
statement does not describe the expanded test. Source:
`results/unet_mars_ext_to_mars_ext.json`.

SpecUNet's stored uncertainty ratios use distinct denominators: 0.55 for
FP/TP median uncertainty, 0.49 after including Nakamura matches among real
events and excluding them from false positives. Do not interchange these
labels. Mask-energy scores are not calibrated event probabilities.

### 3.5 Model/input and inference limits

The revised manuscript specifies the actual normalization and target formula:
SeisCNN waveform z-scores; SpecUNet median/IQR scaling of real/imaginary STFT
components; a magnitude-ratio mask with the implemented noise floor.
The ±300 s catalog-match permutation result is specific to that tolerance;
its p-value is not invariant to tolerance changes.

Known inference boundary behavior remains: CNN can leave a trailing partial
hop unscanned, SpecUNet requires 8192 processed samples, and the denoiser
does not reconstruct its final 64 samples. The app now clearly rejects short
SpecUNet inputs. Algorithm changes need separately validated results.

## 4. Fresh verification on 2026-09-22

| Check | Result | What it establishes |
|---|---|---|
| `python -m pytest tests -q` | **115 passed in 71.02 s** | Covered software contracts, including mocked app interactions |
| `python -m compileall -q planetseis app scripts` | Passed | Python source compilation, not end-to-end execution of every script |
| `python scripts/reproduce_headline.py` | Passed, exit 0 | Numeric reproduction of two historical lunar rows |
| `git diff --check` | Passed | No patch whitespace errors under repository Git settings |
| Split audit with local cache | 2 confirmed cross-split duplicate pairs | Known lunar split contamination remains |

Freshly reproduced historical values:

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| SeisCNN | 0.5556 | 0.5263 | 0.5405 |
| SpecUNet | 0.3548 | 0.5789 | 0.4400 |

The September 22 browser walkthrough used the actual shipped checkpoints:

| Live workflow | Observed output |
|---|---|
| Lunar SpecUNet, `evid00003`, threshold 0.30 / duration 600 s | One candidate at 12,858 s, mask score 0.757; catalog pick 12,720 s. This is outside the benchmark's ±120 s match tolerance. Input and denoised charts rendered. |
| Mars SpecUNet, `evid0006`, threshold 0.30 / duration 240 s | One candidate at 2,115.6 s, mask score 0.819; catalog pick 2,130 s. Input and denoised charts rendered. |
| Mars SeisCNN, `evid0006`, threshold 0.50 | Candidates at 809.3 s and 2,170 s; the second is near the 2,130 s pick and the first is an extra candidate relative to that pick. |
| Lunar SeisCNN triage, `evid00003`, 8× playback | Completed: 138 windows, one auto-accepted event, three review candidates, 96.4% of windows not queued. MC-Dropout outputs can vary on rerun. |
| Model & results tab | Both architectures, recorded metrics, operating settings and split/uncertainty caveats rendered. |

The triage percentage counts overlapping windows rather than transmitted
bytes. The September 15 checks additionally exercised both detector/body
combinations, CSV input, denoising, result persistence and Mars triage through
Streamlit AppTest. The app is available locally at `http://127.0.0.1:8501/`;
the runtime review is recorded in `tasks/todo.md`.

Not established by these checks: corrected independent lunar performance,
fresh training/optimizer/resume behavior, every offline ablation, all external
data downloads/services, cloud hosting or flight deployment. No new training
or corrected score is claimed.

## 5. Research-paper files and Git handling

The revised working manuscript is **`paper/ml4ps_2026.md`**, with
`paper/REVISION_NOTES.md` explaining source/export status. Its abstract,
method, split description, historical-results qualification, limitations,
Mars section, reproducibility section and conclusion were revised.
The original is backed up under `paper/archive_before_2026_09_22/`.

Older Markdown/LaTeX drafts, the Overleaf ZIP and decks have not been
regenerated and must not be assumed to contain these corrections. Existing
external reference metadata was retained rather than independently verified.

All accumulated changes remain local and uncommitted in the two existing
repositories. No branch switch, commit, push, model overwrite, or frozen
split/result replacement was made. Project-root `.gitignore` continues to
ignore the independently versioned `paper/` directory. Inside that repository,
the manuscript is a tracked modification, the revision note is a new file,
and its `.gitignore` excludes the local backup. Older tracked figures and
exports are unchanged. No remote fetch was performed; branch information
describes the local repository and its last-known remote references.

Useful review commands:

```powershell
git status --short
git diff -- app/streamlit_app.py planetseis/preprocessing.py
git diff -- README.md docs/report.md benchmark/README.md
git ls-files --others --exclude-standard
git -C paper status --short
git -C paper diff -- ml4ps_2026.md .gitignore
.venv\Scripts\python -m pytest tests -q
.venv\Scripts\python scripts\audit_splits.py --cache data/cache/lunar/continuous
.venv\Scripts\python scripts\reproduce_headline.py
```

The split-audit command intentionally exits 1 when it finds overlap.

## 6. Next research milestone

Group station/channel acquisition intervals, union event picks, create a
versioned independent benchmark/cache, retrain both existing detector families,
select operating points on validation only, and regenerate the affected
metrics and paper figures. This preserves the project's idea and stack while
addressing the main barrier to a defensible model-performance paper.

## 7. 2026-09-23 — corrected benchmark, retraining and security pass

Section 6's milestone is now done for the lunar detector comparison.

### 7.1 Code and data changes

| Area / files | Change | Why |
|---|---|---|
| `planetseis/grouped_data.py`, `scripts/build_grouped_lunar.py` | Build/audit `lunar_grouped_v1`: channel + UTC-interval + waveform-identity groups, test > val > train precedence, unioned UTC picks, `evid00029` recovery; refuses to overwrite a cache | Removes the confirmed train/test duplicates without touching frozen artifacts |
| `planetseis/training_state.py`, `planetseis/train.py`, `scripts/train_unet.py`, `planetseis/injection.py` | `--data-dir/--out-dir/--resume/--device`; checkpoints record benchmark ID, manifest hash, random initialization, full config; atomic `best.pt`/`last.pt`; resumable RNG/optimizer/scheduler state | Separate, provenance-checked, interruptible corrected runs |
| `scripts/evaluate_grouped.py` | Validation-only selection with historical grids/tie rules, written to an exclusive `*.selection.json` before any test waveform opens; rejects warm-started or other-dataset checkpoints; loads with `weights_only=True` | Preregistered held-out evaluation |
| `scripts/run_grouped_seeds.sh` (new) | Resumable 5-seed × 2-model runner, one GPU job at a time | Benchmark rule 2 (≥5 seeds) |
| `scripts/aggregate_grouped_seeds.py` (new) | Seed mean/SD, Welch test, acquisition-group bootstrap; refuses mixed manifests or repeated seeds | Summary statistics from locked results |
| `scripts/matched_filter_baseline.py` | `--data-dir` / `--output`; refuses to overwrite grouped results | Baseline on the corrected split |
| Security: `app/streamlit_app.py`, `planetseis/preprocessing.py` | Checkpoints load with `weights_only=True`; `load_trace(max_samples=...)` rejects miniSEED whose gap-filled span exceeds 48 M samples before allocation | Blocks pickle code execution from a swapped `.pt`; blocks a tiny-file memory-exhaustion upload |
| Security: `requirements.txt`, `scripts/deploy_hf.py`, `.github/workflows/ci.yml` | `torch>=2.6` (CVE-2025-32434), `streamlit>=1.37` (CVE-2024-42474); Space container runs as non-root uid 1000; CI token `contents: read` | Dependency, container and CI least privilege |
| Tests | `test_aggregate_grouped.py` (2), gap-fill guard test (1): **159 passed** | Regression coverage |
| Docs | README, benchmark README, report, walkthrough, project review, deploy guide, app "Model & results", paper | Corrected numbers cited first; historical rows labelled |

### 7.2 Corrected results (`results/lunar_grouped_v1_seed_summary.json`)

| Detector | Seeds | F1 mean ± SD | Per-seed F1 (42, 1, 2, 3, 4) |
|---|---:|---:|---|
| SeisCNN | 5 | 0.5306 ± 0.0314 | 0.5614, 0.5333, 0.4906, 0.5600, 0.5075 |
| SpecUNet | 5 | 0.4077 ± 0.0431 | 0.4118, 0.3415, 0.4262, 0.4000, 0.4590 |
| Matched filter (val-tuned) | — | 0.2000 | test-tuned oracle 0.2759, not reportable |
| STA/LTA (val-tuned) | — | 0.1684 | thr_on 7.0 is the grid edge |

Welch p = 0.0012 (SeisCNN > SpecUNet), ratio 76.8 %; acquisition-group
bootstrap ΔF1 95 % CI [0.009, 0.233]. Test = 21 spans / 23 events.

### 7.3 Verification

| Check | Result |
|---|---|
| `python -m pytest -q` | 159 passed |
| `scripts/build_grouped_lunar.py --audit-only` | passed (exit 0), manifest SHA256 `da0d8596…46cf` |
| `scripts/audit_splits.py` (historical) | exit 1, as expected: historical duplicates still documented |
| `scripts/reproduce_headline.py` | historical 0.5405 / 0.4400 reproduce, exit 0 |
| Real resume of a finished run | config + manifest hash matched, log intact, no retraining |
| Frozen artifacts | `models/`, historical splits/results unchanged in Git; historical caches/runs untouched (mtimes July) |

Not rerun on the corrected split: Nakamura FP cross-check, uncertainty /
triage, SNR strata and PR curves, denoise chain, archive scan, Mars. The app
still ships the historical checkpoints. Nothing was committed or pushed.

## 8. 2026-09-23 — handoff reconciliation and repeatable aggregation

Verified the completion report against the current workspace: all ten trained
checkpoints and final logs exist (CNN: 60 epochs; U-Net: 30 epochs). Each
checkpoint's hash, seed, selected epoch, from-scratch flag and manifest hash
match its evaluation. Validation-selection file hashes and operating points
match the stored test outputs. Per-trace counts sum to the recorded scores;
five-seed means/SD and Welch p = 0.0011558462 reproduce from those artifacts.
The grouped dataset audit passes with the unchanged manifest fingerprint.

Found a continuation bug: the aggregator's input glob also read its own
`lunar_grouped_v1_seed_summary.json`, raising `KeyError: 'checkpoints'` on
rerun. `scripts/aggregate_grouped_seeds.py` now accepts seed-numbered experiment
filenames only. A new regression generates the same summary twice and checks
the result. **Full suite: 160 passed in 19.77 s**; aggregation tests: 3 passed.
Original results and model weights were preserved; no new training or test
inference was performed for these checks.

Updated `tasks/RESEARCH_HANDOFF.md` with current status, exact paths and
fingerprint, native PowerShell resume commands, pending paper work and a
copy-paste continuation prompt. Clarified that the group bootstrap averages
the fixed observed seeds, and that persistence of the ranking does not prove
leakage had no effect. A changed test set and labels prevent that causal claim.
Both repositories remain uncommitted; their diff checks pass.

## 8. 2026-09-23 — publication pass (secondary analyses, demo models, paper)

Supersedes the "not rerun" and "app still ships historical checkpoints"
statements in section 7.

| Area | Change | Evidence |
|---|---|---|
| Model backup | All ten corrected runs + training logs + grouped cache copied to `C:\apsis_backups\lunar_grouped_v1_2026-09-23` (outside OneDrive) | 119 files, 404 MB; `sha256sum -c SHA256SUMS` passes; identical to source |
| Demo models | Lunar app uses `models/lunar_grouped_v1_seed42.pt` and `models/unet_lunar_grouped_v1_seed42.pt` at 0.97 and 0.25 / 430 s; `models/lunar_grouped_v1_seed42.json` records hashes, epochs, operating points and evaluation/selection hashes; app refuses a mismatched hash or provenance. Demo NPZ picks updated to UTC-corrected values (waveforms byte-identical). Mars unchanged | package = run = evaluation hashes; 17 app tests incl. tamper refusal |
| Secondary scripts | `--data-dir` (+ `--model/--threshold/--output/--tag`) modes in `crosscheck_nakamura.py`, `uncertainty_eval.py`, `uncertainty_unet.py`, `extra_analysis.py`; defaults unchanged; corrected mode never overwrites; `weights_only=True`; catalog parsed before CUDA init | new `results/*_lunar_grouped_v1_seed42.*` files |
| Paper | Corrected Figures 1–3, Fig. 4 embedded, §4.4/§5/abstract/conclusion updated, over-inference about leakage removed; `md_to_tex.py`, `make_overleaf_zip.py`, generated `.tex`/`.html`, Edge-printed preview PDF, Overleaf zip | `paper/REVISION_NOTES.md` |

Corrected seed-42 secondary results:

| Analysis | Corrected split | Historical split |
|---|---|---|
| SpecUNet FPs matching Nakamura (±300 s) | 9 / 31 (29.0 % vs 1.5 % chance, 0/10,000 perm.) | 9 / 20 (45 % vs 1.7 %) |
| Catalog-adjusted precision | 0.511 (benchmark 0.311) | 0.645 (0.355) |
| SeisCNN window MC σ, false alarm / true event | 0.100 / 0.027 = 3.7× | 5.9× |
| SeisCNN review queue | 30 candidates, 0 real Grade-A events | 50, 1 |
| SpecUNet detection σ FP/TP; clean-FP/real | 0.77; 0.70 | 0.55; 0.49 |
| SeisCNN recall below / above median SNR | 0.636 / 0.750 (median 14.4 dB, n = 23) | — |

Not done: a real LaTeX compile (no TeX engine installed locally), and
corrected-split reruns of the screening ablation, denoise chain, archive scan
and Mars. These analyses describe seed 42 only.

## 9. 2026-09-23 — handoff follow-ups (seed coverage, baseline grid, hardening)

| Area | Change | Evidence |
|---|---|---|
| Secondary analyses, all seeds | `scripts/run_grouped_secondary.py` runs the Nakamura cross-check, both uncertainty analyses and SNR/PR for every seed at its locked operating point (read from the evaluation JSON); `scripts/aggregate_grouped_secondary.py` checks manifest hash, checkpoint SHA256 and deterministic detection counts, then reports mean ± SD | `results/lunar_grouped_v1_secondary_summary.json`; per-seed `*_lunar_grouped_v1_seed<N>.*`; 3 tests |
| STA/LTA grid edge | `scripts/sta_lta_grouped.py` repeats validation-only selection on a 2–50 grid with a locked selection file | `results/sta_lta_extended_lunar_grouped_v1.json`: selects 7.0 again (no validation detections above 7.0); test F1 0.168 unchanged |
| Safe loading | Every model-weight `torch.load` in `app/`, `planetseis/`, `scripts/` uses `weights_only=True`; only the trainer's own `last.pt` resume path keeps full unpickling (NumPy RNG state), with a comment; guard test | all 40 non-resume checkpoints verified loadable; `tests/test_safe_loading.py`; headline reproduction unchanged |
| Streamlit | Removed deprecated `use_container_width` (current default is `width="stretch"`; the parameter's removal date has passed) | app tests pass |

Five-seed secondary results (mean ± SD [range]):

| Analysis | Five seeds | Seed 42 |
|---|---|---|
| SpecUNet FPs within ±300 s of a Nakamura event | 0.43 ± 0.13 [0.29, 0.64]; pooled 47/120; chance 1.5 %; p < 1e-4 every seed | 9/31 |
| Catalog-adjusted precision | 0.62 ± 0.10 (benchmark 0.34 ± 0.04) | 0.511 |
| SeisCNN window MC σ, false/true | 7.9 ± 3.5× [3.7, 12.9] | 3.7× |
| SeisCNN review queue real events | 1 seed of 5 (29–42 candidates each) | 0/30 |
| SpecUNet σ FP/TP | 0.86 ± 0.43 [0.50, 1.60] — below 1 in 4 seeds | 0.77 |
| SeisCNN recall below / above median SNR | 0.62 ± 0.10 / 0.70 ± 0.11 (one seed reverses) | 0.636 / 0.750 |

MC-Dropout auto-accept counts need not equal the deterministic evaluation
(stochastic passes, averaged probabilities); only the deterministic
cross-check counts are required to match, and they do for every seed.

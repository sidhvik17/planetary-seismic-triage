# APSIS — handoff for the next agent

Updated: 2026-09-23, end of the follow-up pass. The benchmark repair,
retraining, evaluation, five-seed secondary analyses, demo model switch,
paper figures/LaTeX twin and local commits are **done and verified**. What is
left is listed in section 4. Read this file fully before touching anything.

## Follow-up pass — completed 2026-09-23 (after the publication pass)

| Item | State | Evidence |
|---|---|---|
| Secondary analyses for all five seeds | Done | `scripts/run_grouped_secondary.py` (reads each seed's locked operating point), `scripts/aggregate_grouped_secondary.py` (checks manifest, checkpoint SHA256 and deterministic counts) -> `results/lunar_grouped_v1_secondary_summary.json`; log `runs/logs_grouped/secondary_seeds1-4.log` |
| STA/LTA grid-edge question | Resolved | `scripts/sta_lta_grouped.py` -> `results/sta_lta_extended_lunar_grouped_v1.json` (+ locked `.selection.json`): a 2–50 grid selects 7.0 again, because no validation event is detected above 7.0; test F1 0.168 unchanged |
| `weights_only=True` in offline scripts | Done | 23 files; only `planetseis/training_state.py` (own `last.pt`, NumPy RNG) keeps full unpickling; all 40 other checkpoints verified; `tests/test_safe_loading.py` guards it; headline 0.5405 / 0.4400 still reproduces |
| Streamlit `use_container_width` | Removed | current default `width="stretch"`; app tests pass |
| Docs + paper | Updated to five-seed figures | README, benchmark README, app About, walkthrough, review, change notes §9; paper abstract/§4.1/§4.4/§5/conclusion, new Fig. 3, regenerated .tex/.html/preview PDF/zip |
| Tests | 166 pass | |

Five-seed corrected secondary results (mean ± SD [range]; seed 42 in brackets):

- SpecUNet FPs within ±300 s of a Nakamura S12 event: 0.43 ± 0.13 [0.29,
  0.64], pooled 47/120, chance 1.5 %, permutation p < 1e-4 for every seed;
  catalog-adjusted precision 0.62 ± 0.10 vs benchmark 0.34 ± 0.04 [9/31, 0.511].
- SeisCNN window MC σ false/true: 7.9 ± 3.5× [3.7, 12.9] [3.7×]; the review
  queue held a real Grade-A event in 1 of 5 seeds.
- SpecUNet σ FP/TP: 0.86 ± 0.43 [0.50, 1.60] — below 1 in four seeds, not
  all [0.77]. Do not say SpecUNet uncertainty "always inverts".
- SeisCNN recall below / above the 14.4 dB median SNR: 0.62 ± 0.10 /
  0.70 ± 0.11; one seed reverses the order [0.636 / 0.750].
- Seed 42 (the demo model) is the weakest seed for catalog matches and
  SeisCNN σ separation: quote the five-seed figures, not seed 42.
- MC-Dropout auto-accept counts legitimately differ from the deterministic
  evaluation (stochastic passes); the deterministic cross-check counts match
  the locked evaluations for every seed.

## Publication pass — completed 2026-09-23

| Item | State | Evidence |
|---|---|---|
| Backup of the 10 corrected runs | Done | `C:\apsis_backups\lunar_grouped_v1_2026-09-23` (runs, `runs/logs_grouped`, grouped cache; 119 files, 404 MB). Verify: `cd` there, `sha256sum -c SHA256SUMS`. SHA256 of `SHA256SUMS` itself: `01c7493514f354c12f9ff8c62bc8033ca505ef78f148b36a52fbf0cacd32d279` |
| Corrected secondary analyses (seed 42, fixed in advance) | Done | `results/nakamura_crosscheck_lunar_grouped_v1_seed42.json` (+ `_detections_…csv`), `results/uncertainty_lunar_grouped_v1_seed42.json`, `results/uncertainty_unet_lunar_grouped_v1_seed42.json`, `results/snr_recall_lunar_grouped_v1_seed42.json`, `results/pr_curve_lunar_grouped_v1_seed42.json`, `docs/figures/pr_curve_lunar_grouped_v1_seed42.png`; logs in `runs/logs_grouped/` |
| Demo uses corrected seed 42 | Done | `models/lunar_grouped_v1_seed42.pt`, `models/unet_lunar_grouped_v1_seed42.pt`, `models/lunar_grouped_v1_seed42.json`; app verifies hash + provenance; SeisCNN 0.97, SpecUNet 0.25 / 430 s; Mars unchanged; demo NPZ picks UTC-corrected (waveforms identical) |
| Paper figures, LaTeX, PDF, Overleaf zip | Done except a real LaTeX compile | `paper/figures/fig{1,2,3}_grouped_*.png`; `paper/md_to_tex.py` -> `ml4ps_2026.tex` + `.html`; `ml4ps_2026_preview.pdf` (Edge-printed HTML, inspected); `ml4ps_2026_overleaf.zip`. **No TeX engine installed**, so the .tex passed a structural lint only |
| Verification | Done | 162 tests pass; grouped audit exit 0; historical audit exit 1 (expected); headline 0.5405/0.4400 reproduce; no tracked change in `models/`, frozen splits, historical results or figures |
| Commits (local, not pushed) | Done | see section 0a |

Seed-42 corrected secondary results (historical split in brackets):

- SpecUNet (0.25 / 430 s): 9 of 31 benchmark FPs lie within ±300 s of a
  Nakamura-catalogued S12 event, 29.0 % vs 1.5 % chance, 0/10,000
  permutations; tolerance counts 0/0/4/9 at ±60/120/180/300 s;
  catalog-adjusted precision 0.511 vs benchmark 0.311 [9/20, 45 %, 0.645].
- SeisCNN (0.97): window MC σ false-alarm/true-event 0.100/0.027 = 3.7× [5.9×];
  ECE raw 0.035, temperature-scaled 0.028 (T = 0.834 fitted on val); the
  review queue added 30 candidates and 0 real Grade-A events [50, 1].
- SpecUNet MC σ FP/TP 0.77, clean-FP/real 0.70 [0.55, 0.49] — still inverted.
- SeisCNN recall 0.636 (11 events) below and 0.750 (12) above the 14.4 dB
  median event SNR.
- Detection counts in every secondary run match the locked seed-42
  evaluations (SeisCNN 16/18/7, SpecUNet 14/31/9).

The earlier core task (repair the lunar benchmark leak, retrain, evaluate) is
**done and verified**; sections 0–3 below record it.

## 0a. Commits (local only, not pushed)

Root repository, branch `archive-scan-and-seed-variance` (previous HEAD
`ecf07ef`):

| Commit | Content |
|---|---|
| `46ee44e` | Acquisition-grouped benchmark, resumable training, locked evaluation, audits, aggregation, runner, tests |
| `41e4db4` | Five-seed corrected results + seed-42 secondary analyses and their `--data-dir` script modes |
| `9ffd610` | App hardening, safe loading, corrected seed-42 lunar models, demo NPZ picks, dependency/CI/Space security |
| `0be17f1` | Docs, change notes, this handoff, todo, lessons |
| `5728e1d` | Follow-up: `weights_only=True` for every model-weight load + guard test |
| `5fa5c56` | Follow-up: five-seed secondary analyses, aggregation with provenance checks, extended STA/LTA grid |
| the commit that adds this row (HEAD at hand-off) | Follow-up: docs, app About text, Streamlit deprecation, handoff, todo |

Paper repository `paper/`, branch `main` (previous HEAD `2e21984`):
`63ebfd5` — corrected draft, Figures 1–3, `md_to_tex.py`, generated
`.tex`/`.html`, preview PDF, Overleaf zip, revision notes; `98d5fcd` —
five-seed secondary results, new Fig. 3, regenerated exports.

Nothing was pushed. Check with `git log --oneline -5` in each repository.

## 0. Earlier reconciliation check (2026-09-23, before the publication pass)

The user supplied another account's completion report. This continuation
checked the actual local files before making further changes:

- All **10 trained runs** exist, with `best.pt`, `last.pt` and completed logs:
  five CNN runs reached 60 epochs; five U-Net runs reached 30 epochs.
- Every `best.pt` loads with `weights_only=True`; its SHA256 matches its
  recorded evaluation; seed, selected epoch, random-initialization flag and
  dataset fingerprint agree. All ten validation-selection hashes match, and
  their operating points match the test result files.
- Recomputed per-span TP/FP/FN totals, F1 means/sample SD and Welch p-value
  from the saved predictions/results: the reported numbers agree.
  Recomputed Welch p = **0.0011558462** (reported rounded as 0.0012).
- Fresh grouped audit passed with all 76 events accounted for and no
  cross-split interval, raw-waveform or processed-waveform duplicates.
- At the process check, no training/evaluation job was running. The two Python
  processes were the wrapper and worker for the local Streamlit server.
- Found and fixed a **summary regeneration bug**: `load_runs()` included
  `lunar_grouped_v1_seed_summary.json` as an experiment and raised
  `KeyError: 'checkpoints'`. It now accepts only seed-numbered experiment
  filenames. A regression test runs aggregation twice with its default
  output filename and verifies identical results. Three aggregation tests pass.
- The full suite passed **159 tests before this fix** and **160 tests in
  19.77 seconds after it**, including the new regeneration regression.
- No new training or test-set inference was needed for this reconciliation.
  Original result JSONs and model files were not regenerated or replaced.

**Next useful work:** see section 4. Do not retrain completed models to
continue this task. The paper draft is `paper/ml4ps_2026.md`.

## 1. Project and non-negotiable constraints

APSIS (package `planetseis`) detects, times, denoises and triages single-channel
Moon (Apollo) and Mars (InSight) seismic events with two detectors:
**SeisCNN** (supervised 1D CNN, 118K params) and **SpecUNet** (spectrogram
U-Net, 1.9M params, trained on synthetic mixtures of catalog-derived templates
injected into noise). Stack: Python / ObsPy / PyTorch / Streamlit. B.Tech major
project plus a workshop-paper draft.

- Keep the purpose, stack and both architectures. No new frameworks or models.
- Application repo: `C:\Users\ASUS\OneDrive\Desktop\major`, branch
  `archive-scan-and-seed-variance`; all work to 2026-09-23 is committed
  locally (section 0a), not pushed. Do not reset, rebase or switch branches.
- The manuscript is a **separate Git repo** in `paper/` (branch `main`,
  committed locally, not pushed). The root repo ignores `paper/`.
- Never overwrite: `models/`, `benchmark/lunar_splits.json` and other frozen
  splits, `data/cache/lunar/`, `runs/lunar*`, `runs/unet_lunar*` (historical),
  original `results/*.json`. New results are written with exclusive creation.
- Never warm-start corrected runs from old weights (they saw test acquisitions).
- Do not claim validated deployment, "zero labels" (templates come from catalog
  picks), or parity between the two detectors.

## 2. Environment landmines

- Python: `.venv\Scripts\python.exe`, run from project root. Set
  `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2`.
- **One CUDA process at a time** (two concurrent CUDA jobs segfault on this
  laptop). DataLoader `num_workers=0` on Windows. CPU evaluation beside a GPU
  training job is fine and was used this session.
- The repo lives on **OneDrive**. `runs/` checkpoints written every epoch get
  synced; free RAM fell to ~0.5 GB and one epoch took 311 s instead of ~80 s.
  Consider moving future `--out-dir` targets outside OneDrive.
- A Streamlit server may already be running on `http://127.0.0.1:8501/`.
- GPU: RTX 4060 Laptop 8 GB. SeisCNN run ≈ 4–13 min, SpecUNet run ≈ 25–45 min.

## 3. What was done (2026-09-23 session)

### 3.1 Corrected benchmark `lunar_grouped_v1` (built earlier, verified now)

- `planetseis/grouped_data.py`, `scripts/build_grouped_lunar.py`. Groups the 76
  Grade-A events by channel + overlapping UTC interval + identical samples;
  historical test > val > train; identical copies merged with unioned picks
  (catalog UTC minus actual trace start); `evid00029` recovered (file is HR02,
  catalog says HR00; pick 36,731 s, not 46,500 s) and sent to train.
- Counts: 38/11/18 groups, 39/11/21 traces, 42/11/23 events. Adjacent-day
  partial overlaps are ~2 s, kept as separate spans in the same split.
- `--audit-only` passes (exit 0). Manifest SHA256
  `da0d85966c1090b1b088e42ab0f1957fcb73ec4c2c287505a5aa7245facb46cf`,
  published copy `benchmark/lunar_grouped_v1.json`; cache
  `data/cache/lunar_grouped_v1/` (gitignored).

### 3.2 Training (all from random initialization, 10 runs)

- Runner: `scripts/run_grouped_seeds.sh` (new, resumable: an out-dir with
  `last.pt` gets `--resume`). Logs: `runs/logs_grouped/`.
- SeisCNN: 60 epochs, seeds 42/1/2/3/4 -> `runs/lunar_grouped_v1[_sN]/`.
- SpecUNet: 30 epochs × 6400 samples, batch 32, patience 8, unscreened noise
  pool, seeds 42/1/2/3/4 -> `runs/unet_lunar_grouped_v1[_sN]/`.
- **Decisions and deviations from the earlier plan:** SpecUNet uses 30 epochs,
  not the 40 planned before, matching the recorded historical epoch budget
  (cosine T_max = epochs). The new grouping, unioned/UTC-corrected labels,
  recovered trace and deterministic injection sampling mean this is not an
  experiment isolating only the effect of leakage. Five seeds, because
  benchmark rule 2 requires at least 5. Unscreened, because the frozen headline
  model was unscreened and the historical screening comparison detected no
  difference (p = 0.92); that null result does not prove equivalence.
- Real resume check: `--resume` on the finished seed-42 CNN matched config and
  manifest hash, kept all 60 log rows and trained nothing.

### 3.3 Evaluation (validation-locked, test opened once)

- `scripts/evaluate_grouped.py`, one invocation per family and seed, run on CPU
  while the GPU trained. Each wrote `results/lunar_grouped_v1_seed<N>_{cnn,unet}.json`
  plus a `.selection.json` written **before** test data was opened
  (`test_waveforms_opened: false` in all 10).
- STA/LTA is in the seed-42 CNN file. Matched filter:
  `scripts/matched_filter_baseline.py --data-dir data/cache/lunar_grouped_v1`
  -> `results/matched_filter_lunar_grouped_v1.json`.
- `scripts/aggregate_grouped_seeds.py` (new) ->
  `results/lunar_grouped_v1_seed_summary.json`.

| lunar_grouped_v1 test (21 spans, 23 events, ±120 s) | Seeds | F1 mean ± SD | Mean P / R | Per-seed F1 (42, 1, 2, 3, 4) |
|---|---|---|---|---|
| SeisCNN | 5 | **0.531 ± 0.031** | 0.448 / 0.661 | 0.561, 0.533, 0.491, 0.560, 0.508 |
| SpecUNet | 5 | 0.408 ± 0.043 | 0.344 / 0.530 | 0.412, 0.341, 0.426, 0.400, 0.459 |
| Matched filter (val-tuned; 300 s, k = 10) | — | 0.200 | 0.429 / 0.130 | test-tuned oracle 0.276 (not reportable) |
| STA/LTA (val-tuned; thr_on 7.0 = grid edge) | — | 0.168 | 0.111 / 0.348 | — |

- Welch t = 5.15, **p = 0.0012**, ΔF1 = 0.123; SpecUNet = 76.8 % of SeisCNN.
- Bootstrap over the 18 test acquisition groups (averaged over seeds): SeisCNN
  [0.437, 0.619], SpecUNet [0.319, 0.493], difference **[0.009, 0.233]**.
- Interpretation: the ordering SeisCNN > SpecUNet > the evaluated classical
  baselines persists on this corrected acquisition-disjoint benchmark.
  Corrected means are not lower than the historical ones (0.498; 0.372–0.379),
  but the data and protocols differ. **Do not write that leakage caused no
  inflation**: these experiments do not identify its exact effect.
- The group bootstrap resamples acquisition groups and averages the five
  fixed trained seeds; it does not resample seeds or estimate every source of
  uncertainty. Welch p addresses seed variation conditional on this test set.
  The revised holdout reuses historically studied acquisitions; it is not a
  newly collected, previously unseen external benchmark.

### 3.4 Security review and fixes

| Finding | Severity | Fix |
|---|---|---|
| App loaded checkpoints with `torch.load(weights_only=False)` (pickle code execution if a `.pt` in `runs/` or `models/` is replaced) | Medium | `weights_only=True` in `app/streamlit_app.py` and `scripts/evaluate_grouped.py`; all shipped and new `best.pt` files verified loadable |
| Uploaded miniSEED with records years apart -> `st.merge(fill_value=0)` allocates the whole gap (tiny file, many GB, crash) | Medium (public demo DoS) | `load_trace(..., max_samples=)` rejects gap-filled span > 48 M samples before allocation; app passes it; regression test added |
| `torch>=2.3` allows versions with CVE-2025-32434 (`weights_only` bypass) | Medium | `torch>=2.6` in `requirements.txt` and the Space requirements |
| `streamlit>=1.35` allows CVE-2024-42474 (Windows static-file path traversal) | Low | `streamlit>=1.37` |
| Hugging Face Space Dockerfile ran as root | Low | `useradd -u 1000`, `USER user` in `scripts/deploy_hf.py` |
| CI workflow had the default token scope | Low | `permissions: contents: read` |
| Secret scan (tracked files and history for HF/GitHub/OpenAI/AWS/private keys) | — | none found; `HF_TOKEN` only read from environment |
| Upload filename rendered through `st.caption` markdown | Informational | session-scoped, no raw HTML; left unchanged |
| ~25 offline research scripts and `training_state.prepare_run` still use `weights_only=False` on local files | Informational | left unchanged (local researcher files; `last.pt` stores NumPy RNG state) |

### 3.5 Documentation updated

README (new corrected-benchmark section first), `benchmark/README.md`,
`docs/report.md`, `PROJECT_WALKTHROUGH.md`, `docs/PROJECT_REVIEW.md`,
`docs/deploy.md` (security notes), `docs/CHANGES_AND_RESEARCH_NOTES.md`
(section 7), app "Model & results" tab (corrected table; the demo has since
switched to the corrected seed-42 checkpoints), `tasks/todo.md`, and in the paper repo
`paper/ml4ps_2026.md` (new §4.1, method paragraph, abstract, limitations,
conclusion) and `paper/REVISION_NOTES.md`.

### 3.6 Verification at end of session

- `python -m pytest -q`: **159 passed** (was 156; +3 new tests).
- `build_grouped_lunar.py --audit-only`: exit 0. `audit_splits.py`
  (historical): exit 1, expected. `reproduce_headline.py`: historical
  0.5405 / 0.4400 reproduce, exit 0. `compileall` and `git diff --check` pass.
- Frozen artifacts unchanged (no tracked diffs in `models/`, historical splits
  or results; historical caches and runs keep July mtimes).
- Live app: SpecUNet lunar demo `evid00003` analysed with the new loader
  (score 0.76 near the 12,720 s pick, same as before); new table renders.

## 4. What remains (priority order)

1. **Compile the LaTeX for real.** No TeX engine is installed on this laptop.
   Either upload `paper/ml4ps_2026_overleaf.zip` to Overleaf, or (with the
   user's permission, since it downloads software) install MiKTeX or
   Tectonic and run `pdflatex ml4ps_2026.tex` twice in `paper/`. Inspect the
   PDF (figures, the three tables, abstract), then commit the PDF in `paper/`.
   After any Markdown edit: `python md_to_tex.py; python make_overleaf_zip.py`.
   If the venue needs its own style (e.g. NeurIPS workshop), swap the preamble
   in `md_to_tex.py`; do not hand-edit the generated .tex.
2. **Push only when asked.** Both repositories have local commits (section
   0a). Nothing was pushed. The paper repository's remote is private.
3. **Optional corrected-split reruns** (still historical-split only): the
   Nakamura screening ablation, denoise chain, continuous-archive scan with a
   corrected checkpoint, and Mars (unaffected by the lunar leak). Use the same
   pattern as the secondary scripts: `--data-dir`, new output names,
   exclusive creation, and a fixed seed or all five seeds.
4. **Research limits to keep stating:** 23 test events in 21 spans; the audit
   covers intervals and complete-waveform identity only (not repeated
   deep-moonquake sources); secondary analyses vary strongly across seeds
   (quote mean ± SD, not seed 42); archive and
   Mars results are historical-split analyses; the archive scan gate failed
   (B3). Never write that the leak had no effect: split, labels and test
   population all changed together.
5. **Superseded exports in `paper/`** (`paper_ieee*.tex`,
   `planetseis_ieee_overleaf.zip`, `overleaf_pkg/`, `paper.md`,
   `paper_final.md`, `draft_v2.md`, decks) contain retracted claims. Do not
   submit them; `ml4ps_2026.md` is the source of truth.

## 5. Useful commands

```powershell
git status --short; git -C paper status --short
.venv\Scripts\python -m pytest -q
.venv\Scripts\python scripts\build_grouped_lunar.py --audit-only
.venv\Scripts\python scripts\aggregate_grouped_seeds.py
.venv\Scripts\python scripts\run_grouped_secondary.py      # skips outputs that exist
.venv\Scripts\python scripts\aggregate_grouped_secondary.py
.venv\Scripts\python scripts\reproduce_headline.py
bash scripts/run_grouped_seeds.sh            # resumes; finished runs are no-ops
# A new evaluation must use a NEW output name (exclusive creation):
.venv\Scripts\python scripts\evaluate_grouped.py --data-dir data/cache/lunar_grouped_v1 --unet runs/unet_lunar_grouped_v1/best.pt --output results/<new_name>.json --skip-baseline
```

The aggregation command regenerates a derived summary and is now safe to
repeat with that summary already present. Evaluation outputs and selection
locks remain exclusive: do not delete or overwrite them to tune on test data.
The matched-filter script also computed a labelled test-tuned oracle; only
the validation-selected baseline score belongs in the primary comparison.

### Native PowerShell resume commands (only if a run is actually interrupted)

Use these exact recipes for seed 42; for seeds 1–4 use `--seed N` and the
corresponding `_sN` directory. Keep the original total epochs when resuming:
they determine the learning-rate schedule. Finished runs need no restart.

```powershell
Set-Location C:\Users\ASUS\OneDrive\Desktop\major
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
.venv\Scripts\python -m planetseis.train --body lunar --epochs 60 --seed 42 --data-dir data/cache/lunar_grouped_v1 --out-dir runs/lunar_grouped_v1 --device cuda --resume
.venv\Scripts\python scripts/train_unet.py --body lunar --epochs 30 --epoch-len 6400 --batch-size 32 --patience 8 --seed 42 --data-dir data/cache/lunar_grouped_v1 --out-dir runs/unet_lunar_grouped_v1 --device cuda --resume
```

## 6. Copy-paste prompt for another ChatGPT / Claude account

> Continue APSIS in `C:\Users\ASUS\OneDrive\Desktop\major`. First read
> `tasks/RESEARCH_HANDOFF.md`, `tasks/todo.md`,
> `docs/CHANGES_AND_RESEARCH_NOTES.md`, and both repositories' Git status.
> Preserve the purpose, stack, architectures and historical artifacts. The
> corrected `lunar_grouped_v1` benchmark, five-seed training, five-seed
> secondary analyses and the paper revision are complete and committed
> locally; inspect the stored evidence instead of training again. Work from
> section 4 of this handoff.
> Read values from result JSONs, preserve validation/test separation, and write
> new experiment results to separate paths. Do not choose models or operating
> points using test scores. Explain actual changes and verification. Keep this
> handoff updated before any context/token limit, including commands, files,
> running process IDs, checkpoints and remaining tasks. Push only when
> requested by the user.

Another local account needs access to the same workspace. A remote/cloud chat
cannot access this Windows path from the handoff text alone: provide the code
and paper repositories plus relevant result JSONs. Re-execution additionally
needs the ignored `data/cache/lunar_grouped_v1/`, raw packet for rebuilding,
and all ten `runs/*grouped_v1*/` model directories. These model files are not
included in Git. Do not copy `.venv` between machines; recreate the existing
Python stack from the repository requirements and recorded runtime versions.

## 7. Checks from the earlier reconciliation (superseded by the status blocks at the top)

- Dataset manifest SHA256 unchanged: `da0d85966c1090b1b088e42ab0f1957fcb73ec4c2c287505a5aa7245facb46cf`.
- Baselines unchanged: code HEAD `ecf07efe6c95eb026870cede5cb06dcef3769d25`;
  paper HEAD `2e219841c05f5785c12613e2c6f46851277c7613`.
- Both repositories' `git diff --check` passed; Python compilation passed.
- No tracked differences in shipped models, original split manifests or
  previously tracked result JSONs.
- Full post-fix test rerun: **160 passed in 19.77 s** (exit 0).
- Historical headline reproduction, live UI demonstration and the secret
  scan in section 3 describe the preceding account's recorded checks; those
  three checks were not repeated during this handoff-only reconciliation.

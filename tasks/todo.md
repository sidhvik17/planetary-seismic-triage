# PR #1 CI failure — 2026-09-23

`CI / test` failed on `d21a265`: 4 collection errors,
`ModuleNotFoundError: No module named 'scripts'` (test_aggregate_grouped,
test_aggregate_secondary, test_grouped_evaluation,
test_secondary_interpretation).

- [x] Reproduce with CI's entry point: `.venv/Scripts/pytest.exe tests -q`
  gives the same 4 errors; local runs had used `python -m pytest`.
- [x] Fix: `[tool.pytest.ini_options] pythonpath = ["."]` in `pyproject.toml`.
- [x] Verify in a fresh clone (no ignored data/models/results) with
  `pytest tests -q`: 172 passed, 1 skipped (data-cache test, as in CI).
  Main checkout: 173 passed.

## Review

The tests were correct. Only the invocation differed. Declaring the path in
pytest config makes `pytest` and `python -m pytest` behave the same. The
scripts and tests themselves were left unchanged.

---

# Final handoff completion — 2026-09-23

The latest request authorizes finishing the remaining local build and checks.
Reconciled the actual workspace first: root `2709895`, paper `98d5fcd`, both
clean. Training and secondary inference are already complete; preserve them.

- [x] Read current handoff and Git histories; reconcile completed work.
- [x] Obtain portable Tectonic 0.17.0 outside the repository; verify official
  release archive SHA256 and executable version.
- [x] Compile actual LaTeX PDF, inspect every page, and refresh Overleaf export.
  `paper/build_paper.py` (Tectonic 0.17.0): 10 pages, no overfull boxes, all
  pages viewed in the browser pane; zip and `ml4ps_2026_build.json` refreshed.
- [x] Check secondary-analysis labels and claims against saved evidence; fix
  any confirmed errors without repeating completed training or inference.
  Codex pass found two errors; Claude re-derived both independently:
  (a) stored `sigma_separation_fp_over_tp` used only catalog-unmatched FPs
  (seed 42 0.00718/0.00929 = 0.77); all-FP ratio 1.06 ± 0.38 [0.68, 1.66];
  (b) 45/47 ±300 s catalog matches lie within 360 s of a Grade-A pick in the
  same span; the other two (seeds 2, 4) are ~10 h from any pick, one event.
- [x] Figure 3: show existing-label vs additional-catalog split; regenerate.
  Stacked bars from the audit JSON (0/1/23/47 reproduced: 0+0, 0+1, 21+2, 45+2).
- [x] Remove withdrawn numbers from change notes, handoff, revision notes, memory.
  Change notes §10, revision notes, handoff final-pass block, 3 memory files.
- [x] Independently verify model backup, provenance, full tests, and audits.
  173 passed (3 stale aggregator fixtures fixed + 1 new operating-point case);
  grouped audit 0; historical audit 1 (expected); compileall 0; headline
  0.5405/0.4400; backup 119/119; demo = run = evaluation SHA256; summary and
  audit JSONs regenerate byte-identically.
- [x] Update change report and handoff; commit final scoped changes locally.

## Review

Codex found and fixed two real errors in the earlier secondary claims: the
catalog "matches" are mostly late/early hits on already-labelled events (45/47),
and the SpecUNet σ ratio had excluded catalog-matched FPs (1.06 ± 0.38, not
0.86 ± 0.43). Both re-derived independently before accepting them. Figure 3
now shows the split; no training or inference was rerun. Not pushed.

---

# Handoff §4 follow-ups — 2026-09-23 (continued)

Resume from `tasks/RESEARCH_HANDOFF.md` §4. Same constraints: purpose, stack,
architectures and historical artifacts preserved; validation-only selection;
new results in new files; no push. Expensive optional reruns (screening
ablation, denoise chain, archive scan) remain separate research extensions.
The final completion pass above supersedes the earlier LaTeX-install blocker.

- [x] Secondary analyses for seeds 1–4 at each seed's locked validation point;
  aggregate mean ± SD across all five seeds (new script + test).
  Nakamura 0.43 ± 0.13 (47/120); SeisCNN σ 7.9 ± 3.5×; SpecUNet σ 0.86 ± 0.43
  (below 1 in 4/5 seeds); seed 42 is the weakest seed on the first two.
- [x] STA/LTA: its validation pick sits at the grid edge (7.0); rerun with an
  extended grid, selection on validation, as a separate new result.
  2–50 grid selects 7.0 again (no validation detections above it).
- [x] Offline scripts: load plain `best.pt` files with `weights_only=True`.
  23 files; resume loader excepted; guard test; headline reproduces.
- [x] Check the Streamlit `use_container_width` deprecation on the installed version.
  Deprecated (removal date passed); argument removed, default is stretch.
- [x] Update README/benchmark/paper with the seed-level secondary results;
  regenerate .tex/.html/preview PDF/Overleaf zip.
- [x] Fix stale handoff lines and stale memory facts found by the prompt audit.
- [x] Full tests + audits; commit locally in both repositories.
  166 passed; audits 0 / 1 (expected); headline reproduces. Root `5728e1d`,
  `5fa5c56` + docs commit; paper `98d5fcd`. Not pushed. Still open: real
  LaTeX compile (needs a TeX install), optional historical-only reruns.

---

# Publication completion and commits — 2026-09-23

User requests completion of the remaining handoff items, including commits in
both repositories and a current continuation file. Keep the scientific purpose,
stack and architectures. Seed 42 is the fixed representative for secondary
analyses and the lunar demo; retain its locked validation operating points.

- [x] Back up all ten corrected runs outside OneDrive with checksums.
  `C:\apsis_backups\lunar_grouped_v1_2026-09-23` (runs, logs, grouped cache;
  119 files, 404 MB; `sha256sum -c SHA256SUMS` passes; identical to source).
- [x] Rerun corrected-split Nakamura, MC uncertainty and signal-strength/PR analyses.
  Seed 42, `--data-dir` modes (exclusive outputs). Nakamura 9/31 FPs match
  (29.0 % vs 1.5 %, 0/10,000); SeisCNN σ 3.7×; SpecUNet σ ratio 0.77/0.70;
  SNR recall 0.636/0.750. Counts match the locked seed-42 evaluations.
- [x] Package corrected lunar seed-42 models and verify the demo/deployment paths.
  Package hashes = source runs = evaluation records; app tests updated for
  0.97 and 0.25/430 s plus a tampered-checkpoint refusal test (17 pass).
- [x] Regenerate figures, synchronize Markdown/LaTeX, compile and inspect the PDF.
  Figures 1–3 regenerated from corrected JSONs; `md_to_tex.py` generates the
  .tex/.html twins; Overleaf zip built. **No local TeX engine**: .tex lint
  passed, Edge-printed HTML preview PDF inspected; real LaTeX compile pending.
- [x] Run relevant/full tests, review scientific claims and verify preserved artifacts.
  162 passed; compileall ok; grouped audit 0; historical audit 1 (expected);
  headline reproduces; no tracked changes in models/splits/results/figures;
  live app ran the corrected seed-42 SpecUNet with provenance. Removed the
  "duplicates did not inflate" over-inference from README/walkthrough/paper.
- [x] Commit scoped application changes and the separate paper repository.
  Root: `46ee44e`, `41e4db4`, `9ffd610` + docs commit; paper: `63ebfd5`. Not pushed.
- [x] Record completed work, commit IDs, backups, commands and remaining limitations in handoff.
  Remaining: real LaTeX compile (no TeX engine), optional reruns, push on request.

Execution state and exact commands are recorded in `tasks/RESEARCH_HANDOFF.md`.

---

# Corrected lunar benchmark and retraining — 2026-09-23

Goal: remove acquisition leakage, union labels, retrain the same two models,
and report validation-selected held-out results. Preserve the original idea,
stack and historical artifacts. Keep `tasks/RESEARCH_HANDOFF.md` current for
continuation from another account.

- [x] Inspect local hardware, datasets and raw acquisition metadata.
- [x] Define split policy before any corrected model training or test scoring:
  connected interval/hash groups; historical test > validation > train;
  recovered unassigned events go to train; exact-copy pick unions use UTC.
- [x] Implement/test builder and audit; freeze a separate manifest/cache.
- [x] Implement/test dataset provenance and resumable training.
- [x] Train SeisCNN and SpecUNet from scratch on the corrected data (5 seeds each).
- [x] Select thresholds/duration on validation and evaluate held-out test data.
- [x] Run regression checks and verify historic artifacts remain unchanged.
- [x] Update paper, app provenance, change report and continuation instructions.
- [x] Security review and hardening (checkpoint loading, upload gap-fill
  guard, dependency floors, non-root Space container, CI token scope).

## Review

- **Latest handoff reconciliation:** all 10 checkpoint hashes, dataset/seed
  provenance, validation-selection hashes, per-trace counts and stored seed
  statistics verified. Fixed aggregation reading its own summary as an input;
  added a repeat-generation regression. **160 tests pass in 19.77 s**.
  `tasks/RESEARCH_HANDOFF.md` now includes verified state, remaining paper work,
  native PowerShell resume commands and a prompt for another account. No new
  model training or test inference was performed in this reconciliation.
- **Corrected results** (`results/lunar_grouped_v1_seed_summary.json`):
  SeisCNN F1 0.531 ± 0.031, SpecUNet 0.408 ± 0.043 (5 seeds each, from
  scratch, validation-locked operating points); Welch p = 0.0012;
  group-bootstrap ΔF1 CI [0.009, 0.233]; matched filter 0.200; STA/LTA 0.168.
- **Decisions:** SpecUNet 30 epochs (historical recipe, not the 40 first
  planned); 5 seeds per benchmark rule 2; unscreened noise pool (frozen
  headline setting; screening had no effect). Evaluations ran on CPU while the
  GPU trained, to avoid two concurrent CUDA processes.
- **Verification:** 159 tests pass; grouped audit exit 0; historical audit
  exit 1 (expected); headline reproduction exit 0; real resume check passed;
  live app analysis with `weights_only=True` loading works.
- **Not done:** corrected-split Nakamura cross-check, uncertainty, SNR strata,
  PR curves, figures, LaTeX/PDF export, commits. See `tasks/RESEARCH_HANDOFF.md`.

---

# Runtime and result verification — 2026-09-22

Goal: run and show the existing APSIS project, verify its main workflows and
recorded results, and fix demonstrated failures without changing its purpose
or Python / ObsPy / PyTorch / Streamlit stack.

- [x] Start or reuse the local server and open a visible, persistent app tab.
- [x] Run lunar and Mars analysis, inspect denoising, and complete live triage.
- [x] Verify the full tests, Python compilation, and historical headline scores.
- [x] Resolve demonstrated issues and document actual verification coverage.
- [x] Review both Git repositories and revise the current research manuscript.
- [x] Write an accumulated Markdown change report against the prior Git versions.
- [x] Leave the working app available and explain its outputs and limitations.

## Review

- **Software verification:** 115 tests passed in 71.02 s; Python compilation
  passed. The historical lunar reproduction command exited 0 with SeisCNN
  P/R/F1 0.5556/0.5263/0.5405 and SpecUNet 0.3548/0.5789/0.4400.
- **Live browser:** Lunar SpecUNet produced a candidate at 12,858 s versus the
  12,720 s catalog pick, outside the benchmark's ±120 s matching tolerance.
  Mars SpecUNet produced 2,115.6 s versus the 2,130 s pick; Mars SeisCNN produced
  809.3 s and 2,170 s. Lunar and Mars denoised plots rendered successfully.
- **Triage:** lunar `evid00003` playback completed at 8×: 138 windows screened,
  one auto-accepted event, three review candidates, 96.4% of windows not queued.
  MC-Dropout outputs can vary. This percentage does not measure transmitted
  bytes. Model & results rendered its metrics and research caveats.
- **Split audit:** rerun with the local cache confirmed both exact train/test
  waveform duplicates (expected exit 1). The audit is evidence of an unresolved
  scientific limitation, not a corrected benchmark.
- **Git:** application branch `archive-scan-and-seed-variance`, baseline
  `ecf07efe6c95eb026870cede5cb06dcef3769d25`. The existing nested `paper/`
  repository is on `main`, baseline `2e219841c05f5785c12613e2c6f46851277c7613`.
  Its layout is preserved. No commits, pushes, branch switches or remote fetches.
- **User deliverables:** `docs/CHANGES_AND_RESEARCH_NOTES.md` records accumulated
  code/app/research changes and checks. `paper/ml4ps_2026.md` is revised, with
  a local pre-revision backup and `paper/REVISION_NOTES.md`. Older paper exports
  and figures are unchanged; no regenerated PDF or LaTeX build is claimed.
- **Local app:** `http://127.0.0.1:8501/`, left running with a visible browser
  tab. Choose a demo and detector, then Analyze; enable denoising for SpecUNet.
  Triage has its own Start control. Catalog markers are comparison references,
  and SpecUNet mask-energy scores are not calibrated event probabilities.
- **Coverage limits:** no fresh retraining, optimizer/resume validation, full
  offline ablation rerun, external service audit or flight/cloud deployment.
  Acquisition-grouped splitting, unioned picks and retraining remain required
  before independent lunar performance claims. Purpose and stack are preserved.

---

# Project improvement review — 2026-09-15

Scope: understand APSIS and improve demonstrated reliability and usability issues
while preserving its scientific purpose, existing detector architectures, Python /
ObsPy / PyTorch / Streamlit stack, frozen checkpoints, and evaluation protocol.

User-requested goals for this pass:

1. **Reliable inputs:** reject malformed traces with useful errors while keeping
   clean lunar/Mars preprocessing identical; prove this with regression tests.
2. **Usable demo:** analyze bundled traces with both detectors, run inference
   only on request, retain result settings, and isolate tab failures.
3. **Trustworthy evidence:** ship a repeatable split audit, disclose confirmed
   overlap, correct stale claims, and document the grouped-retraining followup.

- [x] Map the architecture, demo workflows, and research constraints from files.
- [x] Run the existing test suite and inspect independent core/app/docs audits.
- [x] Select and document a small set of evidence-backed improvements.
- [x] Implement the selected changes and focused regression coverage.
- [x] Verify tests and demo behavior, inspect the diff, and record results.

## Review

Baseline: 44 tests pass (29.96 s). Selected work: explicit app analysis and
denoising controls, bundled demo analysis, published operating-point defaults,
clear input errors and per-tab failure isolation, validated trace loading, and
correction of stale statistical claims. Detector/scoring algorithms, models,
dependencies, and frozen results remain unchanged. Additional scientific
correctness concerns will be documented with evidence for a separately
versioned benchmark review.

### Completed goals

1. **Reliable inputs achieved:** CSV and ObsPy validation rejects malformed,
   non-finite, mistimed, or ambiguous channel inputs. Tests prove valid lunar
   and Mars preprocessing remains bit-for-bit identical to the prior pipeline.
2. **Usable demo achieved:** explicit Analyze action, built-in preprocessed
   demos, opt-in denoising, bounded CPU batches, retained result settings,
   published default gates, and local error handling. Hidden tabs no longer
   start inference. Short SpecUNet input explains its full-window requirement.
3. **Trustworthy evidence achieved:** read-only audit and portable SHA256
   evidence confirm two duplicated train/test waveforms. README, benchmark,
   report, demo, and walkthrough disclose the limitation and correct stale
   supervision, seed-comparison, and expanded Mars refinement claims.

### Verification

- Full suite: **115 passed in 67.68 s** (baseline 44).
- Real-checkpoint AppTest: both detectors passed on bundled lunar and Mars
  traces; Mars denoising returned finite, full-length output; completed results
  persisted after unrelated control changes.
- Real CSV upload passed through the loader, preprocessing, SeisCNN and UI;
  the real Mars triage playback completed successfully.
- Split audit: expected exit **1**, 75 filenames / 70 timestamped acquisitions,
  5 duplicated acquisitions, 2 confirmed identical cross-split waveform pairs.
- `git diff --check`: passed. Frozen split manifests, checkpoint weights,
  detector/scoring algorithms, and original result JSON files are preserved.
- A cold app check initially timed out while loading the native PyArrow
  dependency; after import completed, app checks passed without exceptions.
- Existing Streamlit container-width deprecation notices remain; the current
  API is retained for compatibility with the declared Streamlit minimum.

### Research work still required

The **benchmark is not repaired** by these software improvements. Group actual
acquisition spans, union event picks, version new splits/caches, retrain, and
reevaluate before making independent-test performance claims. Follow the
specific plan in `docs/PROJECT_REVIEW.md`. Do not reinterpret the historical
"no test leakage" audit entry below as current evidence; the 2026-09-15
waveform comparison supersedes it.

Historical publication roadmap below is preserved for context.

---

# Roadmap to publication — two-track

**Decision (2026-07-25):** two-track. Track A = arXiv preprint + ML workshop paper
(weeks, current data). Track B = Earth and Space Science / SRL data paper on lunar
catalog extension (months, continuous archive). Local GPU only.

Rationale: the ML audience wants the injection-only label-scarcity claim; the
seismology audience wants the catalog. One paper cannot serve both. Track A ships
from work already done; Track B is where the novelty actually lives.

---

## Findings that drive this plan

Verified during audit (2026-07-25):

- Headline table reproduces bit-exact from shipped checkpoints; `models/*.pt` are
  md5-identical to `runs/*/best.pt`; `unet_mars_ext_v3` == `unet_mars_ext`, so the
  Mars operating point is legitimately val-derived. No test leakage found.
- Greedy scorer == Hungarian optimal assignment on both lunar and mars_ext test
  sets. No published number is affected.
- **75/75 Grade-A picks match a Nakamura S12 event within ±60 s** — benchmark
  labels are a strict subset of Nakamura. Crosscheck design is sound.
- **8.4 % of injection noise-pool window starts (2.0 M of 24.0 M) fall inside the
  guard radius of a catalogued Nakamura event** — real event energy trained with a
  zero mask target.
- All 183 mseed files are curated single-event snippets. n=19 test events is
  structural. This is the ceiling Track B exists to break.
- Nakamura parent catalog: 5,358 S12 events; also at S14/S15/S16 =
  4,152 / 1,963 / 2,589; typed M/C/A/Z = 1,787 / 1,744 / 1,359 / 255;
  1,361 deep-nest labels.

---

## Positioning — the claim, and what it must not be

**Claim the triad:** label-free (synthetic-negative-only) detector + full
continuous four-station PSE long-period archive + **type-stratified recall**
against Nakamura's deep / impact / shallow / artificial classes. That
combination is absent from the literature.

**Do NOT claim** "first ML on Apollo data" (false) or "first new events"
(false). Both are trivially refuted and would cost the paper its credibility.

Prior art to cite and explicitly distinguish:

| Work | What it did | Why we differ |
|---|---|---|
| Bulow et al. 2005/2007 | Matched filter on deep-moonquake clusters; +123 A1, +503 via stacking | Deep-only; needs known templates; cannot find non-repeating events |
| Knapmeyer-Endrun & Hammer 2015 | HMM, **Apollo 16 only**, 3 yr, 200+ new events | **The type-stratified-recall precedent — acknowledge it.** Single station; we do four, full archive |
| Civilini 2021/2023 | CNN + transfer from terrestrial; the 12,085-event graded catalog | That catalog is **Apollo 17 LSPE short-period**, not a PSE LP scan. Attribute carefully |
| Onodera 2024 | Denoise + **manual** classification, >22,000 uncatalogued, 46 new shallow | Short-period, not label-free ML |
| **Al-Qadasi & Bin Waheed 2026** (ESS) | FNO, F1 0.96–0.99 on curated PSE/LSPE test sets | **Nearest competitor — read in full and cite.** Binary event/noise on a curated test set; no full-archive scan, no per-type recall |

Terrestrial-picker failure needs one explicit paragraph with citations, not a
results row: PhaseNet (30 s @ 100 Hz) and EQTransformer (60 s @ 100 Hz) are
trained on impulsive P/S arrivals; lunar events are emergent, scattered by the
megaregolith, and ring for tens of minutes to hours. Their 0.000 is a domain
mismatch, not a fair baseline — it *motivates* spectrogram segmentation.

## Track A — arXiv + ML workshop (target: weeks)

> **Deadline reality (today is 2026-07-25):** ICLR ML4RS 2026 closed 2026-02-06
> — fallback for the 2027 cycle only. **NeurIPS ML4PS 2026 is the live target,
> expected late Aug / early Sep** (2025 edition closed Aug 29; 2026 CFP not yet
> posted — watch `ml4physicalsciences.github.io`, set a mid-August reminder).
> That is **4–6 weeks**. ~4 pages ex-references, double-blind, non-archival, so
> it does not block the journal version. Track A must run in parallel with the
> B1 download, not after it — the download is wall-clock, Track A is attention.
> Post the arXiv preprint immediately to establish precedence on the
> label-free / type-stratified framing.

Claim: *injection-trained spectrogram segmentation matches supervised detection
under extreme label scarcity (45 labeled events), using zero labeled positives.*

- [ ] **A1. Screen the injection noise pool against Nakamura, retrain, keep both
      as an ablation.** `NoisePool.__init__` currently excludes only the 45
      Grade-A picks; add the same Nakamura screen `mine_unet_hardneg.py` already
      uses. Report screened vs unscreened as a result ("training-negative
      contamination costs X F1"), not just a silent fix.
      Re-tune on val, re-eval test, bump the freeze tag to v1.1, keep v1.0
      numbers for the ablation row. Retrain ≈ 30 min from `log.csv` timings.

      **Frame it as positive-unlabeled (PU) learning**, not as a bug fix: the
      noise pool is unlabeled data treated as negative, of which a measured
      8.4 % are hidden positives — the classic case-control contamination
      setting. Cite du Plessis / Niu / Sugiyama and Kiryo et al. (nnPU).
      Reviewers expect this ablation; presenting it proactively is a strength.
- [x] **A2. DONE — matched-filter baseline.** `scripts/matched_filter_baseline.py`,
      `results/matched_filter_lunar.json`. Max normalised cross-correlation
      over the **same 45 Grade-A train templates** the injection engine uses
      (identical labelled-information budget), MAD threshold, same dead-time
      rule / scorer / ±120 s tolerance. Template length AND k both selected on
      val.

      **Val-tuned F1 0.214** (P 0.333 / R 0.158 / MAE 56 s). To foreclose the
      "you under-tuned the baseline" objection, an **oracle allowed to pick
      its operating point on TEST reaches only 0.286** (P 0.261 / R 0.316).
      Both sit below every learned detector (SeisCNN 0.541, SpecUNet 0.440
      unscreened / 0.407 screened) and above STA/LTA (0.182).

      Structural reason, worth one paragraph in the paper: template matching
      wins on *repeating* sources, and the Grade-A benchmark events are
      heterogeneous. The same property is why it stays the right tool for
      deep-moonquake nests — which the archive scan confirms independently,
      since that is exactly the population our detector misses.

- [ ] **A2-old. Matched-filter / template-matching baseline on the 19-file test set.**
      The standard lunar catalog-extension method and the only baseline that
      genuinely threatens the result. PhaseNet/EQT at 0.000 is a strawman
      (terrestrial 100 Hz P/S pickers on 6.625 Hz emergent signals) — keep it,
      but stop leading with it.
- [x] **A3. DONE — 5 seeds, and it resolved TWO open questions.**
      `scripts/seed_variance_screened.py`, `results/seed_variance_unet_screened.json`,
      `results/seed_level_comparison.json`. Screened SpecUNet seeds 42/1/2/3/4:
      F1 0.4074 / 0.4906 / 0.3265 / 0.3111 / 0.3265 → mean **0.372 ± 0.076**.

      1. **The A1 question is answered: contamination costs nothing
         measurable.** Screened mean 0.372 vs unscreened 0.379, Welch
         **p = 0.923**. The single-seed 0.440 → 0.407 drop reported earlier was
         seed noise. Do not re-freeze; there is no effect to freeze.
      2. **"Statistically indistinguishable" is dead.** SeisCNN 0.498 ± 0.043
         vs SpecUNet 0.372 ± 0.076, Welch **p = 0.024** — the supervised
         detector is significantly better at the seed level. The old claim
         came from a single-seed paired bootstrap that resampled test files
         while holding the trained model fixed, so it never saw training
         stochasticity, which for injection training (fresh synthetic events
         every epoch) is first-order.

      **Revised headline claim:** injection-only training reaches **74.7 % of
      supervised mean F1 using zero labelled positive windows**. Not parity.

      Also surfaced: the frozen v1.0 SpecUNet figure (0.440) is the **maximum**
      of its three seeds, against a seed mean of 0.379. Every single-checkpoint
      number in the repo now carries that caveat.
- [ ] **A4. Scorer regression tests.** `evaluate.py` produces every headline
      number and has zero tests. Encode the verified greedy==Hungarian
      equivalence plus the known-suboptimal edge case
      (`score_trace([0,105],[100],120)` → MAE 100 s, optimal 5 s) so it cannot
      silently regress.
- [ ] **A5. Fix the ±300 s justification.** README claims Nakamura times are too
      coarse; own data says they match Grade-A arrivals at ±60 s, 100 %. Correct
      reason is *detector* arrival error (TP MAE 67.7 s, median 77 s). Same
      tolerance, sound reasoning. Permutation test is already tolerance-matched,
      so the p < 10⁻⁴ claim is unaffected.
- [ ] **A6. README drift:** window is 8192 samples (~1236 s) not 4096/~618 s
      (`README.md:212`); tolerance ±120 s not ±60 s (`README.md:214`); broken path
      `scriptseproduce_headline.py` (`README.md:200`).
- [ ] **A7. Soften the headline claim** to match the seed-variance addendum —
      "overlap, not tie" (seed std 0.086 vs delta −0.098), not "statistically
      indistinguishable".
- [ ] **A8. Commit stray files:** `PROJECT_WALKTHROUGH.md`,
      `results/unet_lunar_s{1,2}_to_lunar.json`.
- [ ] **A9. Draft 4–8 page workshop paper** (NeurIPS ML4PS / ICLR ML4Earth) +
      arXiv preprint. Reuse `make_paper_figures.py`.

## Track B — journal data paper (target: months)

Claim: *label-free detection recovers X % of the Apollo 12 long-period catalog and
flags N multi-station-confirmed events absent from it.*

- [x] **B1. Continuous archive ingest.** DONE — `planetseis/archive.py`,
      `scripts/fetch_apollo_archive.py`, `tests/test_archive.py` (10 tests).
      Scope confirmed: all four stations, MHZ, gap policy = interpolate short /
      mask long / carry validity. Full pull running in background; resumable
      (re-run the script, cached days are skipped).

      **XA verified live at IRIS.** S12 1969-11-19→1977-09-30 (2872 d),
      S14 1971-02-05→ (2429 d), S15 1971-07-31→ (2252 d), S16 1972-04-21→
      (1988 d) ≈ 9,541 station-days ≈ **21 GB** cached. S11 is 5 weeks — dropped.
      Archive nominal rate is **exactly 6.625 Hz = our target**, so `preprocess`
      skips resampling entirely. No drift correction needed for detection
      (Nunn's few-seconds-per-day timing drift is well inside the ±120 s
      tolerance) — state it as a limitation, don't fix it.

      Blockers found and fixed on day one, exactly what the gate was for:
      * **`-1` missing-sample flags.** Measured 1973-03-01 S12 MHZ: 464 flags
        (0.081%) in 5 runs, longest 396 samples / 60 s. Valid centreline ~495
        counts, signal range ~15 counts, so each flag is a **33x-dynamic-range
        spike**; a 60 s run is an event-shaped rectangular pulse. Through the
        existing `preprocess` they inflate the day's std **7.7x** and peak
        **48.6x**. `nan_to_num` does not catch them. Now masked before filtering.
      * **Zero-filled gaps** (`preprocessing.load_trace:34`,
        `merge(fill_value=0)`) manufacture a step edge at every dropout —
        harmless on pre-cleaned snippets, not across years. Archive path now
        bridges gaps and tracks validity separately.
      * **Gain state** is the location code: '00' peaked, '01' flat, 5.6x
        sensitivity. One code per day file, never merged.
      * **Own bug, caught before bulk pull:** clipped day slices were labelled
        `day - PAD_SEC` rather than their actual first sample, offsetting a
        station's first day by hours. Start time now derived from the real
        slice origin. This class of error would have silently poisoned every
        Nakamura match and multi-station moveout.

      **Pull complete: 7,777 day files, 16.2 GB, 7,850 valid station-days.**
      Cache lives at `C:\planetseis_archive` (set `PLANETSEIS_ARCHIVE`) —
      deliberately OUTSIDE OneDrive; `data/cache` is a live Files-On-Demand
      folder and 16 GB of incompressible float32 would have triggered exactly
      the sync-induced RAM/IO stalls this project has already lost hours to.

      | Station | MHZ days | Expected | Coverage | valid_frac |
      |---|---|---|---|---|
      | S12 | 2,844 | 2,872 | 99.0 % | 0.969 |
      | S15 | 2,234 | 2,252 | 99.2 % | 0.972 |
      | S16 | 1,983 | 1,988 | 99.7 % | 0.970 |
      | S14 | 716 | 2,429 | **29.5 %** | 0.963 |

      **S14 MHZ has a genuine ~4.5-year hole in the IRIS holding (1972-03 →
      1976-11)** — zero fetch failures; mid-gap, MH1/MH2/SHZ/ATT all return
      data while MHZ alone returns HTTP 204. The vertical component is simply
      absent. This is the "narrow the scope" checkpoint: **S12/S15/S16 carry
      the multi-station claim, S14 is a partial fourth**, stated plainly.
      Using S14's horizontals to fill the gap would introduce an unvalidated
      component domain shift — don't, unless a reviewer asks.

- [ ] **B2a. Stratify B3 by gain state as well as event type.** The pull
      revealed ~28 % of day files are FLAT mode (`loc='01'`): S12 2057/787
      peaked/flat, S15 1598/636, S16 1348/635, S14 716/0. Flat-mode sensitivity
      is 1/5.6 of peaked. Per-window z-scoring absorbs a constant gain factor,
      so the model is largely gain-invariant — but the signal sits 5.6x closer
      to the digitiser LSB, so low-SNR recall is genuinely worse on flat days.
      That is a physical sensitivity difference, not a normalisation artifact,
      and reporting recall split by gain state pre-empts the obvious question.
      `loc` is already stored per day file, so this is nearly free.
- [ ] **B2. Full-archive scan.** ~408 K windows ≈ 7 min GPU inference; I/O and
      preprocessing are the real cost, not the network.
- [ ] **B3. Score against the 5,358 S12 Nakamura events:** recall with n in the
      thousands, **stratified by event type** (M / C / A / Z + deep-nest) — a
      label-free detector's type-stratified recall is a result nobody has
      reported — plus false alarms per day, currently not even measurable.
      Report **false alarms per station-day** (the standard continuous-detection
      convention, so numbers compare directly to the HMM/CNN prior work), using
      the validity mask as the denominator — that is what it exists for. Report
      a FAR-vs-recall curve, not a single operating point, and label
      single-station vs network-coincidence FAR separately (coincidence
      collapses FAR dramatically; conflating them is a reviewer trap).
      **This is the gate:** if FAR/station-day is unacceptable, stop before B4.
- [x] **B3 RUN — GATE DID NOT PASS.** See `results/archive_scan/B3_FINDINGS.md`.
      6,850 valid station-days scanned. Best case (S12, the training station):
      either recall 0.525 at **15.4 unmatched detections/station-day**, or a
      tolerable 0.19/day at recall 0.071. The benchmark-tuned point recovers
      **0.8 %** of the catalog. Shorter coda/duration gates make it worse, not
      better. Not competitive with Knapmeyer-Endrun & Hammer 2015 (>95 %
      impacts, ~70 % deep moonquakes, single-station HMM).

      Mechanism is coherent, not a bug: the injection engine synthesises from
      **45 large Grade-A S12 templates**, while 56 % of the catalog is small
      repetitive deep moonquakes that matched filtering finds *because* they
      repeat. Recall also collapses off the training station (deep moonquakes
      0.52 → 0.21 → 0.14 across S12 → S15 → S16), which qualifies the earlier
      95/96 cross-station claim — that was measured on files curated around a
      known event.

      Kept positives: **shallow moonquakes, the rarest and most significant
      class (28 in the whole catalog), have the highest recall of any class
      (0.60 S15, 0.52 S16)** from a detector that never saw a labeled positive.

      **Catalog parse was wrong until today.** Column 76 is not the event
      type — `M`-coded and blank-coded events share 247 of 250 nests, and
      column-76 `A` occurs 1,359 times where the literature records 9
      artificial impacts. Deep moonquakes are identified by the nest field
      (cols 81-84): 7,318 events / 320 nests / 56 %, matching the published
      figures, and `H` = 28 shallow, matching exactly. Any earlier figure
      naming "artificial impact" recall is invalid.

- [ ] **B4. Multi-station coincidence — DO NOT START on this detector.**
      At 15 unmatched detections per station-day, coincidence would be
      dominated by chance, not discoveries. Blocked behind a training-set fix
      (see B7) or dropped.
- [x] **B7 DECIDED: stop Track B as a catalog-extension paper.**

      The tempting rescue — rebuild templates from nest-labelled deep
      moonquakes (320 nests, 7,318 events) and rescan — **should not be done**,
      for a structural reason rather than an effort one:

      * Using 7,318 catalogue-labelled events as templates **abandons the
        zero-label premise**, which is the entire novelty. It becomes a
        supervised method with abundant labels.
      * Nest templates + correlation **is matched filtering**, and Bulow et al.
        (2005/2007) already did exactly that, adding 500+ deep moonquakes by
        stacking. Our own A2 result shows matched filtering is weak on
        heterogeneous events but strong where sources *repeat* — which is
        precisely the deep-moonquake regime.

      So the niche closes from both sides: label-free is too weak, and
      nest-labelled lands on matched filtering's home turf where prior work
      already wins. There is no gap for this method in the lunar
      deep-moonquake population, and that is worth stating rather than
      spending months discovering.

      **Decision: fold the archive negative into Track A's limitations (done)
      and ship.** Optional follow-on, NOT a paper on its own yet: shallow
      moonquakes recovered 39/61 label-free across three stations — real, but
      n=61 total, so it needs a co-author who cares about that class before it
      justifies a venue.

- [ ] **B7-old (superseded). Decide the reframe.**
      Options in `B3_FINDINGS.md`: (1) publish as a negative/limits paper,
      which the type-stratified numbers and the corrected catalog parse
      support honestly; (2) rebuild the template bank from nest-labelled deep
      moonquakes (320 nests, 7,318 events available) and rescan — the
      experiment most likely to change the answer; (3) drop the archive claim
      and ship Track A alone. Track A is unaffected either way.
- [ ] **B4-old. Multi-station coincidence.** Run S14/S15/S16; keep candidates seen at
      ≥2 stations within physically plausible moveout. Two independent stations +
      moveout converts "false positive" into defensible new event. Cross-check
      against Nakamura; candidates in neither = the discovery list.
      Quantify Apollo timing error before trusting moveout (Nunn: software-clock
      fallback can exceed a minute; drift of a few seconds per day, per station).
- [ ] **B5. Matched-filter baseline over the archive.** Use EQcorrscan
      (frequency-domain, `Party`/`Family`/`lag_calc`) or Fast Matched Filter
      (Beaucé et al. 2018, GPU) rather than rolling our own. Templates from the
      known deep-moonquake nests (A1 etc.). Threshold at ~8-12x MAD of the
      correlation function, per-station min correlation ~0.7. **The argument to
      win:** we (a) match or beat matched filtering on repeating deep events AND
      (b) recover non-repeating events — impacts, shallow moonquakes — that
      template matching structurally cannot. This is the single most likely
      reviewer objection.
- [ ] **B6. Derived candidate catalog → Zenodo DOI**, versioned to match the
      freeze tag. Target **AGU Earth and Space Science** (data/method fit, has a
      "Models and Machine Learning: Applications" section) or **SRL Data Mine**
      (purpose-built citable catalog reference, ≤6000 words / 10 figures,
      requires a DOI'd public dataset — XA already satisfies this). JGR:
      Machine Learning and Computation if foregrounding method over catalog.
      PDS deposit only if reviewers push; Zenodo suffices for acceptance.

---

## Risks — name them now, report them either way

- **A1 may lower benchmark F1.** If it does, that is the honest number and the
  ablation is still publishable. Do not tune it away.
- **B3 may show a much worse false-alarm rate per day than curated snippets
  suggest.** This is the real test of the catalog-extension claim and it could
  falsify it. The repo's honest-negative culture is well suited to reporting that
  outcome; plan for it rather than discovering it late.
- **Apollo timing errors** are a known archive-wide problem; Nakamura times may be
  off by more than the ±300 s budget in places. Quantify before trusting B4
  moveout.
- **Track B depends entirely on B1.** If the continuous archive turns out to be
  unavailable or unusable, Track B dies and Track A is the whole deliverable.
  Verify B1 first, before any other Track B work.

## Citation hygiene — get these exactly right

- **Nakamura count varies by vintage:** ~12,558 (1981) vs **13,058 (2008 final
  revision — the number we use, `levent.1008.dat`)**. Cite the UTIG Technical
  Report 18 handle (`hdl.handle.net/2152/65671`) plus Nunn et al. 2020/2022
  DOIs; the catalog file has no standalone DOI. State version and vintage
  explicitly. Deep moonquakes are >55 % of identified events.
- **Nest counts also vary by definition** (108 located 1981 → 77 after 2003
  merging → ~165 located → ~300 clustered). Say which.
- **The 12,085 figure is LSPE (Apollo 17 short-period)**, not a PSE scan —
  attribute to Civilini 2021/2023 and verify against the 2023 paper directly,
  since secondary sources conflate the two catalogs.
- **Waveform archive citation:** Nunn, Nakamura, Kedar & Panning 2022, PSJ
  3:219, DOI 10.3847/PSJ/ac87af; data DOI 10.7914/SN/XA_1969. Pull the exact
  per-station Table 1 recovery fractions for our four stations rather than
  quoting the aggregate ~0.3 % Barker-code rejection.
- Confirm the ML4PS 2026 deadline before committing the timeline — the date
  above is extrapolated from the stable 2019–2025 pattern, not posted.

## Do NOT do

- Chase the Mars result (n=30, and the MQS team owns that field).
- Add architectures — the capacity ablation already shows >120 K params hurts.
- Revisit SSL — already a documented negative result.

---

## Review

**Session 2026-07-25.**

Landed:
- **B1 complete** — archive ingest built, tested (10 tests), 16.2 GB /
  7,850 valid station-days pulled and relocated off OneDrive. Three archive
  pathologies handled, one self-inflicted timing bug caught pre-pull. S14
  scope narrowed on evidence.
- **A4** — `tests/test_evaluate.py`, 15 tests on the scorer that produces every
  headline number. Pins the greedy-vs-optimal equivalence (verified identical
  on both test sets) so relaxing dead-time suppression surfaces in CI rather
  than in a results table. Suite: 19 → 44.
- **A5/A6/A7** — README corrected: window 8192/±120 s (was 4096/±60 s), the
  `scripts\reproduce_headline.py` path (a stray 0x0D byte, not a typo), the
  ±300 s tolerance now justified by detector arrival error rather than the
  false "Nakamura is coarse" claim, and the headline softened from
  "statistically indistinguishable" to "same performance band, not resolved
  by n=19".
- **A1 in progress** — `scripts/build_nakamura_screen.py` (158 event times
  across 56 files), `NoisePool(screen=...)`, `train_unet.py
  --screen-nakamura`. Screen verified to remove **2,009,601 window starts =
  8.38 %** of the pool, all 45 files retained — matching the audit figure
  exactly. Retrain running as `unet_lunar_screened`, hyperparameters copied
  from the original checkpoint's stored config so the ablation is clean.
- **B2/B3 scripts written** — `scan_archive.py` (pad de-duplication, validity
  gating) and `score_archive_vs_nakamura.py` (per-station denominators,
  observability gating, type + gain-state stratification, candidate export).
  Not yet run: concurrent CUDA has crashed this machine, so the scan waits
  for training to finish.

Deliberately not done: no commits — the harness rule is to commit only when
asked. Everything is on disk and `git status` shows the full change set.

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

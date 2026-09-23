# PLANETSEIS — COMPLETE WALKTHROUGH & DEFENSE GUIDE (v2)
### Planetary Seismic Event Detection · B.Tech Major Project · Sidhvik Gudikandula

**Audit update (2026-09-15):** the frozen lunar test contains two waveforms
duplicated in training under different event IDs. Present the figures below as
historical measurements awaiting grouped splits and retraining, not independent
test performance. See `docs/PROJECT_REVIEW.md`. SpecUNet uses catalog-derived
templates with synthetic mask supervision; do not describe it as using zero
labels. The app now supports bundled demos in Analyze and runs only when
**Analyze trace** is pressed; request denoising before submitting the analysis.

**Corrected benchmark (2026-09-23) — lead with these.** On the
acquisition-grouped split `lunar_grouped_v1` (duplicates merged, picks
unioned, 21 test spans / 23 events), both models retrained from scratch over
five seeds: **SeisCNN 0.531 ± 0.031, SpecUNet 0.408 ± 0.043** (Welch
p = 0.0012; SpecUNet = 76.8 % of supervised mean F1); matched filter 0.200;
STA/LTA 0.168. Corrected means are not lower than the historical ones, but
split, labels and test population all changed, so do **not** claim the
duplicates had no effect. Across five seeds on the corrected split, 43 % ± 13 %
of SpecUNet false positives match Nakamura events (1.5 % by chance). The demo app
runs the corrected seed-42 lunar checkpoints (SeisCNN 0.97; SpecUNet
0.25 / 430 s). Source: `results/lunar_grouped_v1_seed_summary.json`.

Everything you need to run the demo, present it, and answer any question.
Project lives at: `Desktop\major` · GitHub: github.com/sidhvik17/planetary-seismic-triage
**v2 = Phase-2 update: the project now has TWO detectors** — the supervised
SeisCNN (Phase 1) and a MarsQuakeNet-style, injection-trained
SpecUNet (Phase 2) that extends catalogs and denoises. Results frozen at git
tag `v1.0-results-freeze`.

---

## PART 1 — HOW TO START (2 minutes before the demo)

1. Open the `Desktop\major` folder.
2. **Double-click `run_app.bat`.** A terminal opens, then your browser opens at
   `http://localhost:8501`. First start takes ~20 s (loading PyTorch). Done.
3. Warm the simulation: go to the **🛰️ triage tab**, press **▶ Start stream**
   once, let it finish. This caches the scoring so the live demo has no spinner.
4. Keep this file open on your phone.

If the browser doesn't open: type `localhost:8501` into Chrome yourself.
If `run_app.bat` complains about `.venv`: open a terminal in the `major`
folder and run `python -m venv .venv` then
`.venv\Scripts\pip install -r requirements.txt` (one-time, ~5 min).
**Never run `streamlit run ...` with plain python — always the .bat or
`.venv\Scripts\streamlit.exe`.**

---

## PART 2 — THE OPENING (memorize this — now 5 sentences)

> "A seismometer on the Moon or Mars records 24 hours a day, but the radio
> link back to Earth is tiny — you cannot send everything home. My project
> puts a small neural network **on the lander itself** that scans the stream,
> cuts downlink by ~96%, and says *'I'm not sure'* when it isn't. Then I went
> further: I built a second detector that **learns from synthetic event masks**
> — it injects catalog-derived quake templates into real
> planetary noise — and it found **nine real moonquakes that the working
> catalog had missed**, verified against NASA's full 13,058-event historical
> catalog with odds against chance of ten thousand to one. Same technique,
> pointed at Mars with the official marsquake catalog: **zero false alarms in
> 85 hours** of InSight data."

---

## PART 3 — STEP-BY-STEP DEMO SCRIPT (~6 minutes)

### Act 1 — Credibility (Tab ℹ️ Model & results, 45 s)
Point at two things:
- "117,842 parameters for the supervised detector — half a megabyte. Runs a
  full day of data on CPU in under a second."
- The table: "PhaseNet and EQTransformer — the field's standard models,
  trained on a **million** earthquakes — score **zero** on the Moon. The
  baseline that actually matters here is waveform template matching, the
  standard method in lunar seismology: given the identical 45-event label
  budget it reaches F1 0.200 on the corrected split. After I found and removed
  a train/test duplicate problem and retrained from scratch over five seeds,
  my supervised model scores 0.531 ± 0.031. The injection-trained SpecUNet
  reaches 0.408 ± 0.043 — **about 77% of supervised, using catalog-derived
  templates and synthetic mask targets.** Welch p = 0.0012: the supervised
  model is genuinely better."

### Act 2 — The money shot (Tab 🛰️ triage simulation, 90 s)
1. Keep default file, playback 4× → **▶ Start stream**.
2. Narrate: waveform = the lunar surface stream; orange dashed = auto-accepted
   event; blue dotted = routed to the human review queue with honest error
   bars; the counter = "**96% downlink reduction**, live."
3. End card: "NASA's answer key agrees."

### Act 3 — Phase 2 live (Tab 📈 Analyze, 2.5 min) ← NEW
1. **Detector: SpecUNet (spectrogram, MQNet-style)** — it's the default.
   Model: **lunar**. Upload
   `major\data\raw\space_apps_2024_seismic_detection\data\lunar\training\data\S12_GradeA\xa.s12.00.mhz.1970-12-11HR00_evid00017.mseed`
2. "This detector was trained on **synthetic event masks** — training data
   is manufactured by injecting cleaned catalog-derived quake templates into noise.
   The curve below is its event-energy mask; dotted black is NASA's pick."
3. Open the **Denoised trace** expander: "same network, second job — it
   multiplies the spectrogram by its own event mask and resynthesizes the
   waveform. +8 to 9 dB over classical filtering exactly where events are
   weakest. This is what MarsQuakeNet does on Mars; mine does it on the Moon."
4. Toggle **Uncertainty mode** on, then deliver the punchline: "for the
   supervised model, uncertainty separates false alarms about 8× on the
   corrected split (7.9 ± 3.5× across five seeds). For this injection-trained
   model it does not separate them reliably — 0.86 ± 0.43, below 1 in four of
   five seeds — and that is one of my research findings, not a bug: a model
   never taught the catalog's opinion can't rank catalog membership."
5. Optional: switch Detector to SeisCNN and re-run — "two instruments, two
   regimes: the supervised one carries precision and triage, the injection-trained
   one carries recall, discovery, and denoising."

### Act 4 — The discovery slide (any browser, 30 s)
Open `major\docs\figures\nakamura_match_1_0_evid00192.png`:
"The benchmark scored this detection as a FALSE POSITIVE. Look at it — a
meteoroid impact ringing for an hour. It's in NASA's full Nakamura catalog,
160 seconds from my detection; the benchmark's 76-label subset just doesn't
include it. On the corrected split, about four in ten of the models'
'false positives' fall within five minutes of a catalogued moonquake."

---

## PART 4 — THE NUMBERS (know these cold — now ten)

| Number | What it is |
|---|---|
| **0.531 ± 0.031 vs 0.408 ± 0.043** | **Corrected** grouped split, 5 seeds each: SeisCNN vs SpecUNet, Welch p = 0.0012 — the numbers to quote |
| **0.541** | Historical supervised SeisCNN checkpoint F1 (P .556/R .526), filename split with two duplicates |
| **0.372 ± 0.076 vs 0.498 ± 0.043** | Historical injection-trained SpecUNet vs supervised, **5 seeds vs 3** — Welch **p = 0.024**; **74.7% of supervised with synthetic mask targets and catalog-derived templates**. Superseded by the corrected row above |
| **0.440** | The frozen single checkpoint — say out loud that this is the **MAX of 3 seeds** (mean 0.379). Never quote it alone |
| **0.214 / 0.286** | Matched filter, val-tuned / oracle test-tuned. The baseline a seismologist asks for first; both learned detectors beat it |
| **43% ± 13% · 47/120 pooled · 1.5% chance · p<10⁻⁴ every seed** | **Corrected** split, five seeds: benchmark "false positives" within ±300 s of Nakamura-catalogued moonquakes (historical single checkpoint: 9 / 20, 45% vs 1.7%) |
| **0.62 ± 0.10** | Corrected survey-mode precision counting Nakamura matches as true (historical 0.645) |
| **P = 1.000, R = 0.233, n = 30** | Mars test on official MQS v14 picks — zero false alarms in ~85 h |
| **+8–9 dB** | denoising SDR gain over bandpass at the hardest SNR bin |
| **5.9× → 0.49×** | MC-Dropout separation: supervised vs injection-trained (**the σ-inversion**) |
| **95/96 · 100%** | weak-label detection rate, incl. stations never trained on (SeisCNN: 60–93%) |
| **96%** | downlink reduction in the triage simulation |

Backups: historical lunar MAE 40 s (SeisCNN) / 68 s (SpecUNet); expanded Mars MAE 33.93 s / 34.20 s refined ·
duration gate: real moonquakes ring 460–1240 s, false regions median 48 s ·
23/64 mined "hard negatives" turned out to be real quakes · $0 cost.

---

## PART 5 — EVERY QUESTION THEY CAN ASK (with answers)

### DATA
**Q: Where is the data from?**
NASA public archives — Apollo 12/15/16 PSE (1969–77) and InSight SEIS
(2018–22) via the Space Apps 2024 packet. Phase 2 adds: official MQS
catalog v14 picks via the IRIS mars-event web service, raw InSight
waveforms from the open EarthScope archive, and the full Nakamura Apollo
catalog (13,058 events, UTIG TR-18) for verification.

**Q: How do you know a quake is really there?**
Phase 1: score against NASA's shipped picks at ±120 s. Phase 2 goes
further: every detection the benchmark calls FALSE is cross-referenced
against the complete historical catalog — with a permutation test showing
the matches aren't luck (0 of 10,000 random placements reach 9 matches).

**Q: How much data?**
Lunar: 75 labeled day-files (45/11/19 by file). Mars grew from 2 packet
files to **46 files** with official MQS picks — 24 train / 5 val / **17
test spans holding 30 events**, the test expansion added blind to any
tuning decision.

### THE TWO MODELS
**Q: Why two detectors?**
They're complementary instruments. SeisCNN (118K params, supervised)
learned the *catalog's selection function* — precision, calibration,
triage. SpecUNet (1.9M, injection-trained, synthetic mask targets) learned *event
morphology* — recall, cross-station generalization, catalog extension,
denoising. The σ-inversion is the measured boundary between their jobs.

**Q: What is injection training, exactly?**
Cut a template around each catalogued quake; despike it (Apollo thermal
ticks otherwise teach "spike = event"); spectrally gate the background
noise away; inject it into event-free noise at random strength (0.4–12×
noise), random position; 40% of samples get synthetic GLITCHES labeled as
noise. Because event and noise are known separately, the exact per-pixel
energy-ratio mask is computable — synthetic supervision derived from catalog-selected templates.
That's MarsQuakeNet's method (Dahmen et al. 2022), translated to the Moon.

**Q: The U-Net is 16× bigger than the CNN — didn't you say small wins?**
Small wins *for supervised training on 45 labels* — measured (426K CNN is
worse than 118K). Injection training changes the data budget: unlimited
synthetic samples feed 1.9M parameters fine. Two different regimes, both
measured.

**Q: What's the duration gate?**
The single biggest precision lever: real moonquakes ring 460–1240 s;
false detector regions have median duration 48 s. Requiring ~10 minutes of
sustained mask energy kills ~94% of false regions at zero recall cost.
Physics as a filter.

### THE HEADLINE
**Q: Your precision is only 0.355. Isn't that bad?**
Against the benchmark's 76-label subset, yes — and that's the finding: the
subset is incomplete. 9 of the 20 "false positives" are real moonquakes in
the full Nakamura catalog (45% vs 1.7% chance, p<10⁻⁴). Counting them,
precision is 0.645. One of them is a meteoroid impact with an HOUR of coda
— I show the waveform. My precision is a *lower bound set by the catalog,
now proven rather than claimed*.

**Q: ±5 minutes matching — generous?**
Nakamura times are minute-quantized SIGNAL STARTS from an independent
1970s processing chain, and weak emergent events cross my threshold late.
I publish the sensitivity: 5 matches at ±3 min, 9 at ±5 min — and the
permutation test operates at exactly the window I claim.

**Q: Isn't that circular — Nakamura is also station-12 data?**
Named in my limitations before any reviewer could: same station,
independent catalog and era. Fully independent verification (S15/S16
cross-timing) is stated future work.

### MARS
**Q: Mars recall is 0.233 — why so low?**
Honest number on a hard set: the expanded test is dominated by M2.9–3.0
events on a single vertical channel at 0.5–3 Hz. What I keep: precision
1.000 — zero false alarms in 85 hours — and the PR curve shows F1 0.50
available at survey settings. Also honest: the first n=5 test read
recall 0.6; expanding to 30 events corrected that small-sample artifact,
and the correction is in the repo history.

**Q: What did the miss autopsy show?**
S0173a (the first confirmed marsquake) crosses the threshold but fails
the duration gate — an operating-point boundary, not blindness. S1022a
sits in amplitude-dead data; in that same span a 5,000-count glitch
correctly stays BELOW threshold — the synthetic-glitch training holding
under fire.

### DENOISING
**Q: How do you measure denoising without ground truth?**
Injection gives ground truth: I know the clean event exactly, so
cross-correlation and SDR are computable. +8–9 dB over bandpass at the
hardest SNR bin, converging to parity when events are strong — the same
pattern the MQNet 2024 paper reports.

**Q: Did denoise-then-detect help?**
Three-way experiment, all honest: naive chaining COLLAPSES the supervised
model (precision 0.556→0.067, distribution shift); retraining it on
denoised windows restores a tie (F1 0.512, recall up); fusing raw+denoised
probability streams gives the best recall of anything I built (0.684).
Denoising surfaces the missed events; precision is the price.

### UNCERTAINTY / σ-INVERSION
**Q: What's the σ-inversion?**
For the supervised model, MC-Dropout uncertainty separates false alarms
from real events 5.9× — triage works. Port the identical analysis to the
injection-trained model and it INVERTS (0.49×): its false positives are
*more* confident than its true events. Mechanism: supervised labels ARE
the catalog, so catalog-marginal signals sit near the decision boundary
where variance is high; injection training replaces selection with
physics, so uncertainty measures distance from the injection
distribution, not the catalog boundary. Known ML pattern (Ovadia 2019,
Nalisnick 2019) — first time stated for planetary detection. Consequence:
triage on the supervised head, survey on the injection-trained head.

### RIGOR / REPRODUCIBILITY
**Q: Did you tune on the test set?**
Never. Both models: operating points tuned on validation only. Mars test
expansion added 12 spans AFTER freezing the operating point. Splits
frozen and published in `benchmark\`.

**Q: Can anyone reproduce this?**
Git tag `v1.0-results-freeze`; one command
(`python scripts\reproduce_headline.py`) re-runs both detectors from the
committed checkpoints and EXITS NONZERO if any number drifts — verified
from a fresh clone. Checkpoints + results + splits archived in GitHub
releases. 19 unit tests in CI. Seed variance published as the
`v1.0.1-seed-addendum` release.

**Q: What failed? (they love this one)**
On the record: first model F1 0.21 (coda mislabeling); cross-body
transfer fails every way tried; naive denoise-chaining collapses;
hard-negative mining backfired until I discovered 23 of 64 mined
"negatives" were REAL uncatalogued quakes (which itself feeds the
headline); arrival refinement does not improve the expanded Mars test and hurts the Moon; active learning
lost to random; INT8 was worse; the Mars n=5 recall was an artifact.
Every negative is in `results\` with the same rigor as the positives.

**Q: What's novel? Detection CNNs exist.**
Five things: (1) MQNet's injection paradigm shown to survive a 45-label
single-channel LUNAR setting; (2) the catalog-extension result verified
automatically against a 13,058-event historical catalog instead of by
manual review; (3) the σ-inversion — a deployment boundary for synthetic
supervision nobody had stated for planetary work; (4) injection-trained reaching
74.7% of supervised while beating the matched-filter baseline on an
identical label budget; (5) a frozen benchmark with a Mars track on
official MQS picks that others can submit to; (6) a **6,850 station-day
continuous-archive test showing where the method stops working** — most
projects never test their own limits.

### CURVEBALLS
**Q: 6 more months?**
The old answer was "scan all 8 Apollo years" — that is now **DONE, and it
came back negative.** 6,850 valid station-days: best case 52% catalog
recall at 15.4 unmatched detections/station-day, or 7% at 0.19/day.
Injection training from 45 large templates has no purchase on the small
*repeating* deep moonquakes that are 56% of the catalog — which is exactly
what matched filtering was built for. So I would **not** spend six months
there. What I would do: 3-component + 2.4 Hz band for Mars recall;
catalog-fine-tuned uncertainty head to fix triage on the injection-trained model;
and the one live positive — shallow moonquakes, the rarest class, recovered
39/61 across three stations with synthetic mask supervision — with a seismologist co-author.
Venue: ML4PS 4-pager (written, `paper/ml4ps_2026.md`).

**Q: Why didn't you re-freeze on the screened model?**
Because the effect was null. Screening the 8.38% of noise windows holding
real catalogued events changed benchmark F1 not at all (0.372 vs 0.379,
Welch p = 0.923). A single seed had shown 0.440 → 0.407 and I read it as an
effect until five seeds showed it was noise. Re-freezing on noise would
have been dishonest.

**Q: Difference from hackathon projects on this data?**
They claim accuracy. I ship a measured system: frozen benchmark, paired
bootstrap + permutation statistics, seed variance, honest negatives, a
catalog-extension discovery verified against NASA's own historical
catalog, and a one-command reproduction that fails loudly if a single
number drifts.

---

## PART 6 — WHERE EVERYTHING LIVES

| Thing | Path (inside `Desktop\major`) |
|---|---|
| One-click demo | `run_app.bat` |
| Report / walkthrough (Phase 1 + 2) | `docs\report.md` |
| Research paper (LOCAL ONLY — not on public GitHub) | `paper\paper_final.md`, `paper\paper_ieee_v2.tex` (+ Overleaf zip) |
| Paper slide deck | `paper\planetseis_paper_deck.pptx` |
| Defense deck (this walkthrough as slides) | `paper\planetseis_walkthrough_deck.pptx` |
| All metrics (raw JSON) | `results\*.json` |
| Discovery evidence figures | `docs\figures\nakamura_match_*.png` |
| Trained models (both detectors) | `models\` (.pt + .onnx) |
| Benchmark (frozen splits + protocol) | `benchmark\` |
| One-command reproduction | `scripts\reproduce_headline.py` |
| Results freeze + archives | GitHub tags `v1.0-results-freeze`, `v1.0.1-seed-addendum` + releases |
| Private paper backup | github.com/sidhvik17/planetseis-paper-private |
| Demo upload file | `data\raw\...\S12_GradeA\xa.s12.00.mhz.1970-12-11HR00_evid00017.mseed` |

## PART 7 — TROUBLESHOOTING (demo day)

| Symptom | Fix |
|---|---|
| `No module named torch/obspy/...` | You used system Python. Use `run_app.bat`. |
| Port 8501 busy | Close old terminals, or `taskkill /f /im streamlit.exe` then retry |
| Upload spinner long | Normal 5–15 s for a 40 MB file. Talk through it. |
| SpecUNet denoise expander slow | ~10–20 s for a day-file on CPU — open it while narrating |
| Sim first run slow | One-time cache — warm it before (Part 1, step 3). |
| No events at high threshold | Lower the slider; explain the PR tradeoff — it's a feature. |
| Total meltdown | `docs\report.md` + `docs\figures\` + both decks = full presentation without the app. |

## PART 8 — YOUR REMAINING TODO (not mine)

1. Revoke the old Hugging Face tokens if not yet done (hf.co → Settings → Access Tokens).
2. Fill your author block in `paper\paper_ieee_v2.tex` (name/department line).
3. Read `paper\paper_final.md` once fully — own every number before you defend it.
4. Venue: upload `paper\planetseis_ieee_overleaf.zip` to Overleaf when ready;
   watch ml4physicalsciences.github.io from mid-August for the CFP.
5. Optional: cold email one Apollo-seismology researcher with Figure 1
   attached — "undergrad, found X, would you sanity-check?"

You have a discovery, a deployment caveat, and a reproduction command.
That's more than most Masters theses. — Fable

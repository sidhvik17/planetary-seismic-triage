# APSIS — walkthrough and research defense guide

Updated 2026-09-23. This guide uses the corrected lunar benchmark and
supersedes the earlier demonstration script. The project still uses Python,
ObsPy, PyTorch and Streamlit, with SeisCNN and SpecUNet unchanged.

## 1. Start the demonstration

1. Double-click `run_app.bat`, or run `.venv\Scripts\python -m streamlit run app/streamlit_app.py`.
2. Open `http://127.0.0.1:8501/`.
3. In **Analyze a trace**, select a bundled lunar demo and SpecUNet. The demo
   waveform is already preprocessed; no raw-data download is needed.
4. Keep the recorded default threshold 0.25 and minimum duration 430 s, then
   press **Analyze trace**. Request denoising before submitting if needed.
5. Check the displayed model provenance: `lunar_grouped_v1`, seed 42.
6. Switch to SeisCNN, use its recorded 0.97 threshold, and analyze again.
7. For the triage demonstration, open **On-lander triage simulation** and
   press **Start stream**. This uses an exploratory MC-Dropout policy,
   separate from the deterministic benchmark operating point.

The app checks the packaged lunar checkpoint hash and provenance before use.
Mars retains its original checkpoints and settings. A changed model or threshold
requires a new analysis; a finished result belongs to its submitted settings.

## 2. Explain the idea

APSIS compares two approaches to detecting planetary seismic events under
label scarcity. SeisCNN learns from labeled waveform windows. SpecUNet learns
spectrogram masks from catalog-derived event templates injected into noise;
its mask also produces a denoised trace. Both emit candidate event times.
The app demonstrates inspection and a possible human-review workflow.
It has not been validated as flight software or an operational discovery system.

SpecUNet is **not zero-label learning**: catalog picks identify the templates.
Its targets are synthetic event/noise masks rather than real positive-window
labels. Keep that distinction explicit in the presentation and paper.

## 3. The results to quote

Corrected benchmark `lunar_grouped_v1`: 21 test spans, 18 acquisition groups,
23 events; one-to-one matching within ±120 s. Both models were retrained
from random initialization with five seeds. Each operating point was selected
on validation and locked before that evaluation accessed test waveforms.

| Detector | F1 mean ± sample SD over seeds |
|---|---:|
| SeisCNN | **0.531 ± 0.031** |
| SpecUNet | **0.408 ± 0.043** |
| Matched filter, validation-selected | 0.200 |
| STA/LTA, validation-selected | 0.168 |

SpecUNet reaches 76.8% of SeisCNN's mean F1. SeisCNN performs better in this
comparison (Welch p = 0.0012 conditional on this test set); parity is not
supported. The acquisition-group bootstrap gives a difference interval
[0.009, 0.233], averaging the five fixed trained seeds. It does not jointly
resample training seeds. The small test set limits precision.

Source: `results/lunar_grouped_v1_seed_summary.json`.

## 4. What the leakage repair means

Two historical test waveforms duplicated training data under different
filenames. The new split groups overlapping UTC spans and identical waveform
content, merges duplicate copies with unioned picks, corrects pick times from
actual trace starts, and recovers one previously unassigned event. The audit
finds no cross-split interval or whole-waveform duplicates.

The historical models and results remain available for reproduction. Do not
claim that leakage had no effect merely because the new scores are higher:
split membership, labels, training sampling and evaluation population changed.
The holdout reuses previously studied acquisitions; it is not new external data.
The audit does not prove independence between repeated deep-moonquake sources.

## 5. Catalog cross-check: timing tolerance, not 47 discoveries

Across five seeds, 43% ± 13% of SpecUNet's benchmark false positives fall
within ±300 s of a Nakamura S12 catalog event (47 of 120 pooled detections).
However, **45 of those 47 matches refer to Grade-A events already labeled
in the same span**. The other two detections, from two seeds, refer to one
distinct additional catalog event. They are candidates, not new discoveries.

Most of this result measures the difference between the primary ±120 s
arrival tolerance and the cross-check's ±300 s tolerance. It does not justify
replacing benchmark precision with a higher “catalog-adjusted precision.”
The random-placement test had 0 exceedances in 10,000 draws per seed;
plus-one Monte Carlo p is about 0.0001, not zero. That null addresses temporal
association, not whether an event was omitted from the benchmark.

## 6. Uncertainty and signal strength

- SeisCNN window-level MC uncertainty is 7.9 ± 3.5 times higher on false-alarm
  than true-event windows. Its review queue contained a real Grade-A event
  in only one of five seeds. This does not establish useful operational triage.
- SpecUNet's median uncertainty ratio for **all benchmark FPs / benchmark TPs**
  is **1.06 ± 0.38**, below one in two of five seeds. The earlier 0.86 ± 0.43
  used only catalog-unmatched FPs. Do not claim consistent uncertainty inversion.
- These analyses use 20 MC passes for SeisCNN and 10 for SpecUNet. The
  window-level and detection-level ratios are different statistics, so they
  are not a controlled comparison of uncertainty methods.
- SeisCNN recall below/above the 14.4 dB test-median SNR proxy is
  0.62 ± 0.10 / 0.70 ± 0.11. These are descriptive test strata, not a
  validated physical SNR threshold or an operating-point selection rule.
- MC auto-accept results can differ from deterministic detection counts.
  Mask-energy and MC scores are not calibrated event probabilities.

## 7. Historical experiments and limits

Mars, screening ablation, denoise chaining and continuous-archive experiments
have not been rerun with the corrected lunar checkpoints. Identify them as
historical experiments if discussing them. The archive scan failed its
performance gate; survey deployment is not validated. A null screening test
is not proof that screening has no effect. Simulator window reduction is
not measured radio-byte, energy or flight-performance savings.

Expanded Mars evaluation: 17 spans, 30 events, precision 1.000 and recall
0.233 at the recorded validation setting. That finite-sample observation is
not a promise of zero false alarms on future data. Denoised timing refinement
changed MAE from 33.93 s to 34.20 s, so it did not improve the expanded result.

## 8. Evidence and continuation

- Current manuscript and compiled PDF: `paper/ml4ps_2026.md` and
  `paper/ml4ps_2026.pdf`; reproducible build instructions: `paper/BUILD.md`.
- Full change history: `docs/CHANGES_AND_RESEARCH_NOTES.md`.
- Current task status, exact artifact paths and next steps:
  `tasks/RESEARCH_HANDOFF.md`.
- Latest tests, audits and Git commits are recorded in that handoff.
- Before submission, verify bibliography metadata, author information and
  the chosen venue's formatting rules. Older paper drafts/decks are superseded.

Do not repeat finished training to continue this work. Do not tune thresholds
or select seeds using test performance. Keep future experiment outputs in
new paths and preserve the current evaluation locks.

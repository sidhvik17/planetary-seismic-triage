# B3 — full-archive scan vs the Nakamura catalog

**Date:** 2026-07-25 · **Model:** `runs/unet_lunar_screened/best.pt`
(Nakamura-screened injection training, seed 42) · **Data:** 6,850 valid
station-days across S12/S15/S16, `PLANETSEIS_ARCHIVE=C:\planetseis_archive`

## Verdict: the gate does not pass

The plan was B1 → B3 → *if the false-alarm rate is acceptable* → B4 → paper.
**It is not acceptable at any useful recall**, so B4 (multi-station
coincidence of new candidates) should not proceed on this detector.

S12, the station the model was trained on, is the best case:

| threshold | min-dur | detections | recall vs full catalog | survey precision | unmatched / station-day |
|---|---|---|---|---|---|
| 0.10 | 120 s | 45,265 | **0.525** | 0.063 | **15.4** |
| 0.15 | 120 s | 19,109 | 0.364 | 0.104 | 6.22 |
| 0.20 | 120 s | 5,342 | 0.214 | 0.218 | 1.52 |
| 0.15 | 600 s | 902 | 0.071 | **0.427** | 0.19 |
| 0.25 | 600 s | 256 | 0.008 | 0.176 | 0.077 |

Either ~50 % recall at 15 false alarms per station-day, or a tolerable
0.19/day at 7 % recall. Neither is competitive with published lunar work —
Knapmeyer-Endrun & Hammer (2015) report >95 % on impacts and ~70 % on deep
moonquakes from a single-station HMM.

The benchmark-tuned operating point (0.25 / 600 s) recovers **0.8 %** of the
catalog. It is tuned to the Grade-A selection function — the largest, clearest
events — and does not transfer to survey work.

Shortening the gates to suit short events does not rescue it
(`sweep_shortcoda_S12.json`): at coda 300 s / min-dur 30 s the best trade is
recall 0.263 at 4.95 unmatched/day, *worse* than the long-gate configuration.

## What the type stratification shows (the part worth keeping)

Classification corrected on 2026-07-25 — see below. At threshold 0.10 /
min-dur 120 s:

| class | S12 | S15 | S16 |
|---|---|---|---|
| shallow_moonquake | — | **0.60** (12/20) | **0.52** (12/23) |
| deep_moonquake | 0.522 (1849/3544) | 0.214 (502/2351) | 0.143 (607/4241) |
| code_C | 0.638 | 0.345 | 0.307 |
| unclassified | 0.414 | 0.206 | 0.116 |

Two things are real here:

1. **Shallow moonquakes — the rarest and most scientifically significant
   class, 28 in the entire catalog — have the highest recall of any class**,
   from a detector that never saw a labeled positive. They are the largest,
   most impulsive events, i.e. closest to the Grade-A templates the injection
   engine was built from.
2. **Recall collapses off the training station**: 0.52 → 0.21 → 0.14 for deep
   moonquakes across S12 → S15 → S16 at a fixed operating point. The earlier
   cross-station claim (95/96 weak-label files) was measured on files curated
   around a known event; on continuous data the picture is much weaker.

## Why recall is low — the honest mechanism

The injection engine synthesises variants of **45 Grade-A lunar templates**,
all large, all S12. Deep moonquakes are 56 % of the catalog, are tiny, highly
repetitive, and are conventionally found by **matched filtering** precisely
because they repeat (Bulow et al. 2005/2007; Nakamura 2003). A morphology
detector trained on large impulsive events has no reason to find them, and
does not. This is a coherent negative, not a bug.

## Catalog classification was wrong until today — correction recorded

The earlier parse read column 76 as the event type, mapping `A` →
"artificial impact". That is provably wrong:

* Only **9** artificial impacts exist in the literature; column-76 `A` occurs
  **1,359** times.
* Events sharing a single deep-moonquake nest carry *different* column-76
  codes — `M`-coded and blank-coded events share **247 of 250** nests — so
  column 76 does not encode event class at all.

The nest field (columns 81–84, `A###`/`T###`) is what identifies deep
moonquakes: **7,318 events across 320 distinct nests (56 % of the catalog)**,
matching the published ">55 % deep moonquakes" and Nakamura (2003)'s ~7,245.
Column-76 `H` gives **28** shallow moonquakes, matching the published 28
exactly. Remaining codes are reported under their raw letter rather than
given invented names, pending UTIG TR-18 documentation.

Any figure produced before this correction that names an "artificial impact"
recall is invalid.

## What this means for the two-track plan

**Track A (ML4PS / arXiv) is unaffected and should proceed.** Its claim is
about label-free training under label scarcity on the frozen benchmark, which
stands on its own. The archive result strengthens it by supplying an honest
limitation section.

**Track B needs reframing.** The original claim — "label-free full-archive
detection with type-stratified recall" — is not supported. Defensible
alternatives, in order of preference:

1. **Report it as a negative/limits paper.** "Injection-trained detection does
   not transfer from a 45-template large-event catalog to the deep-moonquake
   population; matched filtering remains necessary." With the type-stratified
   numbers and the corrected catalog parse, this is honest and citable, and
   the shallow-moonquake result is a genuine positive inside it.
2. **Fix the training distribution before rescanning.** Build templates from
   nest-labelled deep moonquakes (320 nests, 7,318 events available) rather
   than only Grade-A. This directly targets the failure mode and is the
   experiment most likely to change the answer.
3. **Drop the archive claim** and publish Track A alone.

Do **not** proceed to B4 on the current detector: at 15 unmatched detections
per station-day, multi-station coincidence would be dominated by chance
coincidences rather than discoveries.

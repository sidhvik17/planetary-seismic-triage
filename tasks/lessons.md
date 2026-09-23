# Project lessons

## Training plans and cross-account continuation — 2026-09-23

- Before choosing a training budget, inspect historical checkpoint configs
  and the benchmark's reporting rules. A CLI default is not evidence of the
  historical recipe. Here the relevant U-Net budget was 30 epochs, and the
  benchmark required at least five seeds.
- When the user brings a completion report from another account, pause
  outstanding writers and reconcile files, hashes, logs and active processes
  before resuming. Do not rerun completed training from stale chat context.
- Keep a handoff current before context or token limits. Record exact commands,
  dataset fingerprints, model paths, verified checks, incomplete work and the
  distinction between local ignored artifacts and version-controlled files.
- Do not infer that leakage had no effect from higher corrected scores when
  split membership, event labels and evaluation populations also changed.
- When a catalog cross-check uses a wider arrival tolerance than the primary
  benchmark, compare matched catalog identities with existing benchmark picks
  before describing them as omitted events or discoveries. Temporal association
  and new-event identity are different questions.
- Verify the exact population in a reported ratio. A variable named `fp` may
  exclude catalog-matched detections; do not call it all false positives without
  checking the stored numerator. Monte Carlo p-values should not be zero.
- Re-read the on-disk handoff and Git state after cross-account continuation.
  Old conversation plans must not overwrite work completed by another account.

## CI parity — 2026-09-23

- Run the suite with CI's exact command (`pytest tests -q`), not only
  `python -m pytest`. The `-m` form puts the working directory on `sys.path`;
  the `pytest` console script does not. Four tests importing `scripts.*` passed
  locally and failed collection on PR #1. `pyproject.toml` now sets
  `pythonpath = ["."]`, so both forms match.
- Before pushing, run that command in a fresh clone too. Ignored local
  artifacts (data cache, models, results) can hide missing-file dependencies.

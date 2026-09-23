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

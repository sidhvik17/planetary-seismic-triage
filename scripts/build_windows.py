"""Build cached datasets from the raw packet (milestones 1-2).

Per body:
  data/cache/{body}/splits.json                     file-level splits
  data/cache/{body}/train_windows.npz               balanced X/y/offset
  data/cache/{body}/val_windows.npz
  data/cache/{body}/continuous/{split}/{stem}.npz   full preprocessed traces + picks
Continuous traces are what evaluation runs on; the balanced window sets are
only for training/early-stopping (F-6: never report metrics on balanced data).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import CODA_SEC, DEFAULT as CFG, DATA_CACHE, SEED
from planetseis.data import load_labeled_traces, split_traces
from planetseis.preprocessing import load_trace, preprocess
from planetseis.windows import make_windows, positives_around_pick


def build_body(body: str):
    rng = np.random.default_rng(SEED)
    traces = load_labeled_traces(body)
    print(f"[{body}] labeled trace files: {len(traces)}, "
          f"events: {sum(len(t.arrival_rel_sec) for t in traces)}")
    if not traces:
        print(f"[{body}] nothing found — check packet layout")
        return

    train, val, test = split_traces(traces)
    out_dir = DATA_CACHE / body
    out_dir.mkdir(parents=True, exist_ok=True)
    # stale hard-negative backup would poison the next mining run
    (out_dir / "train_windows_orig.npz").unlink(missing_ok=True)
    (out_dir / "splits.json").write_text(json.dumps({
        s: [t.trace_path.name for t in grp]
        for s, grp in [("train", train), ("val", val), ("test", test)]
    }, indent=2))

    for split_name, group in [("train", train), ("val", val), ("test", test)]:
        cont_dir = out_dir / "continuous" / split_name
        cont_dir.mkdir(parents=True, exist_ok=True)
        X, y, off = [], [], []
        for lt in group:
            try:
                raw, rate, _ = load_trace(lt.trace_path)
            except Exception as e:
                print(f"  skip {lt.trace_path.name}: {e}")
                continue
            proc, prate = preprocess(raw, rate, CFG.preproc)
            np.savez_compressed(
                cont_dir / (lt.trace_path.stem + ".npz"),
                trace=proc, rate=prate, picks=np.array(lt.arrival_rel_sec),
            )
            if split_name == "test":
                continue  # test is evaluated on continuous traces only
            wins = make_windows(proc, prate, lt.arrival_rel_sec, CFG.window)
            pos = [w for w in wins if w.label == 1]
            # Negatives must not touch any event's coda — those windows are
            # full of real event energy and would be contradictory labels.
            coda = CODA_SEC[body]
            win_sec = CFG.window.n_samples / prate
            neg = [
                w for w in wins
                if w.label == 0 and not any(
                    w.start_sec < p + coda and w.start_sec + win_sec > p
                    for p in lt.arrival_rel_sec
                )
            ]
            for pick in lt.arrival_rel_sec:  # multiply scarce positives (F-5)
                pos += positives_around_pick(proc, prate, pick, CFG.window, n_shifts=12, rng=rng)
            n_neg = min(len(neg), int(np.ceil(len(pos) * CFG.train.neg_pos_ratio)) or 8)
            neg = [neg[i] for i in rng.choice(len(neg), size=n_neg, replace=False)] if neg else []
            for w in pos + neg:
                X.append(w.data)
                y.append(w.label)
                off.append(w.offset_frac)
        if split_name != "test" and X:
            np.savez_compressed(
                out_dir / f"{split_name}_windows.npz",
                X=np.stack(X), y=np.array(y, dtype=np.int64),
                offset=np.array(off, dtype=np.float32),
            )
            print(f"  {split_name}: {len(X)} windows "
                  f"({int(np.sum(y))} pos / {len(y) - int(np.sum(y))} neg)")


if __name__ == "__main__":
    bodies = sys.argv[1:] or ["lunar", "mars"]
    for b in bodies:
        build_body(b)

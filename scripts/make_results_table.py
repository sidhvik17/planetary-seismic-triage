"""Aggregate results/*.json into the report's markdown tables."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from planetseis.config import PROJECT_ROOT

RESULTS = PROJECT_ROOT / "results"


def fmt(d, key="cnn"):
    m = d[key]
    mae = f"{m['mae_sec']:.1f}" if m.get("mae_sec") is not None else "—"
    return (f"| {d['model_body'].capitalize()}→{d['eval_body'].capitalize()} "
            f"| {'CNN' if key == 'cnn' else 'STA/LTA'} "
            f"| {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {mae} "
            f"| {m['tp']}/{m['fp']}/{m['fn']} |")


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    header = ("| Train→Test | Model | Precision | Recall | F1 | MAE (s) | TP/FP/FN |\n"
              "|---|---|---|---|---|---|---|")
    l2l, m2m = load("lunar_to_lunar.json"), load("mars_to_mars.json")
    l2m, m2l = load("lunar_to_mars.json"), load("mars_to_lunar.json")
    noaug = load("lunar_noaug_to_lunar.json")

    print("### Same-body detection (held-out test)\n")
    print(header)
    for r in (l2l, m2m):
        if r:
            print(fmt(r))
            if "sta_lta" in r:
                print(fmt(r, "sta_lta"))

    print("\n### Cross-body transfer (zero adaptation)\n")
    print(header)
    for r in (l2m, m2l):
        if r:
            print(fmt(r))

    if noaug and l2l:
        print("\n### Augmentation ablation (lunar→lunar)\n")
        print(header)
        print(fmt(l2l).replace("Lunar→Lunar", "Full augmentation"))
        print(fmt(noaug).replace("Lunar→Lunar", "No augmentation"))

    print("\nThresholds:", {n: r["threshold"] for n, r in
                            [("l2l", l2l), ("m2m", m2m), ("l2m", l2m), ("m2l", m2l)] if r})


if __name__ == "__main__":
    main()

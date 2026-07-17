"""Catalog-extension scan: find events the shipped catalog does not label.

MarsQuakeNet's headline contribution was a fully automated catalog that
replicated the manual one and EXTENDED it with low-SNR events. This is the
same exercise on this project's data, with MC-Dropout uncertainty carried on
every candidate:

  * lunar: the packet's uncatalogued station sets (S12 Grade B, S15, S16 —
    each file curated around one known-but-unlabeled event) AND the labeled
    test split, where any confident detection far from a catalog pick is a
    candidate missed event.
  * mars_ext: all files, candidates = detections not matching any MQS pick.

Output: results/catalog_extension/candidates.csv (one row per detection:
file, group, time, confidence, uncertainty, matches_catalog, needs_review)
plus summary.json with per-group detection rates.

Usage: python scripts/scan_catalog_extension.py --model runs/unet_lunar/best.pt
           [--threshold 0.3] [--mc-passes 10] [--body lunar]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import CODA_SEC, DEFAULT as CFG, DATA_CACHE, PACKET_ROOT, PROJECT_ROOT
from planetseis.detect_spec import detect_events_spec_mc
from planetseis.evaluate import score_trace
from planetseis.preprocessing import load_trace, preprocess
from planetseis.unet import UNET_ARCHS, SpecUNet

OUT_DIR = PROJECT_ROOT / "results" / "catalog_extension"


def iter_lunar_unlabeled():
    root = PACKET_ROOT / "data" / "lunar" / "test" / "data"
    for grp in sorted(d for d in root.iterdir() if d.is_dir()):
        for f in sorted(grp.glob("*.mseed")):
            yield grp.name, f


def iter_labeled(body: str):
    for split in ("train", "val", "test"):
        d = DATA_CACHE / body / "continuous" / split
        for p in sorted(d.glob("*.npz")):
            z = np.load(p)
            yield split, p.stem, z["trace"], float(z["rate"]), list(z["picks"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--body", default="lunar", choices=["lunar", "mars_ext"])
    ap.add_argument("--threshold", type=float, required=True,
                    help="operating point from eval_unet's val sweep")
    ap.add_argument("--min-dur", type=float, default=600.0,
                    help="minimum event duration (s) from eval_unet's sweep")
    ap.add_argument("--mc-passes", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: max unlabeled files per group")
    ap.add_argument("--skip-labeled", action="store_true",
                    help="scan only the unlabeled station sets "
                         "(labeled-split candidates come from "
                         "crosscheck_nakamura.py)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    model = SpecUNet(base=UNET_ARCHS[ckpt.get("arch", "base")])
    model.load_state_dict(ckpt["model"])

    coda = CODA_SEC[args.body]
    tol = CFG.window.match_tolerance_sec
    rows = []
    group_stats: dict[str, dict] = {}

    # 1) labeled files: detections vs catalog -> candidates = unmatched
    for split, stem, trace, rate, picks in (
            [] if args.skip_labeled else iter_labeled(args.body)):
        dets, _, _ = detect_events_spec_mc(
            model, trace, rate, CFG, args.threshold, device,
            suppress_sec=coda, n_passes=args.mc_passes,
            min_dur_sec=args.min_dur)
        for d in dets:
            matched = any(abs(d.time_sec - p) <= tol for p in picks)
            rows.append(dict(group=f"labeled_{split}", file=stem,
                             time_sec=round(d.time_sec, 1),
                             confidence=round(d.confidence, 4),
                             uncertainty=round(d.uncertainty, 4),
                             matches_catalog=matched,
                             needs_review=d.needs_review))
        print(f"labeled/{split} {stem}: {len(dets)} dets, "
              f"{sum(r['matches_catalog'] for r in rows if r['file']==stem)} matched")

    # 2) unlabeled station sets (lunar only)
    if args.body == "lunar":
        for grp, f in iter_lunar_unlabeled():
            st = group_stats.setdefault(grp, {"files": 0, "hit": 0, "dets": 0})
            if args.limit and st["files"] >= args.limit:
                continue
            try:
                raw, rate, _ = load_trace(f)
                proc, prate = preprocess(raw, rate, CFG.preproc)
                dets, _, _ = detect_events_spec_mc(
                    model, proc, prate, CFG, args.threshold, device,
                    suppress_sec=coda, n_passes=args.mc_passes,
                    min_dur_sec=args.min_dur)
            except Exception as e:
                print(f"  {f.name}: SKIP ({e})")
                continue
            st["files"] += 1
            st["hit"] += bool(dets)
            st["dets"] += len(dets)
            for d in dets:
                rows.append(dict(group=grp, file=f.stem,
                                 time_sec=round(d.time_sec, 1),
                                 confidence=round(d.confidence, 4),
                                 uncertainty=round(d.uncertainty, 4),
                                 matches_catalog=None,
                                 needs_review=d.needs_review))
            print(f"{grp} {f.stem}: {len(dets)} dets")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / f"candidates_{args.body}.csv", index=False)

    lab = df[df.group.str.startswith("labeled")] if len(df) else df
    summary = {
        "model": Path(args.model).parent.name,
        "body": args.body,
        "threshold": args.threshold,
        "mc_passes": args.mc_passes,
        "labeled_detections": int(len(lab)),
        "labeled_matched": int(lab.matches_catalog.fillna(False).sum()) if len(lab) else 0,
        "candidates_new": int((~lab.matches_catalog.fillna(False)).sum()) if len(lab) else 0,
        "candidates_new_confident": int(
            ((~lab.matches_catalog.fillna(False)) & (~lab.needs_review)).sum())
        if len(lab) else 0,
        "station_sets": {g: {**s, "rate": round(s["hit"] / max(s["files"], 1), 3)}
                         for g, s in group_stats.items()},
    }
    (OUT_DIR / f"summary_{args.body}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

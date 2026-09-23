"""Create or audit the versioned acquisition-grouped lunar benchmark.

Original caches, split manifests and weights are never changed by this script.
The full source inventory is built before writing any new cache file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import DATA_CACHE, PACKET_ROOT, PROJECT_ROOT
from planetseis.grouped_data import (audit_manifest, build_cache, collect_sources,
                                    plan_manifest, sha256_file)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-root", type=Path, default=PACKET_ROOT)
    parser.add_argument("--historical", type=Path, default=PROJECT_ROOT / "benchmark/lunar_splits.json")
    parser.add_argument("--output", type=Path, default=DATA_CACHE / "lunar_grouped_v1")
    parser.add_argument("--manifest-output", type=Path, default=PROJECT_ROOT / "benchmark/lunar_grouped_v1.json")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        path = args.output / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        report = audit_manifest(manifest, args.output)
        for window in manifest["windows"].values():
            if sha256_file(args.output / window["file"]) != window["sha256"]:
                raise ValueError("Training-window archive hash changed")
        if args.manifest_output.exists() and args.manifest_output.read_bytes() != path.read_bytes():
            raise ValueError("Published manifest differs from cache manifest")
    else:
        if args.manifest_output.exists():
            raise FileExistsError(f"Existing published manifest: {args.manifest_output}; use --audit-only")
        sources, provenance = collect_sources(args.packet_root, args.historical)
        manifest = plan_manifest(sources, provenance)
        # All metadata assertions precede any training-cache write.
        print(json.dumps(audit_manifest(manifest), indent=2), flush=True)
        report = build_cache(manifest, args.packet_root, args.output)
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_output.write_bytes((args.output / "manifest.json").read_bytes())
    print(json.dumps(report, indent=2))
    print(f"manifest SHA256: {sha256_file(args.output / 'manifest.json')}")


if __name__ == "__main__":
    main()

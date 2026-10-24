"""Build a FaceForensics++ manifest CSV (PRD Section 13) from a downloaded FF++ root.

Usage:
    python scripts/build_manifest.py /data/ffpp --splits /data/ffpp/splits
    python scripts/build_manifest.py /data/ffpp --splits /data/ffpp/splits \\
        --limit-per-class 20 --balance --out data/manifests/ffpp_c23_pilot.csv

Writes the CSV (paths relative to the FF++ root) and a JSON sidecar with the
build options, counts, license notes and warnings. Keep both under data/,
which Git ignores: FF++ may only be used for non-commercial research and
education, and no video or per-video data may be committed or shared.
Exits with status 2 if the splits would leak a source group across splits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pulsecheck.dataset import LeakageError, write_manifest
from pulsecheck.dataset.ffpp import METHODS, ManifestOptions, build_manifest
from pulsecheck.dataset.manifest import manifest_counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="FF++ download root")
    parser.add_argument("--out", type=Path, default=Path("data/manifests/ffpp_c23.csv"))
    parser.add_argument("--compression", default="c23", choices=("raw", "c23", "c40"))
    parser.add_argument(
        "--methods",
        default=",".join(METHODS),
        help=f"comma-separated manipulation methods (default: {','.join(METHODS)})",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        help="folder with the official train.json, val.json and test.json; without it, "
        "linked videos are grouped and split by a seeded hash",
    )
    parser.add_argument("--include-dfd", action="store_true", help="add DeepFakeDetection videos")
    parser.add_argument(
        "--limit-per-class", type=int, help="at most N real and N fake videos in each split"
    )
    parser.add_argument(
        "--balance", action="store_true", help="downsample fakes to the real count in each split"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    options = ManifestOptions(
        compression=args.compression,
        methods=tuple(m.strip() for m in args.methods.split(",") if m.strip()),
        splits_dir=args.splits,
        include_dfd=args.include_dfd,
        limit_per_class=args.limit_per_class,
        balance=args.balance,
        seed=args.seed,
    )
    try:
        result = build_manifest(args.root, options)
    except LeakageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    sidecar = write_manifest(result.rows, args.out, result.meta)
    for warning in result.warnings[:20]:
        print(f"warning: {warning}", file=sys.stderr)
    if len(result.warnings) > 20:
        print(f"warning: ... {len(result.warnings) - 20} more (see sidecar)", file=sys.stderr)
    print(f"{len(result.rows)} videos -> {args.out} (sidecar {sidecar.name})")
    for split, counts in result.meta["groups_before_sampling"]["groups_per_split"].items():
        print(f"  {split}: {counts} source groups before sampling")
    for split, counts in manifest_counts(result.rows).items():
        print(f"  {split}: real {counts.get('real', 0)}, fake {counts.get('fake', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Record non-physiological video properties per manifest row (shortcut check).

Usage:
    python scripts/probe_media.py data/manifests/nfi.csv --workers 4

Writes data/features/<manifest name>_media.csv with resolution, container frame
rate and duration, bitrate and mean frame brightness per video. Join it to a
feature table with --extra-columns in train_model.py and evaluate.py. Keep it
under data/ (ignored by Git).
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
from pathlib import Path

from pulsecheck.dataset import read_manifest, read_sidecar
from pulsecheck.dataset.media import probe_media, write_media_table


def _probe(task: tuple[str, Path]) -> dict:
    video_id, path = task
    if not path.is_file():
        return {"video_id": video_id, "probe_error": "file_not_found"}
    try:
        return {"video_id": video_id, **probe_media(path)}
    except Exception as exc:  # noqa: BLE001 - recorded per video
        return {"video_id": video_id, "probe_error": f"{type(exc).__name__}: {exc}"[:200]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--root", type=Path, help="dataset root (default: from the manifest)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    root = args.root or Path(read_sidecar(args.manifest).get("root", "."))
    if not root.is_dir():
        print(f"error: dataset root {root} not found (pass --root)", file=sys.stderr)
        return 1
    out = args.out or Path("data/features") / f"{args.manifest.stem}_media.csv"
    tasks = [(r.video_id, root / r.path) for r in rows]
    if args.workers > 1:
        with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
            results = pool.map(_probe, tasks)
    else:
        results = [_probe(t) for t in tasks]
    write_media_table(out, results)
    errors = sum(bool(r.get("probe_error")) for r in results)
    print(f"{len(results)} videos -> {out} ({errors} with probe errors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Write normalised copies of a manifest's videos and a manifest for them (shortcut fix).

Usage:
    python scripts/normalize_videos.py data/manifests/nfi.csv --workers 8

Each video becomes a fixed face-centred 256x256 crop at a constant 30 fps,
cut to its first 8 s, with one gain per clip to a fixed mean brightness, and
re-encoded with identical libx264 settings (pulsecheck.dataset.normalize).
Copies go under data/Huggingface/<root name>_normalized/ with the same
relative layout; the new manifest (default data/manifests/<name>_norm.csv)
keeps every id, label, split, group and cv_fold and points at the copies. Its
JSON sidecar records the settings and every clip that failed (too short, no
face, decode error) with its reason; failed clips stay in the manifest, so
extraction counts them as failed (normalization_<reason>, e.g.
normalization_too_short) instead of dropping them.
Per-clip summary numbers (crop box, gain, clipped fraction) go to
normalization.csv in the output root. Everything is derived data: keep it
under data/ (ignored by Git) and do not redistribute it.
"""

from __future__ import annotations

import argparse
import csv
import logging
import multiprocessing
import sys
import time
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from pulsecheck import __version__
from pulsecheck.dataset import check_leakage, read_manifest, read_sidecar
from pulsecheck.dataset.manifest import file_sha256, write_manifest
from pulsecheck.dataset.normalize import (
    NormalizeConfig,
    NormalizeFailure,
    normalize_video,
    settings,
)

log = logging.getLogger("pulsecheck.normalize")


def _run(task: tuple[str, Path, Path, NormalizeConfig]) -> dict:
    video_id, src, dst, config = task
    start = time.perf_counter()
    if not src.is_file():
        return {"video_id": video_id, "error": "file_not_found", "detail": str(src)}
    try:
        stats = normalize_video(src, dst, config)
        return {"video_id": video_id, "error": "", **stats}
    except NormalizeFailure as exc:
        return {"video_id": video_id, "error": exc.code, "detail": exc.detail}
    except Exception as exc:  # noqa: BLE001 - recorded per video
        return {"video_id": video_id, "error": "pipeline_error", "detail": f"{exc}"[:200]}
    finally:
        log.debug("%s done in %.1fs", video_id, time.perf_counter() - start)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, help="dataset root (default: from the manifest)")
    parser.add_argument("--out-root", type=Path, help="default: <root>_normalized")
    parser.add_argument("--out-manifest", type=Path, help="default: <manifest>_norm.csv")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--size", type=int, default=NormalizeConfig.size)
    parser.add_argument("--fps", type=float, default=NormalizeConfig.fps)
    parser.add_argument("--duration", type=float, default=NormalizeConfig.duration_s)
    parser.add_argument("--crf", type=int, default=NormalizeConfig.crf)
    parser.add_argument("--target-luma", type=float, default=NormalizeConfig.target_luma)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    rows = read_manifest(args.manifest)
    meta = read_sidecar(args.manifest)
    root = args.root or Path(meta.get("root", "."))
    if not root.is_dir():
        print(f"error: dataset root {root} not found (pass --root)", file=sys.stderr)
        return 1
    out_root = (args.out_root or root.with_name(root.name + "_normalized")).resolve()
    if out_root == root.resolve():
        print("error: --out-root must differ from the dataset root", file=sys.stderr)
        return 1
    out_manifest = args.out_manifest or args.manifest.with_name(args.manifest.stem + "_norm.csv")
    config = NormalizeConfig(
        size=args.size,
        fps=args.fps,
        duration_s=args.duration,
        crf=args.crf,
        target_luma=args.target_luma,
    )
    new_rows = [replace(r, path=Path(r.path).with_suffix(".mp4").as_posix()) for r in rows]
    tasks = [
        (r.video_id, root / r.path, out_root / n.path, config)
        for r, n in zip(rows, new_rows, strict=True)
    ]
    start = time.perf_counter()
    results = []
    with multiprocessing.get_context("spawn").Pool(max(1, args.workers)) as pool:
        for i, result in enumerate(pool.imap_unordered(_run, tasks), 1):
            results.append(result)
            status = result["error"] or "ok"
            log.info("[%d/%d] %s %s", i, len(tasks), result["video_id"], status)

    order = {r.video_id: i for i, r in enumerate(rows)}
    results.sort(key=lambda r: order[r["video_id"]])
    columns = list(dict.fromkeys(k for r in results for k in r))
    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "normalization.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)

    failed = {r["video_id"]: f"{r['error']}: {r.get('detail', '')}" for r in results if r["error"]}
    leakage = check_leakage(new_rows)  # same links as the source manifest
    keep = {k: v for k, v in meta.items() if k not in ("manifest_sha256", "counts")}
    license_notes = (
        f"{meta.get('license_notes', '')} Normalised copies are derived data under the same "
        "terms: keep under data/, do not redistribute."
    ).strip()
    new_rows = [replace(r, license_notes=license_notes) for r in new_rows]
    side = write_manifest(
        new_rows,
        out_manifest,
        {
            **keep,
            "root": str(out_root),
            "compression": f"normalized-crf{config.crf}",
            "license_notes": license_notes,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "pulsecheck_version": __version__,
            "normalized_from": {
                "manifest": args.manifest.name,
                "manifest_sha256": file_sha256(args.manifest),
                "root": str(root.resolve()),
            },
            "normalization": settings(config),
            "normalization_failed": failed,
            "normalization_failure_counts": dict(Counter(f.split(":")[0] for f in failed.values())),
            "leakage_check": leakage,
        },
    )
    elapsed = time.perf_counter() - start
    print(
        f"{len(results) - len(failed)} of {len(results)} videos normalised into {out_root} "
        f"in {elapsed:.0f}s; {len(failed)} failed"
    )
    for video_id, reason in failed.items():
        print(f"  failed {video_id}: {reason}")
    print(f"manifest: {out_manifest} (+ {side.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

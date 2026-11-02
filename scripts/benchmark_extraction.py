"""Semi-synthetic pulse-injection benchmark of the extraction pipeline (PRD FR-12).

Usage:
    # 1. Pick source clips (real train videos only) and build the trace cache:
    #    inject a known pulse, re-encode, re-detect, trace every spatial variant.
    python scripts/benchmark_extraction.py build --workers 10
    # 2. Evaluate pipeline variants on the cache: diagnose the current pipeline,
    #    select a variant on the dev half, recalibrate the quality ramps on dev,
    #    and report current vs selected once on the held-out half.
    python scripts/benchmark_extraction.py evaluate --workers 12
    # Optional: UBFC-rPPG ground truth (if data/UBFC-rPPG/ exists; see README).
    python scripts/benchmark_extraction.py ubfc --workers 8

Sources: FaceForensics++ c23 original (real) videos and normalised NFI-rPPG
real clips, from the train splits only; no validation or test video is read.
Clips are split by source group into dev and held-out halves
(pulsecheck.benchmark.sources). The cache (data/benchmark/cache/) holds
per-frame ROI colour means and per-clip diagnostics, the reports
(data/reports/benchmark/) aggregate numbers only. Both are derived data from
non-commercial datasets: keep them under data/ (ignored by Git) and do not
share them. `build` is resumable: re-run the same command after an
interruption. A log goes to data/benchmark/build.log.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import multiprocessing
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from pulsecheck import __version__
from pulsecheck.benchmark.sources import BenchmarkClip, SourceSpec, select_clips

BENCH_DIR = Path("data/benchmark")
REPORT_DIR = Path("data/reports/benchmark")
log = logging.getLogger("pulsecheck.benchmark")
LICENSE_NOTE = (
    "Derived from FaceForensics++ (non-commercial research and education only) and "
    "NFI-rPPG (no license stated; real clips from Celeb-DF, non-commercial research). "
    "Do not share the cache or per-clip numbers."
)


def _logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(path, encoding="utf-8")],
    )


def write_clip_list(clips: list[BenchmarkClip], path: Path, specs: list[SourceSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(clips[0].to_dict())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for clip in clips:
            writer.writerow(clip.to_dict())
    side = {
        "sources": [
            {
                **s.__dict__,
                "manifest": str(s.manifest),
                "root": str(s.root) if s.root else None,
            }
            for s in specs
        ],
        "counts": {
            f"{d}/{s}": n for (d, s), n in Counter((c.dataset, c.split) for c in clips).items()
        },
        "groups": {
            f"{d}/{s}": n
            for (d, s), n in Counter(
                (c.dataset, c.split) for c in {c.group: c for c in clips}.values()
            ).items()
        },
        "selection": "real clips from each manifest's train split only, round-robin over "
        "shuffled source groups (seed 0); groups split into dev and held-out halves",
        "license_notes": LICENSE_NOTE,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path.with_suffix(".csv.json").write_text(json.dumps(side, indent=2), encoding="utf-8")


def read_clip_list(path: Path) -> list[BenchmarkClip]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            BenchmarkClip(**{**row, "max_duration_s": float(row["max_duration_s"])})
            for row in csv.DictReader(handle)
        ]


def cmd_build(args) -> int:
    from pulsecheck.benchmark.cache import build_clip

    _logging(args.bench_dir / "build.log")
    clip_list = args.bench_dir / "clips.csv"
    if clip_list.is_file():
        clips = read_clip_list(clip_list)
        log.info("using the existing clip list %s (%d clips)", clip_list, len(clips))
    else:
        specs = [
            SourceSpec("ffpp", args.ffpp_manifest, args.ffpp_clips, args.ffpp_seconds),
            SourceSpec("nfi", args.nfi_manifest, args.nfi_clips, args.nfi_seconds),
        ]
        clips = [c for spec in specs if spec.manifest.is_file() for c in select_clips(spec)]
        if not clips:
            print("error: no source clips found (check the manifests)", file=sys.stderr)
            return 1
        write_clip_list(clips, clip_list, specs)
        log.info("selected %d clips -> %s", len(clips), clip_list)
    cache = args.bench_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    tasks = [(clip, str(cache), str(args.model) if args.model else None) for clip in clips]
    start = time.perf_counter()
    statuses: Counter[str] = Counter()

    def record(i: int, result: dict) -> None:
        statuses[result["status"]] += 1
        elapsed = time.perf_counter() - start
        log.info(
            "[%d/%d] %s %s%s %.0fs (elapsed %.0fs)",
            i,
            len(tasks),
            result["clip_id"],
            result["status"],
            f" ({result['reason']})" if result.get("reason") else "",
            result.get("runtime_s", 0.0),
            elapsed,
        )
        if result.get("traceback"):
            log.error(result["traceback"])

    if args.workers <= 1:
        for i, task in enumerate(tasks, 1):
            record(i, build_clip(task))
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(args.workers, maxtasksperchild=4) as pool:
            for i, result in enumerate(pool.imap_unordered(build_clip, tasks), 1):
                record(i, result)
    log.info("done: %s in %.0fs", dict(statuses), time.perf_counter() - start)
    return 0 if not statuses.get("error") else 1


def cmd_evaluate(args) -> int:
    from pulsecheck.benchmark.run import run_benchmark

    _logging(args.bench_dir / "evaluate.log")
    clips = read_clip_list(args.bench_dir / "clips.csv")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    paths = run_benchmark(
        clips,
        args.bench_dir / "cache",
        args.out_dir,
        stamp,
        workers=args.workers,
        n_boot=args.n_boot,
        meta={"pulsecheck_version": __version__, "license_notes": LICENSE_NOTE},
    )
    for path in paths:
        print(f"report: {path}")
    return 0


def cmd_ubfc(args) -> int:
    from pulsecheck.benchmark.ubfc import run_ubfc

    _logging(args.bench_dir / "ubfc.log")
    if not (args.ubfc_root.is_dir() and any(args.ubfc_root.glob("subject*"))):
        print(
            f"UBFC-rPPG not found under {args.ubfc_root}; skipped. Request it from the "
            "dataset authors (see README), accept its terms and unpack DATASET_2 there."
        )
        return 0
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for path in run_ubfc(
        args.ubfc_root,
        args.bench_dir / "ubfc_cache",
        args.out_dir,
        stamp,
        args.selection,
        workers=args.workers,
        model_path=str(args.model) if args.model else None,
    ):
        print(f"report: {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bench-dir", type=Path, default=BENCH_DIR)
    parser.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--model", type=Path, help="face_landmarker.task path")
    parser.add_argument("--workers", type=int, default=1)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="select clips and build the trace cache")
    build.add_argument("--ffpp-manifest", type=Path, default=Path("data/manifests/ffpp_c23.csv"))
    build.add_argument("--nfi-manifest", type=Path, default=Path("data/manifests/nfi_norm.csv"))
    build.add_argument("--ffpp-clips", type=int, default=60)
    build.add_argument("--nfi-clips", type=int, default=40)
    build.add_argument("--ffpp-seconds", type=float, default=10.0)
    build.add_argument("--nfi-seconds", type=float, default=8.0)
    evaluate = sub.add_parser("evaluate", help="variants, selection, held-out report")
    evaluate.add_argument("--n-boot", type=int, default=1000)
    ubfc = sub.add_parser("ubfc", help="UBFC-rPPG ground-truth benchmark, if present")
    ubfc.add_argument("--ubfc-root", type=Path, default=Path("data/UBFC-rPPG"))
    ubfc.add_argument(
        "--selection",
        type=Path,
        help="selection JSON from `evaluate` (default: the latest under --out-dir)",
    )
    args = parser.parse_args()
    return {"build": cmd_build, "evaluate": cmd_evaluate, "ubfc": cmd_ubfc}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

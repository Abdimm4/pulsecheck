"""Extract one feature row per manifest video into a feature table (PRD FR-11).

Usage:
    python scripts/extract_features.py data/manifests/ffpp_c23.csv --workers 4
    python scripts/extract_features.py data/manifests/ffpp_c23.csv --retry-failed

Writes data/features/<manifest name>.csv (one row per video: metadata,
status, runtime and features) and a .csv.json sidecar with the manifest hash,
dataset/license notes and pipeline settings. Rows are appended as videos
finish, so re-running the same command resumes; failed videos are recorded
with a reason code and skipped on resume unless --retry-failed is given.
Frames, landmarks and face geometry are never stored. Keep the output under
data/ (ignored by Git): derived FF++ data must not be committed or shared.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from pulsecheck import __version__
from pulsecheck.config import load_config
from pulsecheck.dataset import read_manifest, read_sidecar
from pulsecheck.dataset.extraction import FatalExtractionError, PipelineAnalyzer, run_extraction
from pulsecheck.dataset.manifest import file_sha256, sidecar_path
from pulsecheck.dataset.table import read_table
from pulsecheck.features import FEATURE_SCHEMA_VERSION

DATASET_KEYS = (
    "dataset",
    "compression",
    "methods",
    "include_dfd",
    "split_source",
    "sampling",
    "license_notes",
    "non_commercial_use_only",
)


def check_landmarker(model: Path | None) -> str | None:
    from pulsecheck.face import FaceLandmarkerDetector, LandmarkerUnavailableError

    try:
        with FaceLandmarkerDetector(model):
            return None
    except LandmarkerUnavailableError as exc:
        return str(exc)


def write_table_sidecar(out: Path, manifest: Path, analyzer: PipelineAnalyzer, summary) -> Path:
    meta = read_sidecar(manifest)
    statuses = Counter(r["status"] for r in read_table(out)) if out.is_file() else Counter()
    side = {
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "dataset": {k: meta.get(k) for k in DATASET_KEYS},
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "pipeline": analyzer.settings(),
        "pulsecheck_version": __version__,
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rows": dict(statuses),
        "last_run": summary.__dict__ if summary else None,
        "stores": "per-video features and runtime only; no frames, landmarks or face geometry",
    }
    path = sidecar_path(out)
    path.write_text(json.dumps(side, indent=2), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, help="feature table CSV (default data/features/)")
    parser.add_argument("--root", type=Path, help="dataset root (default: from the manifest)")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="start a fresh table")
    parser.add_argument("--max-frames", type=int, help="only process the first N frames")
    parser.add_argument("--config", type=Path, help="YAML config (default configs/default.yaml)")
    parser.add_argument("--model", type=Path, help="face_landmarker.task path")
    parser.add_argument("--log-file", type=Path, help="also log to this file")
    args = parser.parse_args()

    out = args.out or Path("data/features") / f"{args.manifest.stem}.csv"
    handlers = [logging.StreamHandler()]
    log_file = args.log_file or out.with_suffix(".log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers
    )

    rows = read_manifest(args.manifest)
    root = args.root or Path(read_sidecar(args.manifest).get("root", "."))
    if not root.is_dir():
        print(f"error: dataset root {root} not found (pass --root)", file=sys.stderr)
        return 1
    problem = check_landmarker(args.model)
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return 1

    config = load_config(args.config)
    features = config.get("features", {})
    analyzer = PipelineAnalyzer(
        model_path=str(args.model) if args.model else None,
        max_frames=args.max_frames,
        signal=config.get("signal") or {},
        quality=features.get("quality") or {},
        pulse_method=features.get("pulse_method", "chrom"),
    )
    if args.overwrite:
        out.unlink(missing_ok=True)
        sidecar_path(out).unlink(missing_ok=True)
    previous = read_sidecar(out)
    if previous and previous.get("pipeline") != analyzer.settings():
        print(
            "error: the existing table was built with different pipeline settings; "
            "use --out for a new table or --overwrite",
            file=sys.stderr,
        )
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    write_table_sidecar(out, args.manifest, analyzer, None)

    summary = None
    try:
        summary = run_extraction(rows, root, out, analyzer, args.workers, args.retry_failed)
    except FatalExtractionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:  # table layout mismatch
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        write_table_sidecar(out, args.manifest, analyzer, summary)

    logging.info(
        "done: %d processed (%d ok, %d failed), %d skipped as already done, %.0fs; "
        "failure reasons %s",
        summary.processed,
        summary.ok,
        summary.failed,
        summary.skipped,
        summary.elapsed_s,
        summary.failure_reasons,
    )
    print(f"feature table: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

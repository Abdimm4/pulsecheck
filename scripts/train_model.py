"""Train logistic-regression and random-forest classifiers on a feature table (PRD FR-7).

Usage:
    python scripts/train_model.py data/features/ffpp_c23.csv
    python scripts/train_model.py data/features/ffpp_c23.csv --families logreg --no-quality-gate
    python scripts/train_model.py data/features/nfi.csv --cv --manifest data/manifests/nfi.csv \
        --gate data/reports/nfi/proposed_gate.json --tag gate-proposed
    python scripts/train_model.py data/features/nfi.csv --cv --manifest data/manifests/nfi.csv \
        --extra-columns data/features/nfi_media.csv --feature-set shortcut --gate off
    python scripts/train_model.py data/features/ffpp_c23.csv \
        --extra-columns data/features/ffpp_c23_media.csv --feature-set combined --gate off

--cv selects hyperparameters, threshold and band by grouped cross-validation
over the manifest's cv_fold column inside the train split (for small
datasets such as NFI), then refits on the whole train split.
--feature-set picks the candidate features. "pulse" (default) is every pulse
feature except the tracking/coverage ones (face coverage, ROI tracking
coverage, all-ROIs-valid fraction, usable ROI count), which depend on
resolution and compression; the quality gate still reads them. "shortcut"
trains on non-physiological features only (resolution, bitrate, brightness,
fps, duration, face coverage), as a check for source/compression shortcuts,
and "combined" on pulse + shortcut, to test whether the pulse features add
anything beyond the shortcut (scripts/evaluate.py reports the deltas). Both
need the media table from scripts/probe_media.py. "pulse-full" is every
table feature (the model 0.1.0 default).

Fits preprocessing on the train split, picks hyperparameters, the decision
threshold and the inconclusive band on the validation split, and never reads
the test split (score it once with scripts/evaluate.py). Saves
models/<dataset>-<family>-<timestamp>.pkl with a .json sidecar holding the
config, feature list, data version, dataset/license tags and train/val
metrics. Train and validation metrics are not held-out estimates.
Exits with status 2 on split leakage.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from pulsecheck.config import load_config
from pulsecheck.dataset import LeakageError
from pulsecheck.dataset.table import join_tables
from pulsecheck.models.training import (
    FAMILIES,
    FEATURE_SETS,
    TrainConfig,
    load_table,
    provenance_from_table,
    resolve_gate,
    train_models,
)


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("features", type=Path, help="feature table from extract_features.py")
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--config", type=Path, help="YAML config (quality gate from `heuristic`)")
    parser.add_argument(
        "--no-quality-gate", action="store_true", help="do not mark low-quality videos inconclusive"
    )
    parser.add_argument(
        "--gate", default="default", help="quality gate: default, off, or a gate JSON file"
    )
    parser.add_argument("--cv", action="store_true", help="grouped CV over cv_fold (no val)")
    parser.add_argument("--manifest", type=Path, help="manifest to join (cv_fold, fake_family)")
    parser.add_argument(
        "--extra-columns", type=Path, action="append", default=[], help="CSV keyed by video_id"
    )
    parser.add_argument("--feature-set", default="pulse", choices=sorted(FEATURE_SETS))
    parser.add_argument("--tag", help="added to the model file name, e.g. gate-off")
    parser.add_argument("--min-val-coverage", type=float, default=0.8)
    parser.add_argument("--max-missing-fraction", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples for CIs")
    parser.add_argument("--forest-trees", type=int, default=300)
    args = parser.parse_args()

    try:
        gate = resolve_gate(
            "off" if args.no_quality_gate else args.gate,
            load_config(args.config).get("heuristic"),
        )
    except (OSError, ValueError) as exc:
        print(f"error: --gate: {exc}", file=sys.stderr)
        return 1
    config = TrainConfig(
        families=tuple(f.strip() for f in args.families.split(",") if f.strip()),
        max_missing_fraction=args.max_missing_fraction,
        min_val_coverage=args.min_val_coverage,
        quality_gate=gate,
        seed=args.seed,
        n_boot=args.n_boot,
        forest_trees=args.forest_trees,
        feature_set=args.feature_set,
    )
    try:
        data = load_table(args.features, join_tables(args.manifest, args.extra_columns))
    except LeakageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    provenance = provenance_from_table(args.features)
    if not provenance["training_dataset"].get("license"):
        print("error: the feature table records no dataset license", file=sys.stderr)
        return 1
    try:
        models = train_models(data, config, provenance, cv=args.cv)
    except LeakageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for model in models:
        side = model.sidecar
        name = "-".join(x for x in (side["model_name"], args.tag, stamp) if x)
        pkl, json_path = model.save(args.out_dir / f"{name}.pkl")
        val = side["metrics"]["cv_out_of_fold" if args.cv else "val"]
        print(f"{side['family']}: saved {pkl} (+ {json_path.name})")
        used = len(side["features"]["used"])
        print(f"  hyperparameters {side['hyperparameters']}, {used} features")
        print(
            f"  threshold {side['decision']['threshold']:.3f} "
            f"± {side['decision']['inconclusive_margin']:.2f}"
        )
        print(
            f"  {'out-of-fold' if args.cv else 'validation'} (optimistic, used for selection): "
            f"ROC-AUC {_fmt(val['roc_auc'])}, "
            f"coverage {_fmt(val['coverage'])}, decided accuracy {_fmt(val['accuracy'])}"
        )
    print("Next: python scripts/evaluate.py", args.features, "--model <saved .pkl> ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Train logistic-regression and random-forest classifiers on a feature table (PRD FR-7).

Usage:
    python scripts/train_model.py data/features/ffpp_c23.csv
    python scripts/train_model.py data/features/ffpp_c23.csv --families logreg --no-quality-gate

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
from pulsecheck.models.training import (
    FAMILIES,
    QualityGate,
    TrainConfig,
    load_table,
    provenance_from_table,
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
    parser.add_argument("--min-val-coverage", type=float, default=0.8)
    parser.add_argument("--max-missing-fraction", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples for CIs")
    parser.add_argument("--forest-trees", type=int, default=300)
    args = parser.parse_args()

    gate = None
    if not args.no_quality_gate:
        gate = QualityGate.from_heuristic(load_config(args.config).get("heuristic"))
    config = TrainConfig(
        families=tuple(f.strip() for f in args.families.split(",") if f.strip()),
        max_missing_fraction=args.max_missing_fraction,
        min_val_coverage=args.min_val_coverage,
        quality_gate=gate,
        seed=args.seed,
        n_boot=args.n_boot,
        forest_trees=args.forest_trees,
    )
    try:
        data = load_table(args.features)
    except LeakageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    provenance = provenance_from_table(args.features)
    if not provenance["training_dataset"].get("license"):
        print("error: the feature table records no dataset license", file=sys.stderr)
        return 1
    try:
        models = train_models(data, config, provenance)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for model in models:
        side = model.sidecar
        pkl, json_path = model.save(args.out_dir / f"{side['model_name']}-{stamp}.pkl")
        val = side["metrics"]["val"]
        print(f"{side['family']}: saved {pkl} (+ {json_path.name})")
        used = len(side["features"]["used"])
        print(f"  hyperparameters {side['hyperparameters']}, {used} features")
        print(
            f"  threshold {side['decision']['threshold']:.3f} "
            f"± {side['decision']['inconclusive_margin']:.2f}"
        )
        print(
            f"  validation (optimistic, used for selection): ROC-AUC {_fmt(val['roc_auc'])}, "
            f"coverage {_fmt(val['coverage'])}, decided accuracy {_fmt(val['accuracy'])}"
        )
    print("Next: python scripts/evaluate.py", args.features, "--model <saved .pkl> ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

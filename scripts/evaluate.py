"""Evaluate trained models and the heuristic baseline on one split (PRD Section 13, FR-11).

Usage:
    python scripts/evaluate.py data/features/ffpp_c23.csv \\
        --model models/ffpp-c23-logreg-<stamp>.pkl --model models/ffpp-c23-forest-<stamp>.pkl

Defaults to the test split, which each model should see once: the model's
sidecar records the evaluation, and a second test evaluation needs --force
and is flagged in the report. Writes data/reports/evaluation-<split>-<stamp>
.md/.json plus per-video predictions (.predictions.csv) under data/, which Git
ignores. Reports state their evaluation conditions and make no general
accuracy claim. Exits with status 2 on split leakage.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from pulsecheck.config import load_config
from pulsecheck.dataset import SPLITS, LeakageError, read_sidecar
from pulsecheck.models import HeuristicConfig
from pulsecheck.models.report import (
    AlreadyEvaluatedError,
    run_evaluation,
    write_predictions,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("features", type=Path, help="feature table from extract_features.py")
    parser.add_argument("--model", type=Path, action="append", default=[], help="repeatable")
    parser.add_argument("--split", default="test", choices=SPLITS)
    parser.add_argument("--out-dir", type=Path, default=Path("data/reports"))
    parser.add_argument("--config", type=Path, help="YAML config (heuristic thresholds)")
    parser.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples for CIs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--force", action="store_true", help="allow re-evaluating a model on the test split"
    )
    args = parser.parse_args()

    heuristic = HeuristicConfig.from_mapping(load_config(args.config).get("heuristic"))
    try:
        report, predictions = run_evaluation(
            args.features,
            args.model,
            split=args.split,
            heuristic_config=heuristic,
            n_boot=args.n_boot,
            seed=args.seed,
            force=args.force,
            table_meta=read_sidecar(args.features),
        )
    except LeakageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (AlreadyEvaluatedError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = f"evaluation-{args.split}-{stamp}"
    json_path, md_path = write_report(report, args.out_dir, stem)
    write_predictions(args.out_dir / f"{stem}.predictions.csv", predictions)
    print(f"report: {md_path} and {json_path.name}; predictions: {stem}.predictions.csv")
    for name, entry in report["methods"].items():
        m = entry["metrics"]
        auc = "n/a" if m["roc_auc"] is None else f"{m['roc_auc']:.3f}"
        cov = "n/a" if m["coverage"] is None else f"{m['coverage']:.2f}"
        print(f"  {name}: n={m['n']}, coverage {cov}, ROC-AUC {auc} (see report for conditions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

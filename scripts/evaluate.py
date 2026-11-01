"""Evaluate trained models and the heuristic baseline on one split (PRD Section 13, FR-11).

Usage:
    python scripts/evaluate.py data/features/ffpp_c23.csv \\
        --model models/ffpp-c23-logreg-<stamp>.pkl --model models/ffpp-c23-forest-<stamp>.pkl
    python scripts/evaluate.py data/features/nfi.csv --manifest data/manifests/nfi.csv \\
        --extra-columns data/features/nfi_media.csv --model models/nfi-logreg-<stamp>.pkl \\
        --heuristic-gate off --tag gate-off
    # cross-dataset: a model trained on one dataset, scored on every video of another
    python scripts/evaluate.py data/features/ffpp_c23.csv --split all \\
        --model models/nfi-logreg-<stamp>.pkl

Defaults to the test split, which each model should see once: the model's
sidecar records the evaluation, and a second test evaluation needs --force
and is flagged in the report. With --manifest, the manifest's fake_family
column adds a per-family breakdown. --heuristic-gate sets the heuristic's
quality gate (default config, off, or a proposed-gate JSON); trained models
use the gate stored in their sidecar. A model trained on another dataset is
marked CROSS-DATASET in the report, which records both datasets. When the
models include a shortcut model (train_model.py --feature-set shortcut) and
pulse and/or combined models of the same family and training dataset, a
"Shortcut baseline" section gives the ROC-AUC differences (pulse - shortcut,
combined - shortcut) with paired group-bootstrap 95% intervals; models with
extra feature columns need the same --extra-columns here. Writes
data/reports/evaluation-<split>[-<tag>]-<stamp> .md/.json plus per-video
predictions (.predictions.csv) under data/, which Git ignores. Reports state
their evaluation conditions and make no general accuracy claim. Exits with
status 2 on split leakage.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from pulsecheck.config import load_config
from pulsecheck.dataset import SPLITS, LeakageError, read_sidecar
from pulsecheck.dataset.table import family_map, join_tables
from pulsecheck.models import HeuristicConfig
from pulsecheck.models.heuristic import with_gate
from pulsecheck.models.report import (
    ALL_SPLITS,
    AlreadyEvaluatedError,
    run_evaluation,
    write_predictions,
    write_report,
)
from pulsecheck.models.training import resolve_gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("features", type=Path, help="feature table from extract_features.py")
    parser.add_argument("--model", type=Path, action="append", default=[], help="repeatable")
    parser.add_argument("--split", default="test", choices=(*SPLITS, ALL_SPLITS))
    parser.add_argument("--out-dir", type=Path, default=Path("data/reports"))
    parser.add_argument("--config", type=Path, help="YAML config (heuristic thresholds)")
    parser.add_argument("--manifest", type=Path, help="join fake_family/cv_fold for breakdowns")
    parser.add_argument(
        "--extra-columns", type=Path, action="append", default=[], help="CSV keyed by video_id"
    )
    parser.add_argument(
        "--heuristic-gate", default="default", help="default, off, or a gate JSON file"
    )
    parser.add_argument("--tag", help="added to the report file name")
    parser.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples for CIs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--force", action="store_true", help="allow re-evaluating a model on the test split"
    )
    args = parser.parse_args()

    heuristic_values = load_config(args.config).get("heuristic")
    heuristic = HeuristicConfig.from_mapping(heuristic_values)
    try:
        if args.heuristic_gate != "default":
            gate = resolve_gate(args.heuristic_gate, heuristic_values)
            heuristic = with_gate(heuristic, None if gate is None else asdict(gate))
        report, predictions = run_evaluation(
            args.features,
            args.model,
            split=args.split,
            heuristic_config=heuristic,
            n_boot=args.n_boot,
            seed=args.seed,
            force=args.force,
            table_meta=read_sidecar(args.features),
            extra=join_tables(args.manifest, args.extra_columns),
            family_of=family_map(args.manifest),
            heuristic_gate=args.heuristic_gate,
        )
    except LeakageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (AlreadyEvaluatedError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = "-".join(x for x in ("evaluation", args.split, args.tag, stamp) if x)
    json_path, md_path = write_report(report, args.out_dir, stem)
    write_predictions(args.out_dir / f"{stem}.predictions.csv", predictions)
    print(f"report: {md_path} and {json_path.name}; predictions: {stem}.predictions.csv")
    if report["evaluation_conditions"]["cross_dataset"]:
        print("  CROSS-DATASET evaluation: see models_trained_on in the report")
    for name, entry in report["methods"].items():
        m = entry["metrics"]
        auc = "n/a" if m["roc_auc"] is None else f"{m['roc_auc']:.3f}"
        cov = "n/a" if m["coverage"] is None else f"{m['coverage']:.2f}"
        print(f"  {name}: n={m['n']}, coverage {cov}, ROC-AUC {auc} (see report for conditions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

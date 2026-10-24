"""Evaluation reports: heuristic predictions on a feature table, JSON and markdown output."""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from pulsecheck import __version__
from pulsecheck.dataset.manifest import file_sha256
from pulsecheck.dataset.table import row_features
from pulsecheck.models.evaluation import (
    FAILED,
    Prediction,
    evaluate_predictions,
    runtime_summary,
)
from pulsecheck.models.heuristic import (
    MODEL_NAME as HEURISTIC_NAME,
)
from pulsecheck.models.heuristic import (
    MODEL_VERSION as HEURISTIC_VERSION,
)
from pulsecheck.models.heuristic import (
    HeuristicConfig,
    classify,
)
from pulsecheck.models.training import TrainedModel, load_model, load_table

DISCLAIMER = (
    "These numbers describe this evaluation only: the dataset, compression, "
    "manipulation methods, split and pipeline settings listed under evaluation "
    "conditions. They are not a general accuracy claim and do not show that "
    "PulseCheck can tell whether any other video is authentic."
)


def heuristic_predictions(
    rows: Sequence[Mapping[str, str]],
    failed: Sequence[Mapping[str, str]] = (),
    config: HeuristicConfig | None = None,
) -> list[Prediction]:
    """Run the uncalibrated heuristic baseline on stored features.

    Its score is cross-region consistency (high = more consistent with real),
    so 1 - score is used for ROC-AUC; quality-gated videos have no score.
    """
    config = config or HeuristicConfig()
    out = []
    for row in rows:
        result = classify(row_features(row), config)
        score = None if result.score is None else 1.0 - result.score
        reason = "" if result.state.value != "inconclusive" else (result.reasons or [""])[0]
        out.append(
            Prediction(
                video_id=row["video_id"],
                label=int(row["label"] == "fake"),
                manipulation_type=row["manipulation_type"],
                group_id=row["group_id"],
                state=result.state.value,
                score=score,
                reason=_reason_code(reason),
            )
        )
    for row in failed:
        out.append(
            Prediction(
                row["video_id"],
                int(row["label"] == "fake"),
                row["manipulation_type"],
                row["group_id"],
                FAILED,
                reason=row["failure_reason"],
            )
        )
    return out


def _reason_code(reason: str) -> str:
    text = reason.lower()
    for key, code in (
        ("signal quality", "low_signal_quality"),
        ("usable signal", "too_few_usable_rois"),
        ("face tracked", "low_face_coverage"),
        ("between the thresholds", "score_between_thresholds"),
        ("no cross-region", "no_consistency_measure"),
    ):
        if key in text:
            return code
    return reason


def heuristic_description(config: HeuristicConfig) -> dict:
    return {
        "kind": "heuristic",
        "model_name": HEURISTIC_NAME,
        "model_version": HEURISTIC_VERSION,
        "calibrated": False,
        "config": asdict(config),
        "notes": (
            "Hand-set, uncalibrated thresholds; not fitted or tuned on any split. "
            "ROC-AUC uses 1 - consistency score."
        ),
    }


def write_predictions(path: Path, by_method: Mapping[str, Sequence[Prediction]]) -> None:
    fields = ["method", *Prediction.__dataclass_fields__]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, preds in by_method.items():
            for p in preds:
                writer.writerow({"method": method, **asdict(p)})


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _with_ci(metrics: Mapping, name: str) -> str:
    ci = metrics.get("confidence_intervals_95", {}).get(name)
    value = _fmt(metrics.get(name))
    return f"{value} [{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci else value


def render_markdown(report: Mapping) -> str:
    cond = report["evaluation_conditions"]
    lines = [
        f"# PulseCheck evaluation report ({cond['split']} split)",
        "",
        f"Generated {report['created_at']} by PulseCheck {report['pulsecheck_version']}.",
        "",
        f"> {DISCLAIMER}",
        "",
        "## Evaluation conditions",
        "",
    ]
    for key, value in cond.items():
        if key == "notes":
            continue
        shown = json.dumps(value) if isinstance(value, dict | list) else value
        lines.append(f"- **{key}**: {shown}")
    lines += ["", *(f"- {note}" for note in cond.get("notes", [])), ""]

    lines += [
        "## Summary",
        "",
        "Thresholded metrics use decided videos only; 95% intervals are a group-level "
        "bootstrap. Positive class: manipulated (fake).",
        "",
        "| Method | Videos | Coverage | Inconclusive | Failed | Accuracy | Precision | Recall "
        "| Specificity | F1 | ROC-AUC |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, entry in report["methods"].items():
        m = entry["metrics"]
        lines.append(
            f"| {name} | {m['n']} | {_with_ci(m, 'coverage')} | {_fmt(m['inconclusive_rate'])} "
            f"| {_fmt(m['failure_rate'])} | {_with_ci(m, 'accuracy')} "
            f"| {_with_ci(m, 'precision')} | {_with_ci(m, 'recall')} "
            f"| {_with_ci(m, 'specificity')} | {_with_ci(m, 'f1')} | {_with_ci(m, 'roc_auc')} |"
        )
    lines.append("")

    for name, entry in report["methods"].items():
        m = entry["metrics"]
        c = m["confusion_matrix"]
        lines += [
            f"## {name}",
            "",
            f"- kind: {entry['kind']}; model: {entry.get('model_name')} "
            f"{entry.get('model_version', '')}",
        ]
        if entry.get("decision"):
            d = entry["decision"]
            lines.append(
                f"- decision threshold {_fmt(d['threshold'])} ± inconclusive margin "
                f"{_fmt(d['inconclusive_margin'])} (fitted on validation); quality gate: "
                f"{json.dumps(d['quality_gate'])}"
            )
        if entry.get("prior_test_evaluations"):
            lines.append(
                f"- WARNING: this model had {entry['prior_test_evaluations']} earlier test "
                "evaluation(s); these results are no longer a single held-out evaluation."
            )
        if entry.get("notes"):
            lines.append(f"- {entry['notes']}")
        lines += [
            "",
            "Confusion matrix (decided videos):",
            "",
            "| | predicted real | predicted manipulated |",
            "|---|---|---|",
            f"| real | {c['tn']} | {c['fp']} |",
            f"| manipulated | {c['fn']} | {c['tp']} |",
            "",
        ]
        forced = m.get("at_threshold_without_inconclusive")
        if forced:
            lines += [
                "At the bare threshold, without the inconclusive band or quality gate: "
                f"accuracy {_fmt(forced['accuracy'])}, recall {_fmt(forced['recall'])}, "
                f"specificity {_fmt(forced['specificity'])} (n={forced['n']}).",
                "",
            ]
        lines += [
            "| Manipulation | Videos | Coverage | Inconclusive | Failed | Accuracy "
            "| ROC-AUC vs all real |",
            "|---|---|---|---|---|---|---|",
        ]
        for method, pm in m["per_manipulation"].items():
            lines.append(
                f"| {method} | {pm['n']} | {_fmt(pm['coverage'])} | "
                f"{_fmt(pm['inconclusive_rate'])} | {_fmt(pm['failure_rate'])} | "
                f"{_fmt(pm['accuracy'])} | {_fmt(pm.get('roc_auc_vs_all_real'))} |"
            )
        lines.append("")
        if m["inconclusive_reasons"]:
            lines.append(f"Inconclusive reasons: {json.dumps(m['inconclusive_reasons'])}")
            lines.append("")

    rt = report["runtime"]
    lines += [
        "## Runtime and failures",
        "",
        f"- extraction runtime per video (s): {json.dumps(rt['extraction'])}",
        f"- failed extractions: {rt['failed']} of {rt['videos']} "
        f"(rate {_fmt(rt['failure_rate'])}); reasons {json.dumps(rt['failure_reasons'])}",
        f"- prediction time per method (s): {json.dumps(rt['prediction_s'])}",
        "",
    ]
    return "\n".join(lines)


def write_report(report: Mapping, out_dir: Path, stem: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


class AlreadyEvaluatedError(RuntimeError):
    """Raised when a model was already scored on the test split (pass force to override)."""


def run_evaluation(
    table_path: str | Path,
    model_paths: Sequence[str | Path],
    split: str = "test",
    heuristic_config: HeuristicConfig | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    force: bool = False,
    record: bool = True,
    table_meta: Mapping | None = None,
) -> tuple[dict, dict[str, list[Prediction]]]:
    """Score trained models and the heuristic on one split of a feature table.

    For the test split each model's sidecar records the evaluation; a second
    test evaluation of the same model raises AlreadyEvaluatedError unless
    `force` is set, and the report then flags it.
    """
    table_path = Path(table_path)
    data = load_table(table_path)  # raises LeakageError
    rows, failed = data.rows[split], data.failed[split]
    if not rows and not failed:
        raise ValueError(f"no {split} videos in {table_path}")
    heuristic_config = heuristic_config or HeuristicConfig()
    table_meta = dict(table_meta or {})
    table_sha = file_sha256(table_path)

    models = [(Path(p), load_model(p)) for p in model_paths]
    prior = {path: _test_evaluations(model) if split == "test" else 0 for path, model in models}
    for path, _ in models:
        if prior[path] and not force:
            raise AlreadyEvaluatedError(
                f"{path.name} was already evaluated on the test split; "
                "pass --force to re-run, and the report will say so"
            )
    for path, model in models:
        trained_on = model.sidecar.get("data_version", {}).get("feature_table_sha256")
        if trained_on and trained_on != table_sha:
            table_meta.setdefault("warnings", []).append(
                f"{path.name} was trained on a different feature table version"
            )

    predictions: dict[str, list[Prediction]] = {}
    methods: dict[str, dict] = {}
    timings: dict[str, float] = {}
    start = time.perf_counter()
    predictions["heuristic"] = heuristic_predictions(rows, failed, heuristic_config)
    timings["heuristic"] = round(time.perf_counter() - start, 4)
    methods["heuristic"] = heuristic_description(heuristic_config)
    for path, model in models:
        name = model.sidecar.get("model_name", path.stem)
        if name in methods:
            name = path.stem
        start = time.perf_counter()
        predictions[name] = model.predict(rows, failed)
        timings[name] = round(time.perf_counter() - start, 4)
        methods[name] = {
            "kind": "trained",
            "model_file": path.name,
            "model_name": model.sidecar.get("model_name"),
            "model_version": model.sidecar.get("model_version"),
            "family": model.sidecar.get("family"),
            "hyperparameters": model.sidecar.get("hyperparameters"),
            "decision": model.sidecar.get("decision"),
            "training_dataset": model.sidecar.get("training_dataset"),
            "prior_test_evaluations": prior[path],
        }
    for name, preds in predictions.items():
        methods[name]["metrics"] = evaluate_predictions(preds, n_boot, seed)

    all_rows = [*rows, *failed]
    reasons = dict(Counter(r["failure_reason"] for r in failed))
    dataset = table_meta.get("dataset", {})
    now = datetime.now(UTC)
    groups = {r["group_id"] for r in all_rows}
    report = {
        "report_type": "pulsecheck-evaluation",
        "created_at": now.isoformat(timespec="seconds"),
        "pulsecheck_version": __version__,
        "disclaimer": DISCLAIMER,
        "evaluation_conditions": {
            "split": split,
            "dataset": dataset.get("dataset"),
            "compression": dataset.get("compression"),
            "manipulation_methods": dataset.get("methods"),
            "split_source": dataset.get("split_source"),
            "sampling": dataset.get("sampling"),
            "license": dataset.get("license_notes"),
            "videos": len(all_rows),
            "real_videos": sum(r["label"] == "real" for r in all_rows),
            "fake_videos": sum(r["label"] == "fake" for r in all_rows),
            "split_groups": len(groups),
            "feature_table": table_path.name,
            "feature_table_sha256": table_sha,
            "manifest_sha256": table_meta.get("manifest_sha256"),
            "pipeline": table_meta.get("pipeline"),
            "notes": [
                "Positive class is manipulated (fake).",
                "Trained models' hyperparameters, thresholds and inconclusive bands were "
                "chosen on the validation split only; the heuristic is uncalibrated.",
                "Failed extractions count against coverage and are listed, never dropped.",
                "Scores are not calibrated probabilities.",
                *table_meta.get("warnings", []),
            ],
        },
        "methods": methods,
        "runtime": {
            "extraction": runtime_summary([float(r["runtime_s"] or "nan") for r in all_rows]),
            "videos": len(all_rows),
            "failed": len(failed),
            "failure_rate": len(failed) / len(all_rows) if all_rows else None,
            "failure_reasons": reasons,
            "prediction_s": timings,
        },
    }

    if record and split == "test":
        for path, model in models:
            model.sidecar.setdefault("test_evaluations", []).append(
                {
                    "split": "test",
                    "at": report["created_at"],
                    "feature_table_sha256": table_sha,
                    "forced_rerun": bool(prior[path]),
                }
            )
            path.with_suffix(".json").write_text(
                json.dumps(model.sidecar, indent=2, allow_nan=False), encoding="utf-8"
            )
    return report, predictions


def _test_evaluations(model: TrainedModel) -> int:
    return sum(e.get("split") == "test" for e in model.sidecar.get("test_evaluations", []))

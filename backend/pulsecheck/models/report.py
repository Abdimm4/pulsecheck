"""Evaluation reports: heuristic predictions on a feature table, JSON and markdown output."""

from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict
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
    paired_auc_delta,
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

ALL_SPLITS = "all"
DISCLAIMER = (
    "These numbers describe this evaluation only: the dataset, compression, "
    "manipulation methods, split and pipeline settings listed under evaluation "
    "conditions. They are not a general accuracy claim and do not show that "
    "PulseCheck can tell whether any other video is authentic."
)
SHORTCUT_CRITERION = (
    "A pulse claim needs pulse > shortcut and combined > shortcut, with the 95% interval "
    "of each ROC-AUC difference above 0. Otherwise the pulse features have not been shown "
    "to carry signal beyond resolution, compression, frame rate, duration, brightness and "
    "face coverage."
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
    if report.get("shortcut_baseline"):
        lines += [*shortcut_markdown(report["shortcut_baseline"], report["methods"]), ""]

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
        for title, key in (("Manipulation", "per_manipulation"), ("Family", "per_family")):
            if key in m:
                lines += [*breakdown_table(title, m[key]), ""]
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


def shortcut_baseline(
    methods: Mapping[str, Mapping],
    predictions: Mapping[str, Sequence[Prediction]],
    n_boot: int = 1000,
    seed: int = 0,
) -> dict | None:
    """ROC-AUC of pulse and combined models minus the shortcut model of the same family.

    Models are matched by training dataset and family through the feature set
    in their sidecars; None if no shortcut model has a pulse or combined partner.
    """
    by_key: dict[tuple, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for name, entry in methods.items():
        if entry.get("kind") == "trained":
            key = ((entry.get("training_dataset") or {}).get("name") or "", entry.get("family"))
            by_key[key][entry.get("feature_set") or "pulse"].append(name)
    comparisons, verdicts = [], []
    for (dataset, family), sets in sorted(by_key.items()):
        for base in sets.get("shortcut", []):
            above = {}
            for feature_set in ("pulse", "combined"):
                found = []
                for other in sets.get(feature_set, []):
                    delta = paired_auc_delta(predictions[base], predictions[other], n_boot, seed)
                    comparisons.append(
                        {
                            "training_dataset": dataset,
                            "family": family,
                            "shortcut_model": base,
                            "model": other,
                            "feature_set": feature_set,
                            **delta,
                        }
                    )
                    found.append(bool(delta["ci_95"] and delta["ci_95"][0] > 0))
                above[feature_set] = all(found) if found else None
            if above["pulse"] is not None or above["combined"] is not None:
                verdicts.append(
                    {
                        "training_dataset": dataset,
                        "family": family,
                        "shortcut_model": base,
                        "pulse_above_shortcut": above["pulse"],
                        "combined_above_shortcut": above["combined"],
                        "criterion_met": bool(above["pulse"] and above["combined"]),
                    }
                )
    if not comparisons:
        return None
    return {"criterion": SHORTCUT_CRITERION, "comparisons": comparisons, "verdicts": verdicts}


def shortcut_markdown(section: Mapping, methods: Mapping[str, Mapping]) -> list[str]:
    def interval(ci) -> str:
        return f"[{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci else "[n/a]"

    lines = [
        "## Shortcut baseline",
        "",
        'Does each pulse (or pulse + shortcut, "combined") model beat a model trained on '
        "non-physiological media properties only? ROC-AUC differences use the videos both "
        "models scored, with a paired group-level bootstrap interval.",
        "",
        f"**{section['criterion']}**",
        "",
        "| Family | Shortcut model | Shortcut ROC-AUC | Model | Set | ROC-AUC "
        "| Δ ROC-AUC vs shortcut [95% CI] | Videos |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in section["comparisons"]:
        base_m = methods[c["shortcut_model"]]["metrics"]
        other_m = methods[c["model"]]["metrics"]
        lines.append(
            f"| {c['family']} | {c['shortcut_model']} | {_with_ci(base_m, 'roc_auc')} "
            f"| {c['model']} | {c['feature_set']} | {_with_ci(other_m, 'roc_auc')} "
            f"| {_fmt(c['delta'])} {interval(c['ci_95'])} | {c['n']} |"
        )
    lines.append("")
    for v in section["verdicts"]:

        def word(flag) -> str:
            return "not evaluated" if flag is None else ("yes" if flag else "no")

        lines.append(
            f"- {v['family']} ({v['training_dataset']}): pulse above shortcut: "
            f"{word(v['pulse_above_shortcut'])}; combined above shortcut: "
            f"{word(v['combined_above_shortcut'])}; criterion "
            f"{'MET' if v['criterion_met'] else 'NOT MET'}."
        )
    return lines


def breakdown_table(title: str, entries: Mapping[str, Mapping]) -> list[str]:
    """Markdown rows for a per-manipulation or per-family breakdown."""
    lines = [
        f"| {title} | Videos | Coverage | Inconclusive | Failed | Accuracy | Recall "
        "| ROC-AUC vs all real |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, e in entries.items():
        ci = e.get("ci_95") or {}

        def with_ci(metric: str, ci_key: str, e=e, ci=ci) -> str:
            interval = ci.get(ci_key)
            value = _fmt(e.get(metric))
            return f"{value} [{_fmt(interval[0])}, {_fmt(interval[1])}]" if interval else value

        has_recall = name != "none" and e.get("recall") is not None
        recall = with_ci("recall", "recall") if has_recall else "-"
        lines.append(
            f"| {name} | {e['n']} | {_fmt(e['coverage'])} | {_fmt(e['inconclusive_rate'])} "
            f"| {_fmt(e['failure_rate'])} | {_fmt(e['accuracy'])} | {recall} "
            f"| {with_ci('roc_auc_vs_all_real', 'roc_auc_vs_all_real')} |"
        )
    return lines


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
    extra: Mapping[str, Mapping[str, str]] | None = None,
    family_of: Mapping[str, str] | None = None,
    heuristic_gate: str = "default",
) -> tuple[dict, dict[str, list[Prediction]]]:
    """Score trained models and the heuristic on one split of a feature table.

    For the test split each model's sidecar records the evaluation; a second
    test evaluation of the same model raises AlreadyEvaluatedError unless
    `force` is set, and the report then flags it. split="all" scores every
    assigned row, for a table from a dataset the models were not trained on.
    `extra` columns (manifest, media table) are joined onto the rows, and
    `family_of` adds a per-family breakdown. A model whose features are not
    columns of the table raises ValueError instead of being imputed blindly.
    """
    table_path = Path(table_path)
    data = load_table(table_path, extra)  # raises LeakageError
    if split == ALL_SPLITS:
        rows = [r for s in data.rows.values() for r in s]
        failed = [r for s in data.failed.values() for r in s]
    else:
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
    columns = set().union(*(r.keys() for r in [*rows, *failed]))
    for path, model in models:
        absent = [f for f in model.features if f not in columns]
        if absent:
            raise ValueError(
                f"{path.name} needs columns the feature table lacks: {absent[:5]} "
                "(join them with --extra-columns, or use a model with matching features)"
            )
    dataset = table_meta.get("dataset", {})
    trained_on_datasets = {
        path.name: (model.sidecar.get("training_dataset") or {}).get("name")
        for path, model in models
    }
    cross = {
        name: trained
        for name, trained in trained_on_datasets.items()
        if trained and dataset.get("dataset") and trained != dataset.get("dataset")
    }
    for name, trained in cross.items():
        table_meta.setdefault("warnings", []).append(
            f"CROSS-DATASET: {name} was trained on {trained} and is evaluated on "
            f"{dataset.get('dataset')}"
        )
    for path, model in models:
        trained_on = model.sidecar.get("data_version", {}).get("feature_table_sha256")
        if trained_on and trained_on != table_sha and path.name not in cross:
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
            "feature_set": (model.sidecar.get("features") or {}).get("set", "pulse"),
            "hyperparameters": model.sidecar.get("hyperparameters"),
            "decision": model.sidecar.get("decision"),
            "training_dataset": model.sidecar.get("training_dataset"),
            "prior_test_evaluations": prior[path],
        }
    for name, preds in predictions.items():
        methods[name]["metrics"] = evaluate_predictions(preds, n_boot, seed, family_of)
    shortcut = shortcut_baseline(methods, predictions, n_boot, seed)

    all_rows = [*rows, *failed]
    reasons = dict(Counter(r["failure_reason"] for r in failed))
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
            "models_trained_on": trained_on_datasets,
            "cross_dataset": bool(cross),
            "heuristic_quality_gate": heuristic_gate,
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
        "shortcut_baseline": shortcut,
        "runtime": {
            "extraction": runtime_summary([float(r["runtime_s"] or "nan") for r in all_rows]),
            "videos": len(all_rows),
            "failed": len(failed),
            "failure_rate": len(failed) / len(all_rows) if all_rows else None,
            "failure_reasons": reasons,
            "prediction_s": timings,
        },
    }

    if record and split in ("test", ALL_SPLITS):
        for path, model in models:
            model.sidecar.setdefault("test_evaluations", []).append(
                {
                    "split": split,
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

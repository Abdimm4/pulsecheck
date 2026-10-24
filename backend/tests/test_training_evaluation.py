import csv
import json
import subprocess
import sys

import numpy as np
import pytest
from ffpp_fixtures import make_feature_table

from pulsecheck.config import REPO_ROOT
from pulsecheck.dataset import LeakageError
from pulsecheck.dataset.table import read_table
from pulsecheck.models.evaluation import (
    FAILED,
    Prediction,
    binary_metrics,
    bootstrap_ci,
    evaluate_predictions,
)
from pulsecheck.models.report import (
    AlreadyEvaluatedError,
    heuristic_predictions,
    render_markdown,
    run_evaluation,
    write_report,
)
from pulsecheck.models.training import (
    QualityGate,
    TrainConfig,
    choose_margin,
    load_model,
    load_table,
    provenance_from_table,
    train_models,
    youden_threshold,
)


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    path = tmp_path_factory.mktemp("table") / "features.csv"
    return make_feature_table(path, failed_ids=("original_090",))


@pytest.fixture
def trained(tmp_path, table, _trained_once):
    """Models trained once per module, saved to a fresh folder per test."""
    data, models = _trained_once
    paths = [m.save(tmp_path / "models" / f"{m.sidecar['model_name']}.pkl")[0] for m in models]
    return data, models, paths


@pytest.fixture(scope="module")
def _trained_once(table):
    data = load_table(table)
    config = TrainConfig(n_boot=50, forest_trees=50)
    return data, train_models(data, config, provenance_from_table(table))


def test_binary_metrics_by_hand():
    y = np.array([1, 1, 1, 0, 0, 0, 0])
    pred = np.array([1, 1, 0, 0, 0, 0, 1])
    m = binary_metrics(y, pred)
    assert m["confusion_matrix"] == {"tn": 3, "fp": 1, "fn": 1, "tp": 2}
    assert m["accuracy"] == pytest.approx(5 / 7)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(3 / 4)
    assert m["f1"] == pytest.approx(2 / 3)
    assert binary_metrics(np.array([0, 0]), np.array([0, 0]))["precision"] is None


def _pred(i, label, state, score=None, method=None, group=None):
    return Prediction(
        f"v{i}",
        label,
        method or ("Deepfakes" if label else "none"),
        group or f"g{i // 2}",
        state,
        score,
    )


def test_coverage_inconclusive_and_failure_rates():
    preds = [
        _pred(0, 1, "more_consistent_with_manipulated", 0.9),
        _pred(1, 0, "more_consistent_with_real", 0.1),
        _pred(2, 1, "inconclusive", 0.5),
        _pred(3, 0, FAILED),
    ]
    m = evaluate_predictions(preds, n_boot=20)
    assert m["n"] == 4 and m["decided"] == 2 and m["failed"] == 1
    assert m["coverage"] == pytest.approx(0.5)
    assert m["inconclusive_rate"] == pytest.approx(1 / 3)
    assert m["failure_rate"] == pytest.approx(0.25)
    assert m["accuracy"] == 1.0
    assert m["roc_auc"] == 1.0  # over the three scored videos
    assert set(m["per_manipulation"]) == {"Deepfakes", "none"}
    assert m["per_manipulation"]["Deepfakes"]["roc_auc_vs_all_real"] == 1.0


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    preds = []
    for i in range(200):
        label = i % 2
        score = float(np.clip(0.5 + (0.2 if label else -0.2) + 0.25 * rng.normal(), 0, 1))
        state = "more_consistent_with_manipulated" if score >= 0.5 else "more_consistent_with_real"
        preds.append(_pred(i, label, state, score))
    point = evaluate_predictions(preds, n_boot=0)
    ci = bootstrap_ci(preds, n_boot=300, seed=1)
    for name in ("accuracy", "roc_auc", "recall", "specificity"):
        lo, hi = ci[name]
        assert lo <= point[name] <= hi
        assert hi - lo < 0.3
    assert ci["coverage"] == [1.0, 1.0]


def test_threshold_and_margin_helpers():
    y = np.array([0, 0, 0, 1, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t = youden_threshold(y, score)
    assert 0.3 < t < 0.7
    assert choose_margin(score, 0.5, min_coverage=1.0, max_margin=0.25) == pytest.approx(0.2)
    assert choose_margin(score, 0.5, min_coverage=0.5, max_margin=0.25) == pytest.approx(0.25)


def test_training_separates_synthetic_classes_and_tags_provenance(trained):
    data, models, paths = trained
    assert {m.sidecar["family"] for m in models} == {"logreg", "forest"}
    for model, path in zip(models, paths, strict=True):
        side = json.loads(path.with_suffix(".json").read_text())
        assert side["training_dataset"]["name"] == "FaceForensics++"
        assert side["training_dataset"]["license"] == "synthetic test data"
        assert side["training_dataset"]["non_commercial_use_only"] is True
        assert "non-commercial" in side["project_status"]
        assert side["data_version"]["feature_table_sha256"]
        assert side["decision"]["calibrated"] is False
        # The always-missing feature is dropped; partly missing ones are kept.
        assert "right_cheek.method_diff_hz" in side["features"]["dropped"]
        assert "forehead__left_cheek.correlation" in side["features"]["used"]
        assert side["metrics"]["val"]["roc_auc"] > 0.8
        assert set(side["metrics"]) == {"train", "val"}  # test untouched
        assert side["test_evaluations"] == []
        assert side["split_sizes"]["train"]["failed"] == 0
        # Reload and predict identically.
        loaded = load_model(path)
        rows = data.rows["val"]
        assert np.allclose(loaded.scores(rows), model.scores(rows))


def test_preprocessing_is_fitted_on_train_only(trained):
    data, models, _ = trained
    logreg = next(m for m in models if m.sidecar["family"] == "logreg")
    imputer = logreg.pipeline.named_steps["impute"]
    train_x = data.matrix("train", logreg.features)
    assert np.allclose(imputer.statistics_, np.nanmedian(train_x, axis=0))
    scaler = logreg.pipeline.named_steps["scale"]
    assert np.allclose(scaler.mean_, imputer.transform(train_x).mean(axis=0))


def test_quality_gate_and_band_make_inconclusive(trained):
    data, models, _ = trained
    model = models[0]
    row = dict(data.rows["val"][0])
    row["signal_quality"] = "0.1"
    pred = model.predict([row])[0]
    assert pred.state == "inconclusive" and pred.reason == "low_signal_quality"
    assert QualityGate().reason({"signal_quality": "", "usable_roi_count": "3"}) == (
        "low_signal_quality"
    )


def test_leaky_feature_table_fails_loudly(tmp_path, table):
    rows = read_table(table)
    with table.open(newline="") as handle:
        header = next(csv.reader(handle))
    for row in rows:
        if row["video_id"] == "Deepfakes_001_000":
            row["split"] = "test" if row["split"] != "test" else "train"
    leaky = tmp_path / "leaky.csv"
    with leaky.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(LeakageError):
        load_table(leaky)


def test_training_needs_both_classes(tmp_path):
    table = make_feature_table(tmp_path / "f.csv", methods=())
    with pytest.raises(ValueError, match="both real and fake"):
        train_models(load_table(table), TrainConfig(n_boot=0), {})


def test_heuristic_baseline_on_table(table):
    data = load_table(table)
    preds = heuristic_predictions(data.rows["test"], data.failed["test"])
    assert len(preds) == len(data.rows["test"]) + len(data.failed["test"])
    states = {p.state for p in preds}
    assert states <= {
        "more_consistent_with_real",
        "more_consistent_with_manipulated",
        "inconclusive",
        FAILED,
    }
    assert any(p.score is not None for p in preds)


def test_evaluation_report_and_single_test_evaluation(tmp_path, table, trained):
    _, _, paths = trained
    meta = json.loads((table.parent / (table.name + ".json")).read_text())
    report, preds = run_evaluation(table, paths, n_boot=50, table_meta=meta)
    cond = report["evaluation_conditions"]
    assert cond["split"] == "test" and cond["dataset"] == "FaceForensics++"
    assert cond["videos"] == 10 * 6
    assert set(report["methods"]) == {"heuristic", "ffpp-c23-logreg", "ffpp-c23-forest"}
    for entry in report["methods"].values():
        m = entry["metrics"]
        assert m["n"] == 60
        for key in (
            "accuracy",
            "precision",
            "recall",
            "specificity",
            "f1",
            "roc_auc",
            "confusion_matrix",
            "per_manipulation",
            "confidence_intervals_95",
            "coverage",
            "inconclusive_rate",
            "failure_rate",
        ):
            assert key in m
    assert report["methods"]["ffpp-c23-logreg"]["metrics"]["roc_auc"] > 0.8
    assert report["runtime"]["extraction"]["n"] == 60
    assert "general accuracy claim" in report["disclaimer"]

    json_path, md_path = write_report(report, tmp_path / "reports", "eval")
    json.loads(json_path.read_text())  # strict JSON
    md = md_path.read_text()
    assert "Evaluation conditions" in md and "not a general accuracy claim" in md
    assert "| Deepfakes |" in md

    # The test split is evaluated once; a re-run needs force and is flagged.
    with pytest.raises(AlreadyEvaluatedError):
        run_evaluation(table, paths, n_boot=10, table_meta=meta)
    report2, _ = run_evaluation(table, paths, n_boot=10, table_meta=meta, force=True)
    assert report2["methods"]["ffpp-c23-logreg"]["prior_test_evaluations"] == 1
    assert "WARNING" in render_markdown(report2)
    side = json.loads(paths[0].with_suffix(".json").read_text())
    assert [e["forced_rerun"] for e in side["test_evaluations"]] == [False, True]


def test_failed_test_videos_count_against_coverage(tmp_path):
    table = make_feature_table(tmp_path / "f.csv", failed_ids=("original_080", "Deepfakes_080_081"))
    report, _ = run_evaluation(table, [], n_boot=0, record=False)
    m = report["methods"]["heuristic"]["metrics"]
    assert m["failed"] == 2
    assert m["failure_reasons"] == {"video_decode_error": 2}
    assert report["runtime"]["failed"] == 2


def test_train_and_evaluate_cli(tmp_path, table):
    scripts = REPO_ROOT / "scripts"
    models_dir = tmp_path / "models"
    done = subprocess.run(
        [
            sys.executable,
            str(scripts / "train_model.py"),
            str(table),
            "--out-dir",
            str(models_dir),
            "--n-boot",
            "20",
            "--forest-trees",
            "50",
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    pkls = sorted(models_dir.glob("*.pkl"))
    assert len(pkls) == 2
    reports = tmp_path / "reports"
    cmd = [
        sys.executable,
        str(scripts / "evaluate.py"),
        str(table),
        "--out-dir",
        str(reports),
        "--n-boot",
        "20",
    ]
    for p in pkls:
        cmd += ["--model", str(p)]
    done = subprocess.run(cmd, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert len(list(reports.glob("*.md"))) == 1
    assert len(list(reports.glob("*.predictions.csv"))) == 1
    done = subprocess.run(cmd, capture_output=True, text=True)
    assert done.returncode == 1 and "already evaluated" in done.stderr

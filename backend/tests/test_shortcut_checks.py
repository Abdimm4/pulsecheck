import csv
import subprocess
import sys

import numpy as np
import pytest
from ffpp_fixtures import make_feature_table

from pulsecheck.config import REPO_ROOT
from pulsecheck.dataset.confounds import audit, cv_r2, single_feature_auc, spearman
from pulsecheck.dataset.media import MEDIA_COLUMNS, SHORTCUT_FEATURES
from pulsecheck.dataset.table import read_columns, read_table
from pulsecheck.features import FEATURE_NAMES, PULSE_CLASSIFIER_FEATURES, TRACKING_FEATURES
from pulsecheck.models.evaluation import Prediction, paired_auc_delta
from pulsecheck.models.report import render_markdown, run_evaluation, shortcut_baseline
from pulsecheck.models.training import (
    FEATURE_SETS,
    MODEL_VERSION,
    TrainConfig,
    load_table,
    provenance_from_table,
    train_models,
)


def test_pulse_classifier_features_leave_out_tracking():
    assert set(TRACKING_FEATURES) <= set(FEATURE_NAMES)
    assert not set(TRACKING_FEATURES) & set(PULSE_CLASSIFIER_FEATURES)
    assert set(PULSE_CLASSIFIER_FEATURES) | set(TRACKING_FEATURES) == set(FEATURE_NAMES)
    for name in ("face_coverage", "usable_roi_count", "forehead.tracking_coverage"):
        assert name in TRACKING_FEATURES
    assert "signal_quality" in PULSE_CLASSIFIER_FEATURES
    assert TrainConfig().candidate_features == PULSE_CLASSIFIER_FEATURES
    assert FEATURE_SETS["pulse-full"] == FEATURE_NAMES
    combined = FEATURE_SETS["combined"]
    assert set(combined) == set(PULSE_CLASSIFIER_FEATURES) | set(SHORTCUT_FEATURES)
    with pytest.raises(ValueError, match="unknown feature set"):
        TrainConfig(feature_set="nope")


def _media_table(path, table, seed=0, informative=True):
    """Media columns per video; bitrate tracks the label when `informative`."""
    rng = np.random.default_rng(seed)
    rows = []
    for row in read_table(table):
        fake = row["label"] == "fake"
        rows.append(
            {
                "video_id": row["video_id"],
                "width": 1280.0 if informative and fake else 854.0,
                "height": 720.0 if informative and fake else 480.0,
                "container_fps": 30.0,
                "container_duration_s": 10.0,
                "bitrate_kbps": float(rng.normal(2000 if informative and fake else 1000, 50)),
                "mean_brightness": float(rng.normal(110, 10)),
                "probe_error": "",
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MEDIA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture(scope="module")
def shortcut_setup(tmp_path_factory):
    folder = tmp_path_factory.mktemp("shortcut")
    table = make_feature_table(folder / "features.csv")
    media = _media_table(folder / "media.csv", table)
    extra = read_columns(media)
    data = load_table(table, extra)
    provenance = provenance_from_table(table)
    paths = []
    for feature_set in ("pulse", "shortcut", "combined"):
        config = TrainConfig(n_boot=20, forest_trees=30, quality_gate=None, feature_set=feature_set)
        for model in train_models(data, config, provenance):
            name = model.sidecar["model_name"]
            paths.append(model.save(folder / "models" / f"{name}.pkl")[0])
    return table, extra, paths


def test_sidecar_records_feature_set_and_candidates(shortcut_setup):
    import json

    _, _, paths = shortcut_setup
    sides = {p.stem: json.loads(p.with_suffix(".json").read_text()) for p in paths}
    pulse = sides["ffpp-c23-logreg"]
    assert pulse["model_version"] == MODEL_VERSION == "0.2.0"
    assert pulse["features"]["set"] == "pulse"
    assert pulse["features"]["candidate_names"] == list(PULSE_CLASSIFIER_FEATURES)
    assert not set(pulse["features"]["used"]) & set(TRACKING_FEATURES)
    assert sides["ffpp-c23-forest-shortcut"]["features"]["candidate_names"] == list(
        SHORTCUT_FEATURES
    )
    assert "bitrate_kbps" in sides["ffpp-c23-logreg-combined"]["features"]["used"]


def _preds(y, score, groups=None):
    return [
        Prediction(f"v{i}", int(label), "m", groups[i] if groups else f"g{i}", "x", float(s))
        for i, (label, s) in enumerate(zip(y, score, strict=True))
    ]


def test_paired_auc_delta():
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 60)
    perfect = y + rng.uniform(0, 0.5, len(y))
    noise = rng.uniform(0, 1, len(y))
    same = paired_auc_delta(_preds(y, noise), _preds(y, noise), n_boot=50)
    assert same["delta"] == 0 and same["ci_95"] == [0.0, 0.0]
    better = paired_auc_delta(_preds(y, noise), _preds(y, perfect), n_boot=200)
    assert better["other_roc_auc"] == 1.0
    assert better["delta"] == pytest.approx(1.0 - better["baseline_roc_auc"])
    assert 0 < better["ci_95"][0] <= better["delta"] <= better["ci_95"][1]
    worse = paired_auc_delta(_preds(y, perfect), _preds(y, noise), n_boot=200)
    assert worse["delta"] == pytest.approx(-better["delta"])
    assert worse["ci_95"][1] < 0
    # Videos one method did not score are left out of both.
    unscored = _preds(y, noise)
    unscored[0].score = None
    assert paired_auc_delta(_preds(y, noise), unscored, n_boot=0)["n"] == len(y) - 1


def test_shortcut_baseline_pairs_models_by_family():
    rng = np.random.default_rng(1)
    y = np.array([0, 1] * 50)
    strong = y + rng.uniform(0, 0.3, len(y))
    weak = rng.uniform(0, 1, len(y))

    def entry(family, feature_set):
        return {
            "kind": "trained",
            "family": family,
            "feature_set": feature_set,
            "training_dataset": {"name": "D"},
        }

    methods = {
        "heuristic": {"kind": "heuristic"},
        "lr-short": entry("logreg", "shortcut"),
        "lr-pulse": entry("logreg", "pulse"),
        "lr-comb": entry("logreg", "combined"),
        "rf-short": entry("forest", "shortcut"),
        "rf-pulse": entry("forest", "pulse"),
    }
    preds = {
        "heuristic": _preds(y, weak),
        "lr-short": _preds(y, weak),
        "lr-pulse": _preds(y, strong),
        "lr-comb": _preds(y, strong),
        "rf-short": _preds(y, strong),
        "rf-pulse": _preds(y, weak),
    }
    section = shortcut_baseline(methods, preds, n_boot=100)
    assert {(c["shortcut_model"], c["model"]) for c in section["comparisons"]} == {
        ("lr-short", "lr-pulse"),
        ("lr-short", "lr-comb"),
        ("rf-short", "rf-pulse"),
    }
    verdicts = {v["family"]: v for v in section["verdicts"]}
    assert verdicts["logreg"]["criterion_met"] is True
    assert verdicts["forest"]["pulse_above_shortcut"] is False
    assert verdicts["forest"]["combined_above_shortcut"] is None
    assert verdicts["forest"]["criterion_met"] is False
    assert shortcut_baseline({"lr-pulse": methods["lr-pulse"]}, preds) is None


def test_evaluation_report_has_shortcut_section(shortcut_setup):
    table, extra, paths = shortcut_setup
    report, _ = run_evaluation(table, paths, n_boot=20, extra=extra, record=False)
    section = report["shortcut_baseline"]
    assert len(section["comparisons"]) == 4
    assert {v["family"] for v in section["verdicts"]} == {"logreg", "forest"}
    # The synthetic media columns give the label away, so nothing beats the shortcut.
    shortcut = report["methods"]["ffpp-c23-logreg-shortcut"]["metrics"]["roc_auc"]
    assert shortcut > 0.95
    assert not any(v["criterion_met"] for v in section["verdicts"])
    md = render_markdown(report)
    assert "## Shortcut baseline" in md and "pulse > shortcut and combined > shortcut" in md
    assert "NOT MET" in md


def test_single_feature_auc_and_spearman():
    y = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    out = single_feature_auc(x, y)
    assert out == {"n": 5, "roc_auc": 1.0, "separation": 1.0}
    flipped = single_feature_auc(-x, y)
    assert flipped["roc_auc"] == 0.0 and flipped["separation"] == 1.0
    a = np.arange(20, dtype=float)
    assert spearman(a, a**3) == pytest.approx(1.0)
    assert spearman(a, -a) == pytest.approx(-1.0)
    assert spearman(a, np.ones(20)) is None  # constant column
    assert spearman(a[:5], a[:5]) is None  # too few rows


def test_cv_r2_and_audit_flags():
    rng = np.random.default_rng(0)
    n = 300
    groups = [f"g{i // 3}" for i in range(n)]
    media = rng.normal(size=(n, 2))
    derived = 2 * media[:, 0] + 0.1 * rng.normal(size=n)  # almost a function of the media
    unrelated = rng.normal(size=n)
    assert cv_r2(media, derived, groups, trees=50) > 0.8
    assert cv_r2(media, unrelated, groups, trees=50) < 0.1
    assert cv_r2(media, np.full(n, np.nan), groups) is None

    y = (media[:, 1] > 0).astype(int)
    columns = {"m0": media[:, 0], "m1": media[:, 1], "derived": derived, "unrelated": unrelated}
    result = audit(
        columns,
        y,
        groups,
        ["derived", "unrelated"],
        shortcut_features=["m0", "m1"],
        media_features=["m0", "m1"],
        trees=50,
    )
    assert result["flagged"] == ["derived"]
    assert result["pulse_features"]["derived"]["max_abs_spearman"] > 0.9
    assert result["shortcut_auc"]["m1"]["roc_auc"] == 1.0
    assert result["shortcut_auc"]["m0"]["separation"] < 0.65


def test_audit_cli(tmp_path):
    table = make_feature_table(tmp_path / "features.csv")
    media = _media_table(tmp_path / "media.csv", table)
    out = tmp_path / "reports"
    done = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "audit_confounds.py"),
            str(table),
            "--media",
            str(media),
            "--out-dir",
            str(out),
            "--trees",
            "20",
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    md = next(out.glob("confound-audit-train-*.md")).read_text(encoding="utf-8")
    assert "## (a) Shortcut columns alone" in md and "| bitrate_kbps |" in md
    assert len(list(out.glob("*.json"))) == 1

"""Trained classifiers on the per-video feature table (PRD FR-7, item 2).

Two families: logistic regression (median imputation + missing-value
indicators + standard scaling) and a random forest (median imputation +
indicators). Protocol:

- the feature table is checked for split leakage first (LeakageError);
- features missing in more than `max_missing_fraction` of training videos
  are dropped; every other missing value is imputed with the training
  median and flagged by an indicator column, so missingness is explicit;
- all preprocessing is fitted on the training split only;
- hyperparameters are chosen by validation ROC-AUC, then the decision
  threshold (Youden's J on validation) and a symmetric inconclusive band
  around it (the widest band that keeps validation coverage at or above
  `min_val_coverage`) are fitted on validation;
- small datasets can instead use grouped cross-validation inside the train
  split (`train_family_cv`, rows carry a `cv_fold`): hyperparameters, the
  threshold and the band are chosen on pooled out-of-fold scores, then the
  chosen pipeline is refitted on the whole train split;
- the test split is never touched here; scripts/evaluate.py scores it once.

Feature sets (FEATURE_SETS, `TrainConfig.features`): "pulse" (the default)
uses PULSE_CLASSIFIER_FEATURES, i.e. every table feature except the tracking
and coverage ones, which depend on resolution and compression and stay with
the quality gate; "shortcut" uses only non-physiological media properties
(pulsecheck.dataset.media); "combined" uses both, to test whether the pulse
features add anything over the shortcut; "pulse-full" is every table feature
(the model 0.1.0 default).

Scores are the classifier's fake-class output. They are not calibrated
probabilities, and the states describe consistency with the training data,
not whether a video is authentic. A quality gate (the heuristic's thresholds
by default) marks low-quality videos inconclusive regardless of the score.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pulsecheck import __version__
from pulsecheck.dataset.manifest import (
    SPLITS,
    SplitRecord,
    check_leakage,
    file_sha256,
    read_sidecar,
)
from pulsecheck.dataset.media import SHORTCUT_FEATURES
from pulsecheck.dataset.table import (
    STATUS_FAILED,
    STATUS_OK,
    attach_columns,
    parse_float,
    read_table,
    split_records,
)
from pulsecheck.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, PULSE_CLASSIFIER_FEATURES
from pulsecheck.models.evaluation import (
    FAILED,
    FAKE_STATE,
    INCONCLUSIVE_STATE,
    REAL_STATE,
    Prediction,
    evaluate_predictions,
    safe_auc,
)

# 0.2.0: the default "pulse" feature set leaves out tracking/coverage features.
MODEL_VERSION = "0.2.0"
FAMILIES = ("logreg", "forest")
PROJECT_STATUS = (
    "PulseCheck is a free, non-commercial research and educational project "
    "(PRD v2.3 Section 29, LG-6). Models trained on non-commercial datasets must "
    "not be used commercially."
)
FEATURE_SETS = {
    "pulse": PULSE_CLASSIFIER_FEATURES,
    "shortcut": SHORTCUT_FEATURES,
    "combined": (*PULSE_CLASSIFIER_FEATURES, *SHORTCUT_FEATURES),
    "pulse-full": FEATURE_NAMES,
}
SCORE_SEMANTICS = (
    "Classifier output for the manipulated class, higher = more consistent with the "
    "manipulated training examples. Not a calibrated probability."
)


@dataclass(frozen=True)
class QualityGate:
    min_signal_quality: float = 0.35
    min_usable_rois: int = 2
    min_face_coverage: float = 0.8

    @classmethod
    def from_heuristic(cls, values: Mapping | None) -> QualityGate:
        values = values or {}
        return cls(**{k: values[k] for k in asdict(cls()) if k in values})

    def reason(self, row: Mapping[str, str]) -> str | None:
        quality = parse_float(row.get("signal_quality", "")) or 0.0
        usable = parse_float(row.get("usable_roi_count", "")) or 0.0
        coverage = parse_float(row.get("face_coverage", "")) or 0.0
        if quality < self.min_signal_quality:
            return "low_signal_quality"
        if usable < self.min_usable_rois:
            return "too_few_usable_rois"
        if coverage < self.min_face_coverage:
            return "low_face_coverage"
        return None


def resolve_gate(spec: str, heuristic: Mapping | None) -> QualityGate | None:
    """Quality gate from a CLI spec: "default" (the config's heuristic gate), "off", or
    the path of a JSON file with QualityGate fields (e.g. a proposed gate)."""
    if spec == "off":
        return None
    if spec == "default":
        return QualityGate.from_heuristic(heuristic)
    values = json.loads(Path(spec).read_text(encoding="utf-8"))
    values = values.get("gate", values)
    unknown = set(values) - set(asdict(QualityGate()))
    if unknown:
        raise ValueError(f"{spec}: unknown quality-gate settings {sorted(unknown)}")
    return QualityGate(**values)


@dataclass
class TrainConfig:
    families: tuple[str, ...] = FAMILIES
    max_missing_fraction: float = 0.95
    min_val_coverage: float = 0.8
    max_margin: float = 0.25
    quality_gate: QualityGate | None = field(default_factory=QualityGate)
    seed: int = 0
    n_boot: int = 1000
    forest_trees: int = 300
    features: tuple[str, ...] | None = None  # candidate features; None = FEATURE_SETS[feature_set]
    feature_set: str = "pulse"  # name recorded in the sidecar and model name

    def __post_init__(self) -> None:
        unknown = set(self.families) - set(FAMILIES)
        if unknown:
            raise ValueError(f"unknown model families {sorted(unknown)}; known: {FAMILIES}")
        if not 0 <= self.min_val_coverage <= 1:
            raise ValueError("min_val_coverage must be in [0, 1]")
        if not self.features and self.feature_set not in FEATURE_SETS:
            raise ValueError(
                f"unknown feature set {self.feature_set!r}; known: {sorted(FEATURE_SETS)}"
            )

    @property
    def candidate_features(self) -> tuple[str, ...]:
        return tuple(self.features) if self.features else FEATURE_SETS[self.feature_set]


def candidate_grid(family: str, seed: int, trees: int = 300) -> list[tuple[dict, Pipeline]]:
    """(params, unfitted pipeline) per candidate, simplest first (ties keep the first)."""

    def imputer() -> SimpleImputer:
        return SimpleImputer(strategy="median", add_indicator=True)

    out = []
    if family == "logreg":
        for c in (0.01, 0.1, 1.0, 10.0):
            pipe = Pipeline(
                [
                    ("impute", imputer()),
                    ("scale", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(C=c, class_weight="balanced", max_iter=5000),
                    ),
                ]
            )
            out.append(({"C": c}, pipe))
    elif family == "forest":
        for depth in (4, 8, None):
            for leaf in (20, 5, 1):
                clf = RandomForestClassifier(
                    n_estimators=trees,
                    max_depth=depth,
                    min_samples_leaf=leaf,
                    class_weight="balanced_subsample",
                    random_state=seed,
                    n_jobs=-1,
                )
                pipe = Pipeline([("impute", imputer()), ("clf", clf)])
                out.append(({"max_depth": depth, "min_samples_leaf": leaf}, pipe))
    else:
        raise ValueError(f"unknown family {family!r}")
    return out


@dataclass
class TableData:
    """Feature-table rows grouped by split; failed rows are kept for reporting."""

    rows: dict[str, list[dict[str, str]]]  # split -> ok rows
    failed: dict[str, list[dict[str, str]]]  # split -> failed rows
    leakage: dict

    def matrix(self, split: str, features: Sequence[str]) -> np.ndarray:
        return feature_matrix(self.rows[split], features)

    def labels(self, split: str) -> np.ndarray:
        return labels(self.rows[split])


def feature_matrix(rows: Sequence[Mapping[str, str]], features: Sequence[str]) -> np.ndarray:
    x = np.full((len(rows), len(features)), np.nan)
    for i, row in enumerate(rows):
        for j, name in enumerate(features):
            value = parse_float(row.get(name, ""))
            if value is not None:
                x[i, j] = value
    return x


def labels(rows: Sequence[Mapping[str, str]]) -> np.ndarray:
    bad = {r["label"] for r in rows} - {"real", "fake"}
    if bad:
        raise ValueError(f"unknown labels {sorted(bad)}")
    return np.array([r["label"] == "fake" for r in rows], dtype=int)


def load_table(path: str | Path, extra: Mapping[str, Mapping[str, str]] | None = None) -> TableData:
    """Read the table, check leakage over every assigned row, group rows by split.

    `extra` (video_id -> columns, e.g. from a manifest or media table) is
    joined onto the rows first; see attach_columns.
    """
    all_rows = read_table(path)
    if extra:
        all_rows = attach_columns(all_rows, extra)
    assigned = [r for r in all_rows if r["split"] in SPLITS]
    leakage = check_leakage(split_records(assigned))  # raises LeakageError

    def pick(split: str, status: str) -> list[dict[str, str]]:
        return [r for r in assigned if r["split"] == split and r["status"] == status]

    rows = {s: pick(s, STATUS_OK) for s in SPLITS}
    failed = {s: pick(s, STATUS_FAILED) for s in SPLITS}
    return TableData(rows, failed, leakage)


def youden_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Threshold maximising TPR - FPR; ties go to the one closest to 0.5."""
    values = np.unique(score)
    cuts = np.concatenate([[values[0] - 1e-9], (values[:-1] + values[1:]) / 2, [values[-1]]])
    best, best_j = 0.5, -np.inf
    for t in cuts:
        pred = score >= t
        tpr = np.mean(pred[y == 1]) if np.any(y == 1) else 0.0
        fpr = np.mean(pred[y == 0]) if np.any(y == 0) else 0.0
        j = tpr - fpr
        if j > best_j + 1e-12 or (abs(j - best_j) <= 1e-12 and abs(t - 0.5) < abs(best - 0.5)):
            best, best_j = float(t), j
    return float(np.clip(best, 0.0, 1.0))


def choose_margin(
    score: np.ndarray, threshold: float, min_coverage: float, max_margin: float
) -> float:
    """Widest symmetric band around the threshold keeping coverage >= min_coverage."""
    best = 0.0
    for margin in np.round(np.arange(0.01, max_margin + 1e-9, 0.01), 4):
        coverage = np.mean(np.abs(score - threshold) >= margin - 1e-9) if score.size else 0.0
        if coverage < min_coverage:  # coverage only falls as the band widens
            break
        best = float(margin)
    return best


@dataclass
class TrainedModel:
    pipeline: Pipeline
    sidecar: dict
    # Out-of-fold predictions on the train split (CV training only; not saved).
    oof: list[Prediction] = field(default_factory=list)

    @property
    def features(self) -> list[str]:
        return self.sidecar["features"]["used"]

    @property
    def threshold(self) -> float:
        return self.sidecar["decision"]["threshold"]

    @property
    def margin(self) -> float:
        return self.sidecar["decision"]["inconclusive_margin"]

    @property
    def gate(self) -> QualityGate | None:
        gate = self.sidecar["decision"]["quality_gate"]
        return QualityGate(**gate) if gate else None

    def scores(self, rows: Sequence[Mapping[str, str]]) -> np.ndarray:
        if not rows:
            return np.zeros(0)
        return self.pipeline.predict_proba(feature_matrix(rows, self.features))[:, 1]

    def predict(
        self, rows: Sequence[Mapping[str, str]], failed: Sequence[Mapping[str, str]] = ()
    ) -> list[Prediction]:
        """One Prediction per row; failed rows become state 'failed'."""
        return self.predict_scores(rows, self.scores(rows), failed)

    def predict_scores(
        self,
        rows: Sequence[Mapping[str, str]],
        scores: np.ndarray,
        failed: Sequence[Mapping[str, str]] = (),
    ) -> list[Prediction]:
        """Predictions from given scores (e.g. out-of-fold) with this model's decision rule."""
        out = []
        for row, score in zip(rows, scores, strict=True):
            state, reason = self.decide(row, float(score))
            forced = int(score >= self.threshold)
            out.append(_prediction(row, state, float(score), forced, reason))
        out.extend(_prediction(r, FAILED, None, None, r["failure_reason"]) for r in failed)
        return out

    def decide(self, row: Mapping[str, str], score: float) -> tuple[str, str]:
        """Result state and inconclusive reason ('' if decided) for one scored row."""
        gate = self.gate
        reason = gate.reason(row) if gate else None
        if reason:
            return INCONCLUSIVE_STATE, reason
        if score >= self.threshold + self.margin - 1e-9:
            return FAKE_STATE, ""
        if score <= self.threshold - self.margin + 1e-9:
            return REAL_STATE, ""
        return INCONCLUSIVE_STATE, "score_in_inconclusive_band"

    def save(self, path: str | Path) -> tuple[Path, Path]:
        path = Path(path).with_suffix(".pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self.pipeline, handle)
        side = path.with_suffix(".json")
        self.sidecar["artifact"] = {"file": path.name, "sha256": file_sha256(path)}
        side.write_text(json.dumps(self.sidecar, indent=2, allow_nan=False), encoding="utf-8")
        return path, side


def _prediction(row, state, score, forced, reason) -> Prediction:
    return Prediction(
        video_id=row["video_id"],
        label=int(row["label"] == "fake"),
        manipulation_type=row["manipulation_type"],
        group_id=row["group_id"],
        state=state,
        score=score,
        forced=forced,
        reason=reason,
    )


def load_model(path: str | Path) -> TrainedModel:
    """Load a .pkl and its sidecar, checking the pickle's hash first.

    Only load pickles you created: unpickling runs code.
    """
    path = Path(path).with_suffix(".pkl")
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    expected = sidecar.get("artifact", {}).get("sha256")
    if expected and file_sha256(path) != expected:
        raise ValueError(f"{path} does not match the hash in its sidecar")
    with path.open("rb") as handle:
        pipeline = pickle.load(handle)  # noqa: S301 - local artifact verified above
    return TrainedModel(pipeline, sidecar)


def select_features(train_x: np.ndarray, names: Sequence[str], max_missing: float):
    missing = np.mean(np.isnan(train_x), axis=0) if train_x.size else np.ones(len(names))
    used = [n for n, m in zip(names, missing, strict=True) if m <= max_missing]
    dropped = {
        n: f"missing in {m:.0%} of training videos"
        for n, m in zip(names, missing, strict=True)
        if m > max_missing
    }
    return used, dropped


def train_family(
    data: TableData, family: str, config: TrainConfig, provenance: Mapping
) -> TrainedModel:
    for split in ("train", "val"):
        y = data.labels(split)
        if len(np.unique(y)) < 2:
            raise ValueError(f"the {split} split needs both real and fake videos (has {len(y)})")
    names = config.candidate_features
    used, dropped = select_features(data.matrix("train", names), names, config.max_missing_fraction)
    if not used:
        raise ValueError("no feature has enough training values")
    x_train, y_train = data.matrix("train", used), data.labels("train")
    x_val, y_val = data.matrix("val", used), data.labels("val")

    candidates = []
    best = None
    for params, pipe in candidate_grid(family, config.seed, config.forest_trees):
        pipe.fit(x_train, y_train)
        score = pipe.predict_proba(x_val)[:, 1]
        auc = safe_auc(y_val, score)
        candidates.append({"params": params, "val_roc_auc": auc})
        if best is None or (auc or 0) > (best[2] or 0) + 1e-12:
            best = (params, pipe, auc, score)
    params, pipe, _, val_score = best

    threshold = youden_threshold(y_val, val_score)
    margin = choose_margin(val_score, threshold, config.min_val_coverage, config.max_margin)
    sidecar = _sidecar(
        family,
        config,
        provenance,
        pipe,
        params,
        used,
        dropped,
        selection={"metric": "validation ROC-AUC", "candidates": candidates},
        threshold=threshold,
        margin=margin,
        fitted_on="validation",
    )
    sidecar["split_sizes"] = {
        s: {"ok": len(data.rows[s]), "failed": len(data.failed[s])} for s in SPLITS
    }
    sidecar["leakage_check"] = data.leakage
    model = TrainedModel(pipe, sidecar)
    sidecar["metrics"] = {
        split: evaluate_predictions(
            model.predict(data.rows[split], data.failed[split]), config.n_boot, config.seed
        )
        for split in ("train", "val")
    }
    sidecar["metrics_conditions"] = (
        "Train metrics are in-sample. Validation metrics are optimistic because "
        "hyperparameters, threshold and band were chosen on validation. Neither is a "
        "held-out estimate; see the test evaluation report."
    )
    return model


def _sidecar(
    family: str,
    config: TrainConfig,
    provenance: Mapping,
    pipe: Pipeline,
    params: dict,
    used: list[str],
    dropped: dict,
    selection: dict,
    threshold: float,
    margin: float,
    fitted_on: str,
) -> dict:
    slug = provenance.get("dataset_slug", "dataset")
    suffix = "" if config.feature_set == "pulse" else f"-{config.feature_set}"
    return {
        "model_name": f"{slug}-{family}{suffix}",
        "family": family,
        "model_version": MODEL_VERSION,
        "pulsecheck_version": __version__,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "training_dataset": dict(provenance.get("training_dataset", {})),
        "project_status": PROJECT_STATUS,
        "data_version": dict(provenance.get("data_version", {})),
        "features": {
            "set": config.feature_set,
            "schema_version": FEATURE_SCHEMA_VERSION,
            "candidates": len(config.candidate_features),
            "candidate_names": list(config.candidate_features),
            "used": used,
            "dropped": dropped,
            "missing_handling": (
                "median imputation fitted on train, plus a missing-indicator column for every "
                "feature with missing training values"
            ),
        },
        "preprocessing": [name for name, _ in pipe.steps[:-1]],
        "hyperparameters": params,
        "selection": selection,
        "decision": {
            "threshold": threshold,
            "threshold_rule": f"Youden's J on {fitted_on}",
            "inconclusive_margin": margin,
            "margin_rule": (
                f"widest symmetric band (step 0.01, max {config.max_margin}) keeping "
                f"{fitted_on} coverage >= {config.min_val_coverage}"
            ),
            "quality_gate": asdict(config.quality_gate) if config.quality_gate else None,
            "score_semantics": SCORE_SEMANTICS,
            "calibrated": False,
        },
        "seed": config.seed,
        "test_evaluations": [],
    }


def check_fold_leakage(rows: Sequence[Mapping[str, str]]) -> dict:
    """Raise LeakageError if a linked group spans CV folds (rows need `cv_fold`)."""
    missing = [r["video_id"] for r in rows if not r.get("cv_fold")]
    if missing:
        raise ValueError(f"{len(missing)} train rows have no cv_fold, e.g. {missing[0]}")
    return check_leakage(
        [
            SplitRecord(rec.video_id, f"fold{row['cv_fold']}", rec.links)
            for rec, row in zip(split_records(rows), rows, strict=True)
        ]
    )


def out_of_fold_scores(
    x: np.ndarray, y: np.ndarray, folds: np.ndarray, pipe: Pipeline
) -> np.ndarray:
    """Score each fold with a clone of `pipe` fitted on the other folds."""
    oof = np.full(len(y), np.nan)
    for fold in np.unique(folds):
        test = folds == fold
        if len(np.unique(y[~test])) < 2:
            raise ValueError(f"training folds without fold {fold} lack one class")
        fitted = clone(pipe).fit(x[~test], y[~test])
        oof[test] = fitted.predict_proba(x[test])[:, 1]
    return oof


def train_family_cv(
    data: TableData, family: str, config: TrainConfig, provenance: Mapping
) -> TrainedModel:
    """Grouped-CV model selection on the train split, then a refit on all of it."""
    rows, failed = data.rows["train"], data.failed["train"]
    y = data.labels("train")
    if len(np.unique(y)) < 2:
        raise ValueError(f"the train split needs both real and fake videos (has {len(y)})")
    fold_leakage = check_fold_leakage(rows)
    folds = np.array([r["cv_fold"] for r in rows])
    if len(np.unique(folds)) < 2:
        raise ValueError("cross-validation needs at least two cv_fold values")
    names = config.candidate_features
    used, dropped = select_features(feature_matrix(rows, names), names, config.max_missing_fraction)
    if not used:
        raise ValueError("no feature has enough training values")
    x = feature_matrix(rows, used)

    candidates = []
    best = None
    for params, pipe in candidate_grid(family, config.seed, config.forest_trees):
        oof = out_of_fold_scores(x, y, folds, pipe)
        auc = safe_auc(y, oof)
        per_fold = [safe_auc(y[folds == f], oof[folds == f]) for f in np.unique(folds)]
        candidates.append({"params": params, "oof_roc_auc": auc, "fold_roc_auc": per_fold})
        if best is None or (auc or 0) > (best[2] or 0) + 1e-12:
            best = (params, pipe, auc, oof)
    params, pipe, _, oof = best
    threshold = youden_threshold(y, oof)
    margin = choose_margin(oof, threshold, config.min_val_coverage, config.max_margin)
    pipe = clone(pipe).fit(x, y)

    sidecar = _sidecar(
        family,
        config,
        provenance,
        pipe,
        params,
        used,
        dropped,
        selection={
            "metric": "pooled out-of-fold ROC-AUC (grouped CV within the train split)",
            "folds": sorted(np.unique(folds).tolist()),
            "candidates": candidates,
        },
        threshold=threshold,
        margin=margin,
        fitted_on="pooled out-of-fold scores",
    )
    sidecar["split_sizes"] = {
        s: {"ok": len(data.rows[s]), "failed": len(data.failed[s])} for s in SPLITS
    }
    sidecar["leakage_check"] = {"splits": data.leakage, "cv_folds": fold_leakage}
    model = TrainedModel(pipe, sidecar)
    model.oof = model.predict_scores(rows, oof, failed)
    sidecar["metrics"] = {
        "train": evaluate_predictions(model.predict(rows, failed), config.n_boot, config.seed),
        "cv_out_of_fold": evaluate_predictions(model.oof, config.n_boot, config.seed),
    }
    sidecar["metrics_conditions"] = (
        "Train metrics are in-sample. Out-of-fold metrics come from models fitted without "
        "each fold, but are optimistic because hyperparameters, threshold and band were "
        "chosen on the same out-of-fold scores. Neither is a held-out estimate; see the "
        "test evaluation report."
    )
    return model


def train_models(
    data: TableData, config: TrainConfig, provenance: Mapping, cv: bool = False
) -> list[TrainedModel]:
    train = train_family_cv if cv else train_family
    return [train(data, family, config, provenance) for family in config.families]


def provenance_from_table(path: str | Path) -> dict:
    """Dataset, license and data-version tags for model sidecars (PRD LG-6).

    Read from the feature table's sidecar, which copies them from the manifest.
    """
    path = Path(path)
    meta = read_sidecar(path)
    dataset = meta.get("dataset") or {}
    name = dataset.get("dataset") or "unknown-dataset"
    short = name.lower().replace("faceforensics++", "ffpp").replace("nfi-rppg", "nfi")
    slug = "-".join(x for x in (short, dataset.get("compression")) if x)
    return {
        "dataset_slug": slug,
        "training_dataset": {
            "name": name,
            "compression": dataset.get("compression"),
            "methods": dataset.get("methods"),
            "include_dfd": dataset.get("include_dfd"),
            "split_source": dataset.get("split_source"),
            "sampling": dataset.get("sampling"),
            "license": dataset.get("license_notes")
            or "UNKNOWN: record the dataset license before using this model",
            "non_commercial_use_only": dataset.get("non_commercial_use_only", True),
        },
        "data_version": {
            "feature_table": path.name,
            "feature_table_sha256": file_sha256(path),
            "manifest_sha256": meta.get("manifest_sha256"),
            "feature_schema_version": meta.get("feature_schema_version", FEATURE_SCHEMA_VERSION),
            "pipeline": meta.get("pipeline"),
        },
    }

"""Confound audit: how much do the pulse features know about the media? (shortcut check)

Two questions, asked on one split of a feature table joined to its media
table (pulsecheck.dataset.media):

(a) How well does each media/shortcut column alone separate real from fake?
    Single-feature ROC-AUC, oriented so that higher values mean "fake"; an AUC
    far from 0.5 in either direction means the column alone is a shortcut.
(b) How much of each pulse classifier feature can be predicted from the media
    columns? The Spearman correlation with each media column, and the
    cross-validated R^2 of a random forest that predicts the feature from all
    of them (folds keep split groups together). A feature with
    |rho| >= RHO_FLAG or R^2 >= R2_FLAG is flagged: a classifier could pass a
    source/compression difference through it.

Rows with a missing value are left out pairwise, per statistic. The audit
describes association only; a flagged feature may still carry pulse signal.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from pulsecheck.dataset.media import SHORTCUT_FEATURES
from pulsecheck.models.evaluation import safe_auc

RHO_FLAG = 0.3
R2_FLAG = 0.2
# Media properties the pulse features are regressed on: the shortcut set minus
# face coverage, which is a tracking measurement rather than a file property.
MEDIA_FEATURES = tuple(f for f in SHORTCUT_FEATURES if f != "face_coverage")
MIN_ROWS = 10


def single_feature_auc(x: np.ndarray, y: np.ndarray) -> dict:
    """ROC-AUC of one column as a fake-vs-real score (higher = fake), missing rows left out."""
    keep = np.isfinite(x)
    auc = safe_auc(y[keep], x[keep])
    return {
        "n": int(keep.sum()),
        "roc_auc": auc,
        # Distance from chance regardless of direction: max(AUC, 1 - AUC).
        "separation": None if auc is None else max(auc, 1 - auc),
    }


def spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < MIN_ROWS or np.ptp(a[keep]) == 0 or np.ptp(b[keep]) == 0:
        return None
    return float(spearmanr(a[keep], b[keep]).statistic)


def cv_r2(
    media: np.ndarray,
    target: np.ndarray,
    groups: Sequence[str],
    folds: int = 5,
    seed: int = 0,
    trees: int = 100,
) -> float | None:
    """Out-of-fold R^2 of predicting `target` from the media columns (grouped folds).

    Pooled over folds: 1 - SSE / total sum of squares around the mean of the
    rows used. Can be negative (worse than predicting the mean).
    """
    keep = np.isfinite(target)
    x, y = media[keep], target[keep]
    g = np.asarray(groups)[keep]
    n_groups = len(np.unique(g))
    if len(y) < MIN_ROWS or n_groups < 2 or np.ptp(y) == 0:
        return None
    pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "reg",
                RandomForestRegressor(
                    n_estimators=trees, min_samples_leaf=5, random_state=seed, n_jobs=-1
                ),
            ),
        ]
    )
    pred = np.full(len(y), np.nan)
    for train, test in GroupKFold(n_splits=min(folds, n_groups)).split(x, y, g):
        pred[test] = pipe.fit(x[train], y[train]).predict(x[test])
    total = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((y - pred) ** 2)) / total if total > 0 else None


def audit(
    columns: Mapping[str, np.ndarray],
    y: np.ndarray,
    groups: Sequence[str],
    pulse_features: Sequence[str],
    shortcut_features: Sequence[str] = SHORTCUT_FEATURES,
    media_features: Sequence[str] = MEDIA_FEATURES,
    folds: int = 5,
    seed: int = 0,
    trees: int = 100,
) -> dict:
    """Run (a) and (b); `columns` maps every named feature to one value per row."""
    shortcut = {name: single_feature_auc(columns[name], y) for name in shortcut_features}
    media = np.column_stack([columns[name] for name in media_features])
    pulse = {}
    for name in pulse_features:
        rho = {m: spearman(columns[name], columns[m]) for m in media_features}
        r2 = cv_r2(media, columns[name], groups, folds, seed, trees)
        max_rho = max((abs(v) for v in rho.values() if v is not None), default=None)
        reasons = []
        if max_rho is not None and max_rho >= RHO_FLAG:
            strongest = max((m for m in rho if rho[m] is not None), key=lambda m: abs(rho[m]))
            reasons.append(f"|rho| {max_rho:.2f} with {strongest}")
        if r2 is not None and r2 >= R2_FLAG:
            reasons.append(f"media R^2 {r2:.2f}")
        pulse[name] = {
            "n": int(np.isfinite(columns[name]).sum()),
            "spearman": rho,
            "max_abs_spearman": max_rho,
            "cv_r2": r2,
            "flagged": bool(reasons),
            "flag_reasons": reasons,
        }
    return {
        "thresholds": {"abs_spearman": RHO_FLAG, "cv_r2": R2_FLAG},
        "media_features": list(media_features),
        "regressor": {
            "model": "RandomForestRegressor",
            "trees": trees,
            "min_samples_leaf": 5,
            "imputation": "median",
            "folds": folds,
            "fold_unit": "split group (GroupKFold)",
            "seed": seed,
        },
        "rows": int(len(y)),
        "real": int(np.sum(y == 0)),
        "fake": int(np.sum(y == 1)),
        "shortcut_auc": shortcut,
        "pulse_features": pulse,
        "flagged": [name for name, entry in pulse.items() if entry["flagged"]],
    }

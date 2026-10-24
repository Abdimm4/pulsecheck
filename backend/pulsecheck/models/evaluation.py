"""Evaluation metrics (PRD Section 13).

The positive class is "fake" (manipulated). Each evaluated video has a state
(more consistent with real / manipulated / inconclusive) or failed
extraction, and optionally a score where higher means more consistent with
manipulated. Metrics are reported on the videos the method decided:

- accuracy, precision, recall (sensitivity), specificity, F1 and the confusion
  matrix use decided videos only (inconclusive and failed are excluded);
- ROC-AUC uses every scored video and ignores thresholds;
- coverage = decided / all videos (failures included), inconclusive rate =
  inconclusive / successfully extracted videos, failure rate = failed / all;
- 95% confidence intervals come from a percentile bootstrap that resamples
  whole split groups (an original plus everything derived from it), because
  videos in one group are not independent.

A metric that cannot be computed (e.g. ROC-AUC with one class) is None.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

from pulsecheck.models.heuristic import ResultState

REAL_STATE = ResultState.REAL.value
FAKE_STATE = ResultState.MANIPULATED.value
INCONCLUSIVE_STATE = ResultState.INCONCLUSIVE.value
FAILED = "failed"
CI_METRICS = ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "coverage")


@dataclass
class Prediction:
    video_id: str
    label: int  # 1 = fake
    manipulation_type: str
    group_id: str
    state: str  # a ResultState value, or "failed"
    score: float | None = None  # higher = more consistent with manipulated
    forced: int | None = None  # label at the bare decision threshold, ignoring band and gate
    reason: str = ""


def _ratio(num: float, den: float) -> float | None:
    return float(num / den) if den else None


def confusion(y: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    return {
        "tn": int(np.sum((y == 0) & (pred == 0))),
        "fp": int(np.sum((y == 0) & (pred == 1))),
        "fn": int(np.sum((y == 1) & (pred == 0))),
        "tp": int(np.sum((y == 1) & (pred == 1))),
    }


def binary_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    c = confusion(y, pred)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    return {
        "n": int(len(y)),
        "accuracy": _ratio(tp + tn, len(y)),
        "precision": precision,
        "recall": recall,
        "specificity": _ratio(tn, tn + fp),
        "f1": f1,
        "confusion_matrix": c,
    }


def safe_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    if len(y) == 0 or len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, score))


def _arrays(preds: Sequence[Prediction]):
    y = np.array([p.label for p in preds], dtype=int)
    state = np.array([p.state for p in preds], dtype=object)
    score = np.array([np.nan if p.score is None else p.score for p in preds], dtype=float)
    return y, state, score


def _point(y, state, score) -> dict:
    decided = (state == REAL_STATE) | (state == FAKE_STATE)
    pred = (state[decided] == FAKE_STATE).astype(int)
    metrics = binary_metrics(y[decided], pred)
    scored = np.isfinite(score)
    metrics["roc_auc"] = safe_auc(y[scored], score[scored])
    metrics["coverage"] = _ratio(decided.sum(), len(y))
    return metrics


def bootstrap_ci(
    preds: Sequence[Prediction], n_boot: int = 1000, seed: int = 0, level: float = 0.95
) -> dict[str, list[float] | None]:
    """Group-level percentile bootstrap CIs for CI_METRICS."""
    y, state, score = _arrays(preds)
    members: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(preds):
        members[p.group_id or p.video_id].append(i)
    groups = [np.array(v) for v in members.values()]
    if len(groups) < 2 or n_boot < 1:
        return dict.fromkeys(CI_METRICS)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {m: [] for m in CI_METRICS}
    for _ in range(n_boot):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        point = _point(y[idx], state[idx], score[idx])
        for m in CI_METRICS:
            if point[m] is not None:
                samples[m].append(point[m])
    alpha = (1 - level) / 2
    out = {}
    for m, values in samples.items():
        # Require most resamples to define the metric, else the interval is misleading.
        if len(values) < 0.9 * n_boot:
            out[m] = None
        else:
            out[m] = [float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha))]
    return out


def _subset_summary(preds: Sequence[Prediction]) -> dict:
    y, state, score = _arrays(preds)
    n = len(preds)
    failed = int(np.sum(state == FAILED))
    inconclusive = int(np.sum(state == INCONCLUSIVE_STATE))
    point = _point(y, state, score)
    return {
        "n": n,
        "failed": failed,
        "inconclusive": inconclusive,
        "decided": int(point["n"]),
        "coverage": point["coverage"],
        "inconclusive_rate": _ratio(inconclusive, n - failed),
        "failure_rate": _ratio(failed, n),
        **{k: v for k, v in point.items() if k not in ("n", "coverage")},
    }


def evaluate_predictions(preds: Sequence[Prediction], n_boot: int = 1000, seed: int = 0) -> dict:
    """Aggregate metrics, CIs and the per-manipulation breakdown."""
    preds = list(preds)
    summary = _subset_summary(preds)
    summary["confidence_intervals_95"] = bootstrap_ci(preds, n_boot, seed)
    summary["bootstrap"] = {"resamples": n_boot, "unit": "split group", "seed": seed}
    summary["failure_reasons"] = dict(Counter(p.reason for p in preds if p.state == FAILED))
    summary["inconclusive_reasons"] = dict(
        Counter(p.reason for p in preds if p.state == INCONCLUSIVE_STATE)
    )

    forced = [p for p in preds if p.forced is not None]
    if forced:
        y = np.array([p.label for p in forced])
        summary["at_threshold_without_inconclusive"] = binary_metrics(
            y, np.array([p.forced for p in forced])
        )

    reals = [p for p in preds if p.label == 0]
    per_method = {}
    for method in sorted({p.manipulation_type for p in preds}):
        subset = [p for p in preds if p.manipulation_type == method]
        entry = _subset_summary(subset)
        if method != "none":
            # Fakes of one method against all real test videos (ROC-AUC only);
            # the thresholded rates are within the method's own videos.
            pair = subset + reals
            y, _, score = _arrays(pair)
            scored = np.isfinite(score)
            entry["roc_auc_vs_all_real"] = safe_auc(y[scored], score[scored])
        entry.pop("roc_auc", None)  # undefined within a single class
        per_method[method] = entry
    summary["per_manipulation"] = per_method
    return summary


def runtime_summary(runtimes: Sequence[float]) -> dict:
    r = np.array([x for x in runtimes if x is not None and np.isfinite(x)], dtype=float)
    if r.size == 0:
        return {"n": 0}
    return {
        "n": int(r.size),
        "mean_s": float(r.mean()),
        "median_s": float(np.median(r)),
        "p95_s": float(np.quantile(r, 0.95)),
        "max_s": float(r.max()),
        "total_s": float(r.sum()),
    }

"""Benchmark metrics with group-bootstrap 95% intervals (PRD FR-12).

Each row is one heart-rate estimate (one clip, condition and ROI). Metrics:

- detection_rate: fraction of rows with |HR_est - HR_true| <= 5 bpm; a row
  without an estimate counts as not detected. On the a = 0 null it is the
  false-detection rate: how often the estimate lands within 5 bpm of the
  clip's target HR although nothing was injected;
- estimate_rate: fraction of rows with any estimate;
- mae_bpm, rmse_bpm: over rows with an estimate;
- snr_true_db: mean SNR at the true frequency (de Haan & Jeanne 2013).

Intervals resample source groups with replacement (all clips of a group
together), with one set of bootstrap weights per split so that differences
between variants are paired.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

METRICS = ("detection_rate", "estimate_rate", "mae_bpm", "rmse_bpm", "snr_true_db")


class GroupBootstrap:
    def __init__(self, groups: Iterable[str], n_boot: int = 1000, seed: int = 0) -> None:
        self.groups = sorted(set(groups))
        self.index = {g: i for i, g in enumerate(self.groups)}
        rng = np.random.default_rng(seed)
        n = len(self.groups)
        self.weights = (
            rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(np.float64)
            if n
            else np.zeros((n_boot, 0))
        )

    def _sums(self, rows: Sequence[Mapping]) -> dict[str, np.ndarray]:
        g = np.array([self.index[r["group"]] for r in rows], dtype=np.int64)
        n = len(self.groups)
        detected = np.array([bool(r["detected"]) for r in rows], dtype=np.float64)
        has = np.array([r["error"] is not None for r in rows])
        err = np.array([abs(r["error"]) if r["error"] is not None else 0.0 for r in rows])
        snr_ok = np.array([r.get("snr_true_db") is not None for r in rows])
        snr = np.array(
            [r["snr_true_db"] if r.get("snr_true_db") is not None else 0.0 for r in rows]
        )

        def add(values):
            return np.bincount(g, weights=values, minlength=n) if len(g) else np.zeros(n)

        return {
            "n": add(np.ones(len(rows))),
            "det": add(detected),
            "n_est": add(has.astype(float)),
            "abs": add(err),
            "sq": add(err**2),
            "n_snr": add(snr_ok.astype(float)),
            "snr": add(snr),
        }

    @staticmethod
    def _stats(s: Mapping[str, np.ndarray], w: np.ndarray | None) -> dict[str, np.ndarray]:
        def tot(key):
            return s[key].sum() if w is None else w @ s[key]

        with np.errstate(invalid="ignore", divide="ignore"):
            n, n_est, n_snr = tot("n"), tot("n_est"), tot("n_snr")
            return {
                "detection_rate": tot("det") / n,
                "estimate_rate": n_est / n,
                "mae_bpm": tot("abs") / n_est,
                "rmse_bpm": np.sqrt(tot("sq") / n_est),
                "snr_true_db": tot("snr") / n_snr,
            }

    def summarize(self, rows: Sequence[Mapping]) -> dict:
        """Point estimates and 95% intervals of every metric for one cell."""
        out: dict = {"n": len(rows), "n_groups": len({r["group"] for r in rows})}
        if not rows:
            return out | {m: None for m in METRICS}
        sums = self._sums(rows)
        point = self._stats(sums, None)
        boot = self._stats(sums, self.weights)
        for metric in METRICS:
            value = float(point[metric])
            out[metric] = value if np.isfinite(value) else None
            draws = boot[metric][np.isfinite(boot[metric])]
            out[f"{metric}_ci"] = (
                [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
                if draws.size and np.isfinite(value)
                else None
            )
        return out

    def paired_delta(
        self, rows_a: Sequence[Mapping], rows_b: Sequence[Mapping], metric: str
    ) -> dict:
        """metric(a) - metric(b) with a paired 95% interval (same group resamples)."""
        if not rows_a or not rows_b:
            return {"delta": None, "ci": None}
        sa, sb = self._sums(rows_a), self._sums(rows_b)
        point = float(self._stats(sa, None)[metric] - self._stats(sb, None)[metric])
        draws = self._stats(sa, self.weights)[metric] - self._stats(sb, self.weights)[metric]
        draws = draws[np.isfinite(draws)]
        ci = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
        return {"delta": point if np.isfinite(point) else None, "ci": ci if draws.size else None}


def roc_auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """ROC-AUC (Mann-Whitney, ties counted half), None if a class is missing."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    pos, neg = s[y], s[~y]
    if not pos.size or not neg.size:
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size)
    values = np.concatenate([pos, neg])[order]
    # Average ranks over ties.
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and values[j + 1] == values[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))

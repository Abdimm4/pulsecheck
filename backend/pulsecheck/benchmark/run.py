"""Run the injection benchmark on the trace cache and write its report (PRD FR-12).

Steps (all selection on the dev half; the held-out half is only read at the end):

1. Baseline: the current pipeline (standard ROIs, per-ROI estimates, CHROM,
   and green for comparison) on every condition; the diagnosis breaks its dev
   detection rate down by amplitude, compression, ROI, skin pixels, motion,
   dataset and the clip's native in-band noise.
2. Selection: greedy, one group of options at a time (method x ROI/HR source,
   then statistic x landmark smoothing, detrending x filter order, Welch
   window x peak picking, then the method again), each scored by the dev
   detection rate pooled over realistic amplitudes (0.2-1 %) at CRF 23 and 28.
   A challenger replaces the incumbent only if it is better by more than
   MIN_GAIN, so defaults survive ties. The full trail is reported.
3. Quality ramps: on dev (amplitudes > 0, all degradations) for the selected
   variant, the spectral-concentration and SNR ramps are set from the
   distributions of correct and incorrect estimates, and the AUC of the
   quality score for correct vs incorrect HR recovery is reported on dev and,
   once, on held-out.
4. Held-out: current vs selected by amplitude x degradation, with group
   bootstrap intervals and paired differences.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import multiprocessing
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
from scipy import signal as sps

from pulsecheck.benchmark.cache import DEGRADATIONS, condition_key, load_clip
from pulsecheck.benchmark.injection import AMPLITUDES, REALISTIC_AMPLITUDES
from pulsecheck.benchmark.metrics import GroupBootstrap, roc_auc
from pulsecheck.benchmark.sources import DEV, HELDOUT, BenchmarkClip
from pulsecheck.benchmark.variants import TOLERANCE_BPM, Variant, estimate, score_estimate
from pulsecheck.face.regions import FACE_ROI

log = logging.getLogger("pulsecheck.benchmark")

SELECTION_AMPLITUDES = REALISTIC_AMPLITUDES
SELECTION_DEGRADATIONS = ("crf23", "crf28")
MIN_GAIN = 0.005  # a challenger must beat the incumbent's dev score by more than this
DEGRADATION_NAMES = tuple(d.name for d in DEGRADATIONS)

CURRENT = Variant("current-chrom", method="chrom")
CURRENT_GREEN = Variant("current-green", method="green")


# --------------------------------------------------------------------------- evaluation


def _native_noise(arrays: Mapping[str, np.ndarray]) -> float | None:
    """In-band (0.7-3 Hz) std of the a = 0 / CRF 18 whole-face green trace, % of its mean."""
    key = condition_key(0.0, "crf18")
    t = arrays.get(f"{key}|t")
    rgb = arrays.get(f"{key}|s0/mean/{FACE_ROI}|rgb")
    if t is None or rgb is None or len(t) < 30:
        return None
    green = rgb[:, 1]
    ok = np.isfinite(green)
    if ok.mean() < 0.7:
        return None
    green = np.interp(t, t[ok], green[ok])
    fs = (len(t) - 1) / (t[-1] - t[0])
    sos = sps.butter(4, [0.7, 3.0], btype="bandpass", fs=fs, output="sos")
    return float(100 * np.std(sps.sosfiltfilt(sos, green / green.mean() - 1.0)))


def transfer_stats(meta: Mapping, arrays: Mapping[str, np.ndarray]) -> list[dict]:
    """How much of the injected pulse survives encoding, per amplitude x degradation.

    The difference between an injected condition and the a = 0 null of the
    same degradation (same frames, same codec settings) is regressed on the
    injected waveform: `gain` is the recovered fraction of the injected
    relative change (1 = intact) and `r` their correlation. Measured on the
    green mean of the large forehead/cheek ROIs, where the injection weight is
    1, so the expected gain without a codec is close to 1.
    """
    from pulsecheck.benchmark.injection import PulseSpec
    from pulsecheck.face.regions import ROI_NAMES

    if not meta.get("pulse"):
        return []
    pulse = PulseSpec(**meta["pulse"])
    fps = meta["source"]["fps"]
    out = []
    for degradation in DEGRADATION_NAMES:
        null = condition_key(0.0, degradation)
        g0 = [arrays.get(f"{null}|s0/mean/large:{roi}|rgb") for roi in ROI_NAMES]
        if any(g is None for g in g0):
            continue
        for amplitude in AMPLITUDES[1:]:
            cond = condition_key(amplitude, degradation)
            g1 = [arrays.get(f"{cond}|s0/mean/large:{roi}|rgb") for roi in ROI_NAMES]
            if any(g is None for g in g1):
                continue
            m = min(min(len(g) for g in g0), min(len(g) for g in g1))
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN ROI traces
                rel = np.nanmean(
                    [
                        (b[:m, 1] - a[:m, 1]) / np.nanmean(a[:m, 1])
                        for a, b in zip(g0, g1, strict=True)
                    ],
                    axis=0,
                )
            ref = pulse.intensity(np.arange(m) / fps)
            ok = np.isfinite(rel)
            if ok.sum() < 30:
                continue
            slope = float(np.polyfit(ref[ok], rel[ok], 1)[0])
            out.append(
                {
                    "amplitude": amplitude,
                    "degradation": degradation,
                    "gain": slope / amplitude,
                    "r": float(np.corrcoef(ref[ok], rel[ok])[0, 1]),
                }
            )
    return out


def _transfer_clip(task) -> list[dict]:
    clip, cache_dir = task
    loaded = load_clip(cache_dir, clip["clip_id"])
    if loaded is None:
        return []
    rows = transfer_stats(*loaded)
    for row in rows:
        row["dataset"] = clip["dataset"]
    return rows


def injection_transfer(clips, cache_dir, workers: int = 1) -> list[dict]:
    """Median gain and correlation of the surviving pulse per dataset x amplitude x degradation."""
    tasks = [(c.to_dict(), str(cache_dir)) for c in clips]
    if workers <= 1:
        results = [_transfer_clip(t) for t in tasks]
    else:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            results = pool.map(_transfer_clip, tasks, chunksize=1)
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for rows in results:
        for r in rows:
            for ds in ("all", r["dataset"]):
                cells[(ds, r["amplitude"], r["degradation"])].append(r)
    return [
        {
            "dataset": ds,
            "amplitude": amp,
            "degradation": deg,
            "n": len(rows),
            "gain_median": float(np.median([r["gain"] for r in rows])),
            "r_median": float(np.median([r["r"] for r in rows])),
        }
        for (ds, amp, deg), rows in sorted(cells.items())
    ]


def _evaluate_clip(task) -> list[dict]:
    clip, cache_dir, variants, conditions = task
    loaded = load_clip(cache_dir, clip["clip_id"])
    if loaded is None:
        return []
    meta, arrays = loaded
    true_hr = meta["true_hr_bpm"]
    noise = _native_noise(arrays)
    rows = []
    for amplitude, degradation in conditions:
        cond = condition_key(amplitude, degradation)
        if f"{cond}|t" not in arrays:
            continue
        diag = meta["conditions"][cond]
        t = arrays[f"{cond}|t"]
        for variant in variants:
            keys = dict(variant.roi_keys())
            traces = {key: arrays[f"{cond}|{key}|rgb"] for key in keys.values()}
            for est in estimate(variant, t, traces):
                pixels = diag["median_skin_pixels"]
                prefix = "large:" if variant.layout == "large" else ""
                if est.roi in keys:
                    skin = pixels.get(est.roi if est.roi == FACE_ROI else prefix + est.roi)
                else:  # fused: mean over the regional ROIs
                    skin = float(np.mean([pixels[prefix + r] for r in keys]))
                rows.append(
                    {
                        "variant": variant.name,
                        "clip_id": clip["clip_id"],
                        "dataset": clip["dataset"],
                        "group": clip["group"],
                        "split": clip["split"],
                        "amplitude": amplitude,
                        "degradation": degradation,
                        "roi": est.roi,
                        "hr_true": true_hr,
                        "peak_snr_db": est.peak_snr_db,
                        "concentration": est.concentration,
                        "prominence": est.prominence,
                        "reason": est.reason,
                        "skin_pixels": skin,
                        "motion": meta["motion"]["median_iod_per_frame"],
                        "bitrate_kbps": diag["bitrate_kbps"],
                        "native_noise_pct": noise,
                        "target_separated": meta.get("target_separated"),
                        **score_estimate(est, true_hr),
                    }
                )
    return rows


def evaluate(
    variants: Sequence[Variant],
    clips: Sequence[BenchmarkClip],
    cache_dir: Path,
    conditions: Sequence[tuple[float, str]],
    workers: int = 1,
) -> list[dict]:
    tasks = [(c.to_dict(), str(cache_dir), list(variants), list(conditions)) for c in clips]
    if workers <= 1:
        results = [_evaluate_clip(t) for t in tasks]
    else:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            results = pool.map(_evaluate_clip, tasks, chunksize=1)
    return [row for rows in results for row in rows]


ALL_CONDITIONS = [(a, d) for a in AMPLITUDES for d in DEGRADATION_NAMES]
SELECTION_CONDITIONS = [(a, d) for a in SELECTION_AMPLITUDES for d in SELECTION_DEGRADATIONS] + [
    (0.0, d) for d in SELECTION_DEGRADATIONS
]


def _select(rows, **match) -> list[dict]:
    return [r for r in rows if all(r[k] == v for k, v in match.items())]


def selection_score(rows: Sequence[dict], variant: str) -> tuple[float, float]:
    """(dev detection rate, dev MAE) pooled over the selection conditions."""
    pool = [
        r
        for r in rows
        if r["variant"] == variant
        and r["amplitude"] in SELECTION_AMPLITUDES
        and r["degradation"] in SELECTION_DEGRADATIONS
    ]
    if not pool:
        return 0.0, math.inf
    detected = float(np.mean([r["detected"] for r in pool]))
    errors = [abs(r["error"]) for r in pool if r["error"] is not None]
    return detected, float(np.mean(errors)) if errors else math.inf


def null_rate(rows: Sequence[dict], variant: str) -> float | None:
    pool = [
        r
        for r in rows
        if r["variant"] == variant
        and r["amplitude"] == 0
        and r["degradation"] in SELECTION_DEGRADATIONS
    ]
    return float(np.mean([r["detected"] for r in pool])) if pool else None


# --------------------------------------------------------------------------- selection


def _stages(best: Variant) -> list[tuple[str, Callable[[Variant], list[Variant]]]]:
    def method_roi(v: Variant) -> list[Variant]:
        out = []
        for method in ("green", "chrom", "pos"):
            for layout, source in (
                ("standard", "per_roi"),
                ("standard", "fused"),
                ("large", "per_roi"),
                ("large", "fused"),
                ("standard", "face"),
            ):
                out.append(
                    Variant(
                        f"{method}-{source}-{layout}" if source != "face" else f"{method}-face",
                        method=method,
                        layout=layout,
                        hr_source=source,
                        statistic=v.statistic,
                        smoothing=v.smoothing,
                        signal=v.signal,
                    )
                )
        return out

    def stat_smooth(v: Variant) -> list[Variant]:
        from pulsecheck.benchmark.cache import SMOOTHING

        return [
            Variant(
                f"{v.name}|{stat}|s{smooth}",
                v.method,
                v.layout,
                v.hr_source,
                stat,
                smooth,
                v.signal,
            )
            for stat in ("mean", "robust")
            for smooth in SMOOTHING
        ]

    def detrend_filter(v: Variant) -> list[Variant]:
        return [
            v.with_signal(f"{v.name}|{detrend}|order{order}", detrend=detrend, filter_order=order)
            for detrend in ("linear", "smoothness_priors")
            for order in (2, 4, 6)
        ]

    def spectrum_peak(v: Variant) -> list[Variant]:
        return [
            v.with_signal(
                f"{v.name}|welch{window:g}|{picking}", welch_window_s=window, peak_picking=picking
            )
            for window in (10.0, 6.0, 4.0)
            for picking in ("argmax", "harmonic")
        ]

    def method_again(v: Variant) -> list[Variant]:
        return [
            Variant(
                f"{v.name}|{method}",
                method,
                v.layout,
                v.hr_source,
                v.statistic,
                v.smoothing,
                v.signal,
            )
            for method in ("green", "chrom", "pos")
        ]

    return [
        ("pulse method and ROI / HR source", method_roi),
        ("ROI statistic and landmark smoothing", stat_smooth),
        ("detrending and filter order", detrend_filter),
        ("Welch window and peak picking", spectrum_peak),
        ("pulse method, rechecked with the final settings", method_again),
    ]


def _same_settings(a: Variant, b: Variant) -> bool:
    keys = ("method", "layout", "hr_source", "statistic", "smoothing")
    return all(getattr(a, k) == getattr(b, k) for k in keys) and dict(a.signal) == dict(b.signal)


def select_variant(dev_clips, cache_dir, workers) -> tuple[Variant, list[dict]]:
    """Greedy selection on dev; returns the chosen variant and the trail."""
    incumbent = CURRENT
    trail = []
    for title, make in _stages(incumbent):
        candidates = make(incumbent)
        if not any(_same_settings(c, incumbent) for c in candidates):
            candidates.append(incumbent)
        rows = evaluate(candidates, dev_clips, cache_dir, SELECTION_CONDITIONS, workers)
        scored = []
        for c in candidates:
            det, mae = selection_score(rows, c.name)
            scored.append(
                {
                    "variant": c.name,
                    "settings": c.describe(),
                    "detection_rate": det,
                    "mae_bpm": mae if math.isfinite(mae) else None,
                    "null_false_detection": null_rate(rows, c.name),
                }
            )
        scored.sort(key=lambda s: (-s["detection_rate"], s["mae_bpm"] or math.inf))
        inc = next(s for s in scored if _same_settings(_variant(s), incumbent))
        top = scored[0]
        chosen = inc
        if top["detection_rate"] > inc["detection_rate"] + MIN_GAIN:
            chosen = top
        incumbent = _variant(chosen)
        log.info(
            "stage %r: chose %s (dev score %.3f)",
            title,
            chosen["variant"],
            chosen["detection_rate"],
        )
        trail.append({"stage": title, "candidates": scored, "chosen": chosen["variant"]})
    return incumbent, trail


def _variant(scored: Mapping) -> Variant:
    s = dict(scored["settings"])
    s["signal"] = tuple(sorted(s["signal"].items()))
    return Variant(**s)


# --------------------------------------------------------------------------- quality


def _ramp(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def quality_scores(rows, concentration_range, snr_range) -> np.ndarray:
    c = np.array([r["concentration"] if r["concentration"] is not None else 0.0 for r in rows])
    s = np.array([r["peak_snr_db"] if r["peak_snr_db"] is not None else -60.0 for r in rows])
    return np.sqrt(_ramp(c, *concentration_range) * _ramp(s, *snr_range))


def calibrate_quality(dev_rows: Sequence[dict], held_rows: Sequence[dict], old: Mapping) -> dict:
    """Quality ramps from dev estimates (amplitude > 0) and their AUCs on dev and held-out.

    Each ramp runs from the median value among incorrect estimates (quality 0
    for a typical failure) to the median among correct estimates (quality 1
    for a typical success); the score is the geometric mean of both ramps.
    """

    def usable(rows):
        return [r for r in rows if r["amplitude"] > 0 and r["hr_est"] is not None]

    dev, held = usable(dev_rows), usable(held_rows)
    ok = np.array([r["detected"] for r in dev])
    out: dict = {
        "n_dev": len(dev),
        "n_heldout": len(held),
        "dev_correct_fraction": float(ok.mean()),
    }
    ranges = {}
    for name, key in (("concentration_range", "concentration"), ("snr_range_db", "peak_snr_db")):
        values = np.array([r[key] for r in dev], dtype=float)
        lo, hi = float(np.median(values[~ok])), float(np.median(values[ok]))
        if not hi > lo:
            lo, hi = old[name]
        ranges[name] = [round(lo, 3), round(hi, 3)]
    out["ramps"] = ranges
    out["old_ramps"] = {k: list(v) for k, v in old.items()}
    for split, rows in (("dev", dev), ("heldout", held)):
        labels = [r["detected"] for r in rows]
        out[f"auc_{split}"] = {
            "quality_new": roc_auc(
                quality_scores(rows, ranges["concentration_range"], ranges["snr_range_db"]), labels
            ),
            "quality_old": roc_auc(
                quality_scores(rows, old["concentration_range"], old["snr_range_db"]), labels
            ),
            "peak_snr_db": roc_auc([r["peak_snr_db"] for r in rows], labels),
            "concentration": roc_auc([r["concentration"] for r in rows], labels),
            "prominence": roc_auc([r["prominence"] for r in rows], labels),
        }
        q = quality_scores(rows, ranges["concentration_range"], ranges["snr_range_db"])
        labels = np.asarray(labels)
        out[f"gate_{split}"] = [
            {
                "min_quality": thr,
                "coverage": float(np.mean(q >= thr)),
                "correct_among_passed": float(labels[q >= thr].mean())
                if (q >= thr).any()
                else None,
            }
            for thr in (0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8)
        ]
    return out


# --------------------------------------------------------------------------- aggregation


def cell_table(rows, boot: Mapping[str, GroupBootstrap], variants, datasets=("all",)) -> list[dict]:
    """Metrics per variant x split x dataset x amplitude x degradation."""
    index: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        for ds in datasets:
            if ds == "all" or r["dataset"] == ds:
                index[(r["variant"], r["split"], ds, r["amplitude"], r["degradation"])].append(r)
    out = []
    for (variant, split, ds, amp, deg), cell in sorted(index.items()):
        if variant not in variants:
            continue
        summary = boot[split].summarize(cell)
        out.append(
            {
                "variant": variant,
                "split": split,
                "dataset": ds,
                "amplitude": amp,
                "degradation": deg,
                **summary,
            }
        )
    return out


def _tercile_breakdown(rows, key: str) -> list[dict]:
    values = np.array([r[key] for r in rows if r[key] is not None], dtype=float)
    if values.size < 6:
        return []
    edges = np.percentile(values, [100 / 3, 200 / 3])
    out = []
    for i, label in enumerate(("low", "middle", "high")):
        lo = -np.inf if i == 0 else edges[i - 1]
        hi = np.inf if i == 2 else edges[i]
        cell = [r for r in rows if r[key] is not None and lo <= r[key] < hi]
        if cell:
            out.append(
                {
                    "tercile": label,
                    "range": [float(min(r[key] for r in cell)), float(max(r[key] for r in cell))],
                    "n": len(cell),
                    "detection_rate": float(np.mean([r["detected"] for r in cell])),
                }
            )
    return out


def diagnose(rows: Sequence[dict], variant: str) -> dict:
    """Where the current pipeline fails, on dev rows at realistic amplitudes."""
    real = [
        r
        for r in rows
        if r["variant"] == variant and r["split"] == DEV and r["amplitude"] in REALISTIC_AMPLITUDES
    ]
    focus = [r for r in real if r["degradation"] in SELECTION_DEGRADATIONS]
    by = defaultdict(list)
    for r in focus:
        by[("roi", r["roi"])].append(r)
        by[("dataset", r["dataset"])].append(r)
    for r in real:
        by[("degradation", r["degradation"])].append(r)
    out = {
        "scope": "dev rows at amplitudes 0.2-1 %; ROI/dataset/terciles at CRF 23-28",
        "by_roi": {k[1]: _rate(v) for k, v in by.items() if k[0] == "roi"},
        "by_dataset": {k[1]: _rate(v) for k, v in by.items() if k[0] == "dataset"},
        "by_degradation": {k[1]: _rate(v) for k, v in by.items() if k[0] == "degradation"},
        "by_skin_pixels": _tercile_breakdown(focus, "skin_pixels"),
        "by_motion": _tercile_breakdown(focus, "motion"),
        "by_native_noise": _tercile_breakdown(focus, "native_noise_pct"),
        "by_bitrate": _tercile_breakdown(focus, "bitrate_kbps"),
        "rejected": float(np.mean([r["hr_est"] is None for r in focus])) if focus else None,
    }
    # How far off the misses are: near the native HR, a harmonic, or elsewhere.
    misses = [r for r in focus if r["hr_est"] is not None and not r["detected"]]
    out["miss_error_median_bpm"] = (
        float(np.median([abs(r["error"]) for r in misses])) if misses else None
    )
    return out


def _rate(rows) -> dict:
    return {"n": len(rows), "detection_rate": float(np.mean([r["detected"] for r in rows]))}


# --------------------------------------------------------------------------- report


def _fmt(v, digits=3):
    return "n/a" if v is None else f"{v:.{digits}f}"


def _ci(cell, metric, digits=2, pct=False):
    v, ci = cell.get(metric), cell.get(f"{metric}_ci")
    if v is None:
        return "n/a"
    scale = 100 if pct else 1
    d = 0 if pct else digits
    text = f"{v * scale:.{d}f}"
    if ci:
        text += f" [{ci[0] * scale:.{d}f}, {ci[1] * scale:.{d}f}]"
    return text


def _grid(cells, variant, split, metric, pct=False, dataset="all") -> list[str]:
    lookup = {
        (c["amplitude"], c["degradation"]): c
        for c in cells
        if c["variant"] == variant and c["split"] == split and c["dataset"] == dataset
    }
    head = "| amplitude | " + " | ".join(DEGRADATION_NAMES) + " |"
    lines = [head, "|---" * (len(DEGRADATION_NAMES) + 1) + "|"]
    for amp in AMPLITUDES:
        label = "0 (null)" if amp == 0 else f"{100 * amp:g} %"
        row = [
            _ci(lookup[(amp, d)], metric, pct=pct) if (amp, d) in lookup else "-"
            for d in DEGRADATION_NAMES
        ]
        lines.append(f"| {label} | " + " | ".join(row) + " |")
    return lines


def _transfer_grid(cells, dataset="all") -> list[str]:
    lookup = {(c["amplitude"], c["degradation"]): c for c in cells if c["dataset"] == dataset}
    head = "| amplitude | " + " | ".join(DEGRADATION_NAMES) + " |"
    lines = [head, "|---" * (len(DEGRADATION_NAMES) + 1) + "|"]
    for amp in AMPLITUDES[1:]:
        row = [
            f"{lookup[(amp, d)]['gain_median']:.2f} (r {lookup[(amp, d)]['r_median']:.2f})"
            if (amp, d) in lookup
            else "-"
            for d in DEGRADATION_NAMES
        ]
        lines.append(f"| {100 * amp:g} % | " + " | ".join(row) + " |")
    return lines


def render_markdown(report: Mapping) -> str:
    sel = report["selected"]
    cells = report["cells"]
    lines = [
        "# Pulse extraction benchmark (semi-synthetic injection, PRD FR-12)",
        "",
        f"Created {report['created_at']} with PulseCheck "
        f"{report['meta'].get('pulsecheck_version')}.",
        "Numbers describe this benchmark only: a synthetic pulse injected into real, "
        "already-compressed videos. They are not heart-rate accuracy on real pulses.",
        "",
        "## Data and conditions",
        "",
    ]
    data = report["data"]
    lines += [
        "| dataset | split | clips | groups | median source px (h) | median IOD px | "
        "median motion (IOD/frame) |",
        "|---|---|---|---|---|---|---|",
    ]
    for d in data["summary"]:
        lines.append(
            f"| {d['dataset']} | {d['split']} | {d['clips']} | {d['groups']} | "
            f"{_fmt(d['source_height'], 0)} | {_fmt(d['iod_px'], 0)} | {_fmt(d['motion'], 4)} |"
        )
    lines += ["", f"Clips that failed to build: {data['failed'] or 'none'}.", ""]
    lines += ["Median measured bitrate (kbit/s) and whole-face skin pixels per degradation:", ""]
    lines += [
        "| dataset | " + " | ".join(DEGRADATION_NAMES) + " |",
        "|---" * (len(DEGRADATION_NAMES) + 1) + "|",
    ]
    for ds, per in data["bitrate_kbps"].items():
        lines.append(
            f"| {ds} bitrate | " + " | ".join(_fmt(per.get(d), 0) for d in DEGRADATION_NAMES) + " |"
        )
    for ds, per in data["face_pixels"].items():
        lines.append(
            f"| {ds} face px | " + " | ".join(_fmt(per.get(d), 0) for d in DEGRADATION_NAMES) + " |"
        )
    lines += [
        "",
        f"Injected pulse: {report['injection']}",
        "",
        "Injected pulse surviving the encode (median over clips; gain = recovered fraction "
        "of the injected change on the large forehead/cheek ROIs, 1 = intact; r = "
        "correlation with the injected waveform):",
        "",
        *_transfer_grid(report.get("injection_transfer") or []),
        "",
        "Detection = |HR_est - HR_true| <= 5 bpm (a missing estimate counts as a miss). "
        "On the a = 0 null the same rate is the false-detection rate against the clip's "
        "(never injected) target HR. Brackets: 95% group-bootstrap intervals "
        f"({report['n_boot']} resamples of source groups).",
        "",
        "## 1. Current pipeline (baseline) and diagnosis",
        "",
        "Current = standard forehead/cheek ROIs, one estimate per ROI (as the features "
        "report them), CHROM, linear detrend, Butterworth order 4, Welch 10 s, argmax peak.",
        "",
        "Detection rate (%), current CHROM, dev:",
        "",
        *_grid(cells, CURRENT.name, DEV, "detection_rate", pct=True),
        "",
        "Detection rate (%), current green, dev:",
        "",
        *_grid(cells, CURRENT_GREEN.name, DEV, "detection_rate", pct=True),
        "",
    ]
    diag = report["diagnosis"]
    lines += [f"Diagnosis ({diag['scope']}):", ""]
    lines.append(
        "- by ROI: "
        + ", ".join(
            f"{k} {100 * v['detection_rate']:.0f}% (n={v['n']})" for k, v in diag["by_roi"].items()
        )
    )
    lines.append(
        "- by dataset: "
        + ", ".join(
            f"{k} {100 * v['detection_rate']:.0f}% (n={v['n']})"
            for k, v in diag["by_dataset"].items()
        )
    )
    lines.append(
        "- by degradation: "
        + ", ".join(
            f"{k} {100 * v['detection_rate']:.0f}%" for k, v in diag["by_degradation"].items()
        )
    )
    for key, label in (
        ("by_skin_pixels", "skin pixels per ROI"),
        ("by_motion", "motion (IOD/frame)"),
        ("by_native_noise", "native in-band noise (% of mean, a=0/CRF 18 face green)"),
        ("by_bitrate", "bitrate (kbit/s)"),
    ):
        parts = [
            f"{t['tercile']} [{t['range'][0]:.4g}-{t['range'][1]:.4g}] "
            f"{100 * t['detection_rate']:.0f}%"
            for t in diag[key]
        ]
        lines.append(f"- by {label} tercile: " + "; ".join(parts))
    lines.append(
        f"- signals rejected (no estimate): {_fmt(diag['rejected'])}; median |error| "
        f"of the misses: {_fmt(diag['miss_error_median_bpm'], 1)} bpm"
    )
    lines += ["", report["diagnosis_text"], ""]

    lines += [
        "## 2. Variant selection (dev only)",
        "",
        "Score: dev detection rate pooled over amplitudes 0.2, 0.5 and 1 % at CRF 23 and "
        f"28 (MAE breaks ties); a challenger must win by more than {MIN_GAIN:g}. "
        "Null = false-detection rate at a = 0, CRF 23-28.",
        "",
    ]
    for stage in report["selection_trail"]:
        lines += [
            f"### {stage['stage']}",
            "",
            "| variant | detection | MAE (bpm) | null |",
            "|---|---|---|---|",
        ]
        for c in stage["candidates"]:
            mark = " **(chosen)**" if c["variant"] == stage["chosen"] else ""
            null = c["null_false_detection"]
            null = "n/a" if null is None else f"{100 * null:.1f}%"
            lines.append(
                f"| {c['variant']}{mark} | {100 * c['detection_rate']:.1f}% | "
                f"{_fmt(c['mae_bpm'], 1)} | {null} |"
            )
        lines.append("")
    lines += ["Selected variant:", "", "```json", json.dumps(sel, indent=2), "```", ""]

    q = report["quality"]
    lines += [
        "## 3. Quality ramps",
        "",
        "Fitted on dev estimates of the selected variant (amplitudes > 0, all degradations; "
        f"n={q['n_dev']}, {100 * q['dev_correct_fraction']:.0f}% correct). Each ramp runs from "
        "the median among incorrect to the median among correct estimates.",
        "",
        f"- spectral concentration: {q['old_ramps']['concentration_range']} -> "
        f"{q['ramps']['concentration_range']}",
        f"- SNR proxy (dB): {q['old_ramps']['snr_range_db']} -> {q['ramps']['snr_range_db']}",
        "",
        "| ROC-AUC, correct vs incorrect | dev | held-out |",
        "|---|---|---|",
    ]
    for key in ("quality_new", "quality_old", "peak_snr_db", "concentration", "prominence"):
        lines.append(f"| {key} | {_fmt(q['auc_dev'][key])} | {_fmt(q['auc_heldout'][key])} |")
    lines += [
        "",
        "Gate on the new quality score (dev / held-out): coverage, fraction correct among passed:",
        "",
    ]
    lines += [
        "| min quality | dev coverage | dev correct | held-out coverage | held-out correct |",
        "|---|---|---|---|---|",
    ]
    for a, b in zip(q["gate_dev"], q["gate_heldout"], strict=True):
        lines.append(
            f"| {a['min_quality']} | {_fmt(a['coverage'])} | {_fmt(a['correct_among_passed'])} | "
            f"{_fmt(b['coverage'])} | {_fmt(b['correct_among_passed'])} |"
        )

    lines += [
        "",
        "## 4. Held-out: current vs selected",
        "",
        "Detection rate (%), current CHROM, held-out:",
        "",
        *_grid(cells, CURRENT.name, HELDOUT, "detection_rate", pct=True),
        "",
        "Detection rate (%), selected, held-out:",
        "",
        *_grid(cells, "selected", HELDOUT, "detection_rate", pct=True),
        "",
        "MAE (bpm), current CHROM, held-out:",
        "",
        *_grid(cells, CURRENT.name, HELDOUT, "mae_bpm"),
        "",
        "MAE (bpm), selected, held-out:",
        "",
        *_grid(cells, "selected", HELDOUT, "mae_bpm"),
        "",
        "SNR at the true frequency (dB), selected, held-out:",
        "",
        *_grid(cells, "selected", HELDOUT, "snr_true_db"),
        "",
        "Paired difference selected - current (held-out, detection rate, percentage points):",
        "",
        "| condition | delta [95% CI] |",
        "|---|---|",
    ]
    for d in report["heldout_deltas"]:
        ci = d["ci"]
        text = "n/a" if d["delta"] is None else f"{100 * d['delta']:+.1f}"
        if ci:
            text += f" [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]"
        lines.append(f"| {d['condition']} | {text} |")
    lines += [
        "",
        "Held-out, by dataset (realistic amplitudes, CRF 23-28):",
        "",
        "| dataset | current | selected |",
        "|---|---|---|",
    ]
    for ds, v in report["heldout_by_dataset"].items():
        lines.append(f"| {ds} | {_fmt(v['current'])} | {_fmt(v['selected'])} |")
    lines += ["", "## Limitations", ""]
    lines += [f"- {text}" for text in report["limitations"]]
    lines += ["", f"License: {report['meta'].get('license_notes')}", ""]
    return "\n".join(lines)


LIMITATIONS = (
    "The injected pulse is spatially uniform in phase and has a fixed per-channel signature; "
    "real pulses vary in phase and strength across the face and over time.",
    "The source videos were already compressed (FF++ c23, NFI at CRF 23 after an earlier "
    "source encode), so every condition is at least a second generation; CRF 18 is not "
    "'uncompressed'.",
    "Each source clip contains its subject's own pulse. The injected HR is >= 15 bpm from the "
    "native HR estimated at a = 0 / CRF 18, but that estimate can be wrong, so a variant that "
    "locks onto the native pulse is counted as a miss.",
    "Clips are 10 s (FF++) and 8 s (NFI); results for longer clips will differ.",
    "Clips were cropped to a face-centred square (2x the face box) before re-encoding; the "
    "codec's rate control then sees less background than in the full frame.",
    "Selection used dev clips only, but the options were chosen with knowledge of the current "
    "pipeline's weaknesses, and the held-out half is small (about 50 clips in ~40 groups).",
)


def heldout_deltas(rows, boot: GroupBootstrap, a: str, b: str) -> list[dict]:
    out = []
    held = [r for r in rows if r["split"] == HELDOUT]
    pooled = (
        "realistic amplitudes (0.2-1 %), CRF 23-28",
        lambda r: (
            r["amplitude"] in REALISTIC_AMPLITUDES and r["degradation"] in SELECTION_DEGRADATIONS
        ),
    )
    conds = [pooled] + [
        (
            f"{100 * amp:g} %, {deg}",
            lambda r, amp=amp, deg=deg: r["amplitude"] == amp and r["degradation"] == deg,
        )
        for amp in AMPLITUDES
        for deg in DEGRADATION_NAMES
    ]
    for label, keep in conds:
        ra = [r for r in held if r["variant"] == a and keep(r)]
        rb = [r for r in held if r["variant"] == b and keep(r)]
        out.append({"condition": label, **boot.paired_delta(ra, rb, "detection_rate")})
    return out


def run_benchmark(
    clips: Sequence[BenchmarkClip],
    cache_dir: Path,
    out_dir: Path,
    stamp: str,
    workers: int = 1,
    n_boot: int = 1000,
    meta: Mapping | None = None,
    old_quality: Mapping | None = None,
) -> list[Path]:
    from datetime import UTC, datetime

    from pulsecheck.features.vector import QualityConfig

    built = {c.clip_id: load_meta(cache_dir, c.clip_id) for c in clips}
    ok = [c for c in clips if built[c.clip_id] and built[c.clip_id].get("status") == "ok"]
    failed = {
        c.clip_id: (built[c.clip_id] or {}).get("reason", "missing") for c in clips if c not in ok
    }
    log.info("%d clips built, %d failed or missing", len(ok), len(failed))
    dev = [c for c in ok if c.split == DEV]
    held = [c for c in ok if c.split == HELDOUT]

    selected, trail = select_variant(dev, cache_dir, workers)
    final = Variant(**{**selected.describe(), "name": "selected", "signal": selected.signal})
    variants = [CURRENT, CURRENT_GREEN, final]
    rows = evaluate(variants, ok, cache_dir, ALL_CONDITIONS, workers)
    boot = {
        DEV: GroupBootstrap([c.group for c in dev], n_boot, seed=0),
        HELDOUT: GroupBootstrap([c.group for c in held], n_boot, seed=1),
    }
    names = {v.name for v in variants}
    cells = cell_table(rows, boot, names, datasets=("all", "ffpp", "nfi"))

    old = {k: list(v) for k, v in (old_quality or vars(QualityConfig())).items()}
    quality = calibrate_quality(
        _select(rows, variant="selected", split=DEV),
        _select(rows, variant="selected", split=HELDOUT),
        old,
    )
    diag = diagnose(rows, CURRENT.name)
    by_dataset = {}
    for ds in ("ffpp", "nfi"):
        pool = [
            r
            for r in rows
            if r["split"] == HELDOUT
            and r["dataset"] == ds
            and r["amplitude"] in REALISTIC_AMPLITUDES
            and r["degradation"] in SELECTION_DEGRADATIONS
        ]
        by_dataset[ds] = {
            "current": _mean_detect(pool, CURRENT.name),
            "selected": _mean_detect(pool, "selected"),
        }
    report = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "meta": dict(meta or {}),
        "n_boot": n_boot,
        "tolerance_bpm": TOLERANCE_BPM,
        "injection": (
            "multiplicative, skin only (face oval minus eyes, brows, lips), weight 1 on "
            "forehead/cheeks and 0.5 elsewhere, BVP signature (R, G, B) = (0.43, 1, 0.69), "
            "two-Gaussian PPG waveform, HR0 50-120 bpm with +/-3 bpm slow variability, "
            "stochastic rounding; amplitude = peak-to-peak relative change of green. "
            "Degradations: libx264 CRF 18/23/28/35 (preset medium, yuv420p) and CRF 23 at "
            "half resolution."
        ),
        "data": data_summary(ok, built, failed),
        "injection_transfer": injection_transfer(ok, cache_dir, workers),
        "selected": selected.describe(),
        "selection_trail": trail,
        "diagnosis": diag,
        "diagnosis_text": diagnosis_text(diag, cells),
        "quality": quality,
        "cells": cells,
        "heldout_deltas": heldout_deltas(rows, boot[HELDOUT], "selected", CURRENT.name),
        "heldout_by_dataset": by_dataset,
        "limitations": list(LIMITATIONS),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"benchmark-{stamp}"
    json_path = stem.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=1, default=_json), encoding="utf-8")
    md_path = stem.with_suffix(".md")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    csv_path = stem.with_suffix(".csv")
    _write_cells(csv_path, cells)
    selection_path = out_dir / f"selection-{stamp}.json"
    selection_path.write_text(
        json.dumps({"selected": selected.describe(), "quality": quality["ramps"]}, indent=2),
        encoding="utf-8",
    )
    return [md_path, json_path, csv_path, selection_path]


def _mean_detect(rows, variant) -> float | None:
    pool = [r["detected"] for r in rows if r["variant"] == variant]
    return float(np.mean(pool)) if pool else None


def diagnosis_text(diag: Mapping, cells: Sequence[Mapping]) -> str:
    """Short plain-language reading of the diagnosis numbers."""

    def best(entries):
        return max(entries, key=lambda t: t["detection_rate"]) if entries else None

    parts = []
    for key, label in (
        ("by_skin_pixels", "skin pixels"),
        ("by_motion", "motion"),
        ("by_native_noise", "native in-band noise"),
    ):
        ter = diag[key]
        if len(ter) == 3:
            parts.append(
                f"{label}: {100 * ter[0]['detection_rate']:.0f}% (low tercile) vs "
                f"{100 * ter[2]['detection_rate']:.0f}% (high tercile)"
            )
    return (
        "Detection by tercile (current CHROM, dev, realistic amplitudes, CRF 23-28): "
        + "; ".join(parts)
        + "."
    )


def data_summary(clips, built, failed) -> dict:
    groups = defaultdict(list)
    for c in clips:
        groups[(c.dataset, c.split)].append(built[c.clip_id])
    summary = []
    for (ds, split), metas in sorted(groups.items()):
        summary.append(
            {
                "dataset": ds,
                "split": split,
                "clips": len(metas),
                "groups": len({m["clip"]["group"] for m in metas}),
                "source_height": float(np.median([m["source"]["height"] for m in metas])),
                "iod_px": float(np.median([m["source"]["interocular_px_median"] for m in metas])),
                "motion": float(
                    np.median([m["motion"]["median_iod_per_frame"] or 0 for m in metas])
                ),
            }
        )
    bitrate: dict = defaultdict(dict)
    face_px: dict = defaultdict(dict)
    for ds in sorted({c.dataset for c in clips}):
        metas = [built[c.clip_id] for c in clips if c.dataset == ds]
        for d in DEGRADATION_NAMES:
            key = condition_key(0.0, d)
            bitrate[ds][d] = float(np.median([m["conditions"][key]["bitrate_kbps"] for m in metas]))
            face_px[ds][d] = float(
                np.median([m["conditions"][key]["median_skin_pixels"][FACE_ROI] for m in metas])
            )
    separated = [built[c.clip_id].get("target_separated") for c in clips]
    return {
        "summary": summary,
        "failed": failed,
        "bitrate_kbps": dict(bitrate),
        "face_pixels": dict(face_px),
        "target_not_separated": int(sum(1 for s in separated if s is False)),
    }


def load_meta(cache_dir: Path, clip_id: str) -> dict | None:
    from pulsecheck.benchmark.cache import cache_name

    path = Path(cache_dir) / f"{cache_name(clip_id)}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _json(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(type(value).__name__)


def _write_cells(path: Path, cells: Iterable[Mapping]) -> None:
    cells = list(cells)
    fields = ["variant", "split", "dataset", "amplitude", "degradation", "n", "n_groups"]
    for m in ("detection_rate", "estimate_rate", "mae_bpm", "rmse_bpm", "snr_true_db"):
        fields += [m, f"{m}_lo", f"{m}_hi"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for c in cells:
            row = {k: c.get(k) for k in fields[:7]}
            for m in ("detection_rate", "estimate_rate", "mae_bpm", "rmse_bpm", "snr_true_db"):
                row[m] = c.get(m)
                ci = c.get(f"{m}_ci") or [None, None]
                row[f"{m}_lo"], row[f"{m}_hi"] = ci
            writer.writerow(row)

"""UBFC-rPPG ground-truth benchmark (PRD FR-12, item 2).

UBFC-rPPG (Bobbia, Macwan, Benezeth, Mansouri & Dubois 2019, "Unsupervised
skin tissue segmentation for remote photoplethysmography", Pattern
Recognition Letters 124) records faces with a webcam and a contact pulse
oximeter. It must be requested from its authors and used under their terms,
so it is never downloaded here. Expected layout ("DATASET_2"):

    data/UBFC-rPPG/subject1/vid.avi
    data/UBFC-rPPG/subject1/ground_truth.txt
    ...

ground_truth.txt has three whitespace-separated lines: the PPG trace, the
oximeter's heart rate (bpm) and the sample timestamps (s). The reference HR
of an analysis window is the dominant frequency of the PPG trace in that
window (0.7-3.0 Hz), which is how most rPPG work scores UBFC; the
oximeter's own HR line is kept for reference.

Videos are traced once into `<cache>/<subject>.npz` (per-frame ROI colour
means for every spatial variant, as in the injection benchmark), then each
variant estimates HR over sliding windows. Metrics: MAE, RMSE and the
fraction of windows within 5 bpm, with subject-bootstrap intervals.
"""

from __future__ import annotations

import json
import logging
import math
import multiprocessing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal as sps

from pulsecheck.benchmark.variants import TOLERANCE_BPM, Variant, estimate

log = logging.getLogger("pulsecheck.benchmark")

WINDOW_S = 10.0
STEP_S = 5.0
PPG_BAND_HZ = (0.7, 3.0)
LICENSE_NOTE = (
    "UBFC-rPPG: obtained from the dataset authors under their terms (research use); "
    "do not redistribute videos, frames or derived per-subject data."
)


@dataclass(frozen=True)
class UbfcSubject:
    subject: str
    video: Path
    ground_truth: Path


@dataclass
class GroundTruth:
    ppg: np.ndarray
    hr_bpm: np.ndarray  # the oximeter's HR line
    t: np.ndarray  # seconds


def parse_ground_truth(path: str | Path) -> GroundTruth:
    """Read a DATASET_2 ground_truth.txt (PPG, HR and time lines)."""
    lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) < 3:
        raise ValueError(f"{path}: expected 3 lines (PPG, HR, time), got {len(lines)}")
    ppg, hr, t = (np.array([float(v) for v in ln.split()]) for ln in lines[:3])
    if not (len(ppg) == len(hr) == len(t)):
        raise ValueError(f"{path}: line lengths differ ({len(ppg)}, {len(hr)}, {len(t)})")
    if len(t) > 1 and not np.all(np.diff(t) > 0):
        raise ValueError(f"{path}: timestamps are not increasing")
    return GroundTruth(ppg=ppg, hr_bpm=hr, t=t)


def find_subjects(root: str | Path) -> list[UbfcSubject]:
    """Subjects with both vid.avi and ground_truth.txt, in natural order."""
    root = Path(root)
    out = []
    for folder in root.glob("subject*"):
        video, gt = folder / "vid.avi", folder / "ground_truth.txt"
        if video.is_file() and gt.is_file():
            out.append(UbfcSubject(folder.name, video, gt))

    def key(s: UbfcSubject):
        digits = "".join(ch for ch in s.subject if ch.isdigit())
        return (int(digits) if digits else math.inf, s.subject)

    return sorted(out, key=key)


def windows(duration_s: float, length_s: float = WINDOW_S, step_s: float = STEP_S):
    """(start, end) analysis windows covering [0, duration_s]."""
    start = 0.0
    out = []
    while start + length_s <= duration_s + 1e-9:
        out.append((start, start + length_s))
        start += step_s
    return out


def reference_hr(gt: GroundTruth, start: float, end: float) -> float | None:
    """Dominant PPG frequency in [start, end), in bpm (zero-padded periodogram)."""
    keep = (gt.t >= start) & (gt.t < end)
    if keep.sum() < 16:
        return None
    t, x = gt.t[keep], gt.ppg[keep]
    fs = (len(t) - 1) / (t[-1] - t[0])
    nfft = max(len(x), 1 << int(math.ceil(math.log2(fs / 0.01))))
    f, p = sps.periodogram(sps.detrend(x), fs=fs, window="hann", nfft=nfft)
    band = (f >= PPG_BAND_HZ[0]) & (f <= PPG_BAND_HZ[1])
    if not band.any() or p[band].max() <= 0:
        return None
    return 60.0 * float(f[band][int(np.argmax(p[band]))])


def _trace_subject(task) -> dict:
    from pulsecheck.benchmark.cache import trace_video

    subject, cache_dir, model_path = task
    out = Path(cache_dir) / f"{subject.subject}.npz"
    if out.is_file():
        return {"subject": subject.subject, "status": "cached"}
    try:
        arrays, diag = trace_video(subject.video, model_path)
    except Exception as exc:  # noqa: BLE001 - reported per subject
        return {"subject": subject.subject, "status": "error", "reason": str(exc)[:200]}
    part = out.with_name(out.stem + ".part.npz")
    np.savez_compressed(part, **arrays)
    part.replace(out)
    out.with_suffix(".json").write_text(json.dumps(diag), encoding="utf-8")
    return {"subject": subject.subject, "status": "ok"}


def evaluate_subject(subject: UbfcSubject, arrays: dict, variants: list[Variant]) -> list[dict]:
    gt = parse_ground_truth(subject.ground_truth)
    t = arrays["t"]
    rows = []
    for start, end in windows(min(float(t[-1]), float(gt.t[-1] - gt.t[0]))):
        ref = reference_hr(gt, gt.t[0] + start, gt.t[0] + end)
        keep = (t >= start) & (t < end)
        if ref is None or keep.sum() < 30:
            continue
        for variant in variants:
            traces = {key: arrays[f"{key}|rgb"][keep] for _, key in variant.roi_keys()}
            for est in estimate(variant, t[keep], traces):
                error = None if est.hr_bpm is None else est.hr_bpm - ref
                rows.append(
                    {
                        "variant": variant.name,
                        "group": subject.subject,
                        "window_start_s": start,
                        "roi": est.roi,
                        "hr_ref": ref,
                        "hr_est": est.hr_bpm,
                        "error": error,
                        "detected": error is not None and abs(error) <= TOLERANCE_BPM,
                        "snr_true_db": None,
                    }
                )
    return rows


def ubfc_variants(selected: Variant) -> list[Variant]:
    from pulsecheck.benchmark.run import CURRENT, CURRENT_GREEN

    same = {k: v for k, v in selected.describe().items() if k not in ("name", "method")}
    same["signal"] = selected.signal
    out = [
        CURRENT,
        CURRENT_GREEN,
        Variant(**{**same, "name": "selected", "method": selected.method}),
    ]
    out += [
        Variant(**{**same, "name": f"selected-settings-{m}", "method": m})
        for m in ("green", "chrom", "pos")
        if m != selected.method
    ]
    return out


def run_ubfc(
    root: Path,
    cache_dir: Path,
    out_dir: Path,
    stamp: str,
    selection: Path | None = None,
    workers: int = 1,
    model_path: str | None = None,
) -> list[Path]:
    from pulsecheck.benchmark.metrics import GroupBootstrap

    subjects = find_subjects(root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(s, str(cache_dir), model_path) for s in subjects]
    if workers <= 1:
        statuses = [_trace_subject(t) for t in tasks]
    else:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            statuses = pool.map(_trace_subject, tasks, chunksize=1)
    for status in statuses:
        log.info("UBFC %s: %s %s", status["subject"], status["status"], status.get("reason", ""))
    if selection is None:
        found = sorted(out_dir.glob("selection-*.json"))
        selection = found[-1] if found else None
    selected = None
    if selection is not None:
        sel = json.loads(Path(selection).read_text(encoding="utf-8"))["selected"]
        sel["signal"] = tuple(sorted(sel["signal"].items()))
        selected = Variant(**sel)
    from pulsecheck.benchmark.run import CURRENT

    variants = ubfc_variants(selected or CURRENT)
    rows = []
    for subject in subjects:
        path = cache_dir / f"{subject.subject}.npz"
        if not path.is_file():
            continue
        with np.load(path) as data:
            arrays = {k: data[k] for k in data.files}
        rows += evaluate_subject(subject, arrays, variants)
    boot = GroupBootstrap([r["group"] for r in rows], 1000, seed=2)
    results = {
        v.name: boot.summarize([r for r in rows if r["variant"] == v.name]) for v in variants
    }
    report = {
        "dataset": "UBFC-rPPG DATASET_2",
        "subjects": len(subjects),
        "windows": {"length_s": WINDOW_S, "step_s": STEP_S},
        "reference": "dominant PPG frequency per window (0.7-3.0 Hz)",
        "selection": str(selection) if selection else None,
        "variants": {v.name: v.describe() for v in variants},
        "results": results,
        "license_notes": LICENSE_NOTE,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"ubfc-{stamp}"
    stem.with_suffix(".json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    lines = [
        "# UBFC-rPPG ground-truth benchmark (PRD FR-12)",
        "",
        f"{len(subjects)} subjects, {WINDOW_S:g} s windows every {STEP_S:g} s; reference HR = "
        "dominant PPG frequency per window. Brackets: 95% subject-bootstrap intervals.",
        "",
        "| variant | windows | MAE (bpm) | RMSE (bpm) | within 5 bpm |",
        "|---|---|---|---|---|",
    ]
    for name, r in results.items():
        if r.get("mae_bpm") is None:
            lines.append(f"| {name} | {r['n']} | n/a | n/a | n/a |")
            continue
        ci = r["detection_rate_ci"] or [math.nan, math.nan]
        lines.append(
            f"| {name} | {r['n']} | {r['mae_bpm']:.1f} | {r['rmse_bpm']:.1f} | "
            f"{100 * r['detection_rate']:.0f}% [{100 * ci[0]:.0f}, {100 * ci[1]:.0f}] |"
        )
    lines += ["", f"License: {LICENSE_NOTE}", ""]
    stem.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    return [stem.with_suffix(".md"), stem.with_suffix(".json")]

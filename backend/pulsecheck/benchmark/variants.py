"""Pipeline variants evaluated on the cached benchmark traces (PRD FR-12).

A Variant fixes everything after the face landmarks: which ROI traces are
used (layout, statistic, landmark smoothing), how a heart rate is read from
them (`hr_source`) and the signal settings (pulse method, detrending, filter,
Welch window, peak picking). Spatial options were traced into the cache, so
evaluating a variant only re-runs the signal processing.

hr_source:
- "per_roi": every regional ROI gives its own estimate (the current
  features: each ROI's dominant frequency is reported separately);
- "fused": one estimate from the average of the regional ROIs' unit-power
  spectra (pulsecheck.signals.processing.fuse_spectra);
- "face": the whole-face skin ROI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np

from pulsecheck.face.regions import FACE_ROI, ROI_NAMES
from pulsecheck.signals.processing import (
    SignalConfig,
    SignalRejected,
    dominant_peak,
    fuse_spectra,
    process_trace,
    snr_at_frequency,
)

HR_SOURCES = ("per_roi", "fused", "face")
TOLERANCE_BPM = 5.0


@dataclass(frozen=True)
class Variant:
    name: str
    method: str = "chrom"
    layout: str = "standard"
    hr_source: str = "per_roi"
    statistic: str = "mean"
    smoothing: int = 0
    signal: tuple[tuple[str, object], ...] = ()  # SignalConfig overrides

    def __post_init__(self) -> None:
        if self.hr_source not in HR_SOURCES:
            raise ValueError(f"hr_source must be one of {HR_SOURCES}")
        self.signal_config()  # validates the overrides

    def signal_config(self) -> SignalConfig:
        return SignalConfig(**dict(self.signal))

    def with_signal(self, name: str, **overrides) -> Variant:
        merged = {**dict(self.signal), **overrides}
        defaults = SignalConfig().to_dict()
        kept = tuple(sorted((k, v) for k, v in merged.items() if defaults.get(k) != v))
        return replace(self, name=name, signal=kept)

    def roi_keys(self) -> list[tuple[str, str]]:
        """(label, cache key) of the traces this variant reads."""
        base = f"s{self.smoothing}/{self.statistic}/"
        if self.hr_source == "face":
            return [(FACE_ROI, base + FACE_ROI)]
        prefix = "large:" if self.layout == "large" else ""
        return [(roi, f"{base}{prefix}{roi}") for roi in ROI_NAMES]

    def describe(self) -> dict:
        data = asdict(self)
        data["signal"] = dict(self.signal)
        return data


@dataclass
class Estimate:
    roi: str  # ROI label, "fused" or "face"
    hr_bpm: float | None
    peak_snr_db: float | None = None  # the pipeline's SNR proxy at its own peak
    concentration: float | None = None
    prominence: float | None = None
    reason: str | None = None  # why there is no estimate
    spectrum: object = field(default=None, repr=False)


def estimate(variant: Variant, t: np.ndarray, traces: dict[str, np.ndarray]) -> list[Estimate]:
    """Heart-rate estimates of one variant from one condition's cached traces."""
    config = variant.signal_config()
    finite = t[np.isfinite(t)]
    span = (float(finite[0]), float(finite[-1])) if finite.size >= 2 else None
    signals = {
        label: process_trace(t, traces[key], variant.method, config, span, stability=False)
        for label, key in variant.roi_keys()
    }
    if variant.hr_source in ("per_roi", "face"):
        return [_from_signal(label, sig) for label, sig in signals.items()]
    ok = [s for s in signals.values() if s.ok and s.spectrum is not None]
    if not ok:
        reasons = sorted({s.reason or "rejected" for s in signals.values()})
        return [Estimate("fused", None, reason=",".join(reasons))]
    try:
        fused = fuse_spectra([s.spectrum for s in ok])
        peak = dominant_peak(
            fused, config.peak_halfwidth_hz, config.peak_picking, config.harmonic_weight
        )
    except SignalRejected as exc:
        return [Estimate("fused", None, reason=exc.reason)]
    return [
        Estimate(
            "fused",
            60.0 * peak.frequency_hz,
            peak.snr_db,
            peak.concentration,
            peak.prominence,
            spectrum=fused,
        )
    ]


def _from_signal(label: str, sig) -> Estimate:
    if not sig.ok or sig.peak is None:
        return Estimate(label, None, reason=sig.reason)
    return Estimate(
        label,
        60.0 * sig.peak.frequency_hz,
        sig.peak.snr_db,
        sig.peak.concentration,
        sig.peak.prominence,
        spectrum=sig.spectrum,
    )


def score_estimate(est: Estimate, true_bpm: float) -> dict:
    """Error, detection (within TOLERANCE_BPM) and the SNR at the true frequency."""
    if est.hr_bpm is None:
        return {"hr_est": None, "error": None, "detected": False, "snr_true_db": None}
    error = est.hr_bpm - true_bpm
    return {
        "hr_est": est.hr_bpm,
        "error": error,
        "detected": abs(error) <= TOLERANCE_BPM,
        "snr_true_db": snr_at_frequency(est.spectrum, true_bpm / 60.0),
    }

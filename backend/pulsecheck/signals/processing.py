"""Pulse signal processing for per-ROI colour traces (PRD FR-5).

Pipeline for one ROI:

1. Validate timestamps and samples, then resample the valid samples onto a
   uniform grid by linear interpolation (missing frames are NaN in the input).
2. Normalise each channel by its mean and detrend it (linear, constant, or
   smoothness priors, Tarvainen et al. 2002).
3. Build a pulse signal: the green channel, CHROM (de Haan & Jeanne 2013) or
   POS (Wang et al. 2017).
4. Band-pass it (zero-phase Butterworth, default 0.7-3.0 Hz) when the sample
   rate and length support the filter.
5. Estimate a Welch power spectrum and summarise its dominant peak, picked
   either as the strongest bin ("argmax") or by a harmonic sum that favours
   a frequency whose second harmonic is also present ("harmonic").

Signals that are too short, sampled too slowly or mostly missing are rejected
with a reason code and a human-readable detail instead of being analysed.
Every result records the settings it was produced with.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields

import numpy as np
from scipy import signal as sps

PULSE_METHODS = ("green", "chrom", "pos")
DETREND_MODES = ("linear", "constant", "none", "smoothness_priors")
PEAK_PICKING = ("argmax", "harmonic")
# Caps the SNR proxy when the off-peak band power is zero (a pure tone).
MAX_SNR_DB = 60.0


@dataclass(frozen=True)
class SignalConfig:
    band_hz: tuple[float, float] = (0.7, 3.0)  # pass band, roughly 42-180 bpm
    filter_order: int = 4  # Butterworth order of each band edge
    resample_hz: float | None = None  # None: the median observed frame rate
    max_resample_hz: float = 60.0
    min_duration_s: float = 6.0  # about four cycles at the lower band edge
    min_sample_rate_hz: float = 7.5  # 2.5x the default upper band edge
    min_valid_fraction: float = 0.7  # of frames with a valid sample
    max_gap_s: float = 2.0  # longest run of missing samples that is interpolated
    detrend: str = "linear"
    # smoothness_priors detrending: high-pass corner (half amplitude) in Hz.
    detrend_cutoff_hz: float = 0.4
    chrom_window_s: float = 1.6  # CHROM overlap-add window (de Haan used 32 frames at 20 fps)
    pos_window_s: float = 1.6  # POS window (Wang et al. 2017 used 1.6 s)
    welch_window_s: float = 10.0  # Welch segment; a shorter signal is one segment
    spectrum_resolution_hz: float = 0.01  # zero-padded bin spacing
    peak_halfwidth_hz: float = 0.1  # spectral window counted as "the peak"
    stability_window_s: float = 6.0
    stability_step_s: float = 1.0
    peak_picking: str = "argmax"
    harmonic_weight: float = 0.5  # "harmonic": score P(f) + weight * P(2f)

    def __post_init__(self) -> None:
        low, high = self.band_hz
        if not 0 < low < high:
            raise ValueError(f"band_hz must satisfy 0 < low < high, got {self.band_hz}")
        if self.detrend not in DETREND_MODES:
            raise ValueError(f"detrend must be one of {DETREND_MODES}, got {self.detrend!r}")
        if self.filter_order < 1:
            raise ValueError("filter_order must be at least 1")
        if self.peak_picking not in PEAK_PICKING:
            raise ValueError(f"peak_picking must be one of {PEAK_PICKING}")
        if self.detrend_cutoff_hz <= 0:
            raise ValueError("detrend_cutoff_hz must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping | None) -> SignalConfig:
        """Build from a config section (e.g. configs/default.yaml `signal:`)."""
        values = dict(values or {})
        if "bandpass_hz" in values:
            values["band_hz"] = values.pop("bandpass_hz")
        if "band_hz" in values:
            values["band_hz"] = tuple(float(v) for v in values["band_hz"])
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown signal settings: {sorted(unknown)}")
        return cls(**values)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["band_hz"] = list(self.band_hz)
        return data


@dataclass
class Spectrum:
    freqs_hz: np.ndarray
    power: np.ndarray  # power spectral density, arbitrary units
    band_hz: tuple[float, float]

    def in_band(self) -> np.ndarray:
        low, high = self.band_hz
        return (self.freqs_hz >= low) & (self.freqs_hz <= high)


@dataclass
class PeakStats:
    frequency_hz: float
    # Margin of the peak over the next-strongest in-band peak outside
    # +/- peak_halfwidth_hz, relative to the peak height, 0-1 (1: no rival).
    prominence: float
    # Fraction of in-band power within +/- peak_halfwidth_hz of the peak, 0-1.
    concentration: float
    # Peak power over the median in-band power, in dB, capped at MAX_SNR_DB.
    snr_db: float


@dataclass
class ProcessedSignal:
    method: str
    ok: bool
    reason: str | None = None  # machine-readable rejection code
    detail: str | None = None  # human-readable rejection explanation
    fs_hz: float | None = None
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))  # uniform grid, seconds
    signal: np.ndarray = field(default_factory=lambda: np.zeros(0))  # unit-variance pulse
    spectrum: Spectrum | None = None
    peak: PeakStats | None = None
    window_freqs_hz: np.ndarray = field(default_factory=lambda: np.zeros(0))
    quality: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)

    def summary(self) -> dict:
        """JSON-serialisable summary without the sample arrays."""
        return {
            "method": self.method,
            "ok": self.ok,
            "reason": self.reason,
            "detail": self.detail,
            "fs_hz": self.fs_hz,
            "peak": asdict(self.peak) if self.peak else None,
            "window_freqs_hz": self.window_freqs_hz.tolist(),
            "quality": self.quality,
            "settings": self.settings,
        }


class SignalRejected(Exception):
    def __init__(self, reason: str, detail: str, quality: dict | None = None) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail
        self.quality = quality or {}


@dataclass
class UniformSamples:
    t: np.ndarray  # (M,) uniform grid
    values: np.ndarray  # (M, C) interpolated samples
    fs_hz: float
    quality: dict


def resample_uniform(
    timestamps_s: np.ndarray,
    values: np.ndarray,
    config: SignalConfig,
    span_s: tuple[float, float] | None = None,
) -> UniformSamples:
    """Validate a sampled series and interpolate it onto a uniform time grid.

    `values` is (N,) or (N, C); a row with any NaN counts as missing. The grid
    covers `span_s` (default: first to last finite timestamp), so several ROIs
    of one video can share it. Raises SignalRejected with a reason code.
    """
    t = np.asarray(timestamps_s, dtype=np.float64).ravel()
    x = np.asarray(values, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"values must be (N,) or (N, C), got shape {x.shape}")
    if len(t) != len(x):
        raise ValueError(f"{len(t)} timestamps but {len(x)} samples")

    finite_t = np.isfinite(t)
    t, x = t[finite_t], x[finite_t]
    order = np.argsort(t, kind="stable")
    t, x = t[order], x[order]
    keep = np.diff(t, prepend=-np.inf) > 0  # drop duplicate timestamps
    t, x = t[keep], x[keep]
    quality: dict = {
        "input_samples": int(finite_t.size),
        "dropped_timestamps": int(finite_t.size - t.size),
    }
    if t.size < 2:
        raise SignalRejected(
            "invalid_timestamps", "Fewer than two distinct, finite timestamps", quality
        )

    observed_fs = 1.0 / float(np.median(np.diff(t)))
    quality["observed_fs_hz"] = observed_fs
    if not math.isfinite(observed_fs) or observed_fs < config.min_sample_rate_hz:
        raise SignalRejected(
            "invalid_sample_rate",
            f"Sample rate {observed_fs:.2f} Hz is below the minimum "
            f"{config.min_sample_rate_hz:.2f} Hz needed for the {config.band_hz[1]:g} Hz band edge",
            quality,
        )

    start, end = span_s if span_s is not None else (float(t[0]), float(t[-1]))
    duration = end - start
    quality["duration_s"] = duration
    if not duration >= config.min_duration_s:
        raise SignalRejected(
            "too_short",
            f"Signal lasts {duration:.2f} s; at least {config.min_duration_s:g} s is required",
            quality,
        )

    valid = np.isfinite(x).all(axis=1)
    in_span = (t >= start) & (t <= end)
    valid_fraction = float(valid[in_span].mean()) if in_span.any() else 0.0
    quality["valid_fraction"] = valid_fraction
    if valid.sum() < 2 or valid_fraction < config.min_valid_fraction:
        raise SignalRejected(
            "insufficient_valid_samples",
            f"Only {valid_fraction:.0%} of samples are valid; "
            f"at least {config.min_valid_fraction:.0%} is required",
            quality,
        )

    tv, xv = t[valid], x[valid]
    gaps = np.diff(np.concatenate([[start], tv, [end]]))
    longest_gap = float(gaps.max())
    quality["longest_gap_s"] = longest_gap
    if longest_gap > config.max_gap_s:
        raise SignalRejected(
            "gap_too_long",
            f"Longest run without a valid sample is {longest_gap:.2f} s; "
            f"at most {config.max_gap_s:g} s is interpolated",
            quality,
        )

    fs = config.resample_hz or min(observed_fs, config.max_resample_hz)
    n = int(math.floor(duration * fs)) + 1
    grid = start + np.arange(n) / fs
    resampled = np.column_stack([np.interp(grid, tv, xv[:, c]) for c in range(x.shape[1])])
    # A grid point is "filled" when no valid sample lies within one frame of it.
    idx = np.clip(np.searchsorted(tv, grid), 1, tv.size - 1)
    nearest = np.minimum(np.abs(tv[idx] - grid), np.abs(tv[idx - 1] - grid))
    quality["filled_fraction"] = float(np.mean(nearest > 1.0 / observed_fs))
    quality["resampled_fs_hz"] = fs
    quality["resampled_samples"] = n
    return UniformSamples(t=grid, values=resampled, fs_hz=fs, quality=quality)


def smoothness_priors_detrend(x: np.ndarray, fs_hz: float, cutoff_hz: float) -> np.ndarray:
    """Smoothness-priors detrending (Tarvainen, Ranta-aho & Karjalainen 2002), along axis 0.

    Removes the trend (I + lambda^2 D2' D2)^-1 z, where D2 is the
    second-difference operator. Away from the ends this is a zero-phase
    high-pass with gain x / (1 + x), x = lambda^2 (2 sin(pi f / fs))^4, so
    lambda = (fs / (2 pi cutoff_hz))^2 puts the half-amplitude point at cutoff_hz.
    """
    from scipy import sparse
    from scipy.sparse.linalg import splu

    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n < 3:
        return x - x.mean(axis=0)
    lam = (fs_hz / (2 * math.pi * cutoff_hz)) ** 2
    d2 = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
    system = (sparse.identity(n) + lam**2 * (d2.T @ d2)).tocsc()
    trend = splu(system).solve(x.reshape(n, -1)).reshape(x.shape)
    return x - trend


def normalize_detrend(
    x: np.ndarray, mode: str = "linear", fs_hz: float | None = None, cutoff_hz: float = 0.4
) -> np.ndarray:
    """Divide each channel by its mean (relative change around 0), then detrend."""
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0)
    if np.any(np.abs(mean) < 1e-12):
        raise SignalRejected("invalid_values", "A channel has zero mean and cannot be normalised")
    rel = x / mean - 1.0
    if mode == "none":
        return rel
    if mode == "smoothness_priors":
        if not fs_hz:
            raise ValueError("smoothness_priors detrending needs the sample rate")
        return smoothness_priors_detrend(rel, fs_hz, cutoff_hz)
    return sps.detrend(rel, axis=0, type=mode)


def bandpass(
    x: np.ndarray, fs_hz: float, band_hz: tuple[float, float], order: int = 4
) -> tuple[np.ndarray, dict]:
    """Zero-phase Butterworth band-pass along axis 0, when it is supported.

    Returns the (possibly unfiltered) signal and a record of what was done.
    The filter is skipped, with a reason, when the upper edge is not below
    Nyquist or the signal is too short for the filter's edge padding.
    """
    low, high = band_hz
    nyquist = fs_hz / 2.0
    info = {
        "type": "butterworth_bandpass",
        "order": order,
        "band_hz": [low, high],
        "zero_phase": True,
        "edge_handling": "odd extension (scipy sosfiltfilt)",
        "applied": False,
        "reason": None,
    }
    if high >= 0.95 * nyquist:
        info["reason"] = f"upper edge {high:g} Hz is too close to Nyquist ({nyquist:.2f} Hz)"
        return x, info
    sos = sps.butter(order, [low, high], btype="bandpass", fs=fs_hz, output="sos")
    padlen = 3 * (2 * len(sos) + 1)
    info["padlen_samples"] = padlen
    min_samples = max(padlen + 1, int(math.ceil(2.0 * fs_hz / low)))  # two low-edge cycles
    if len(x) < min_samples:
        info["reason"] = f"{len(x)} samples is fewer than the {min_samples} the filter needs"
        return x, info
    info["applied"] = True
    return sps.sosfiltfilt(sos, x, axis=0, padlen=padlen), info


def _standardize(x: np.ndarray) -> np.ndarray:
    std = float(np.std(x))
    if not math.isfinite(std) or std < 1e-12:
        raise SignalRejected("flat_signal", "Signal has no measurable variation")
    return (x - x.mean()) / std


def chrom_combine(xf: np.ndarray, yf: np.ndarray, fs_hz: float, window_s: float) -> np.ndarray:
    """CHROM S = Xf - alpha*Yf with alpha = std(Xf)/std(Yf) per Hann window, overlap-added.

    Falls back to a single global alpha when the signal is shorter than two windows.
    S runs in antiphase to the green channel for a typical skin pulse.
    """
    win = int(round(window_s * fs_hz))
    win += win % 2
    if win < 4 or 2 * win > len(xf):
        alpha = np.std(xf) / max(np.std(yf), 1e-12)
        return xf - alpha * yf
    hop = win // 2
    hann = np.hanning(win)
    out = np.zeros(len(xf))
    for start in range(0, len(xf) - win + 1, hop):
        seg = slice(start, start + win)
        alpha = np.std(xf[seg]) / max(np.std(yf[seg]), 1e-12)
        s = xf[seg] - alpha * yf[seg]
        out[seg] += hann * (s - s.mean())
    return out


def pos_combine(rgb: np.ndarray, fs_hz: float, window_s: float) -> np.ndarray:
    """POS (Wang, den Brinker, Stuijk & de Haan 2017, "Algorithmic principles of
    remote PPG", IEEE TBME 64(7)), overlap-added over sliding windows.

    In each window the channels are divided by their window mean, projected
    onto the plane orthogonal to the skin tone (S1 = G - B, S2 = G + B - 2R)
    and combined as h = S1 + (std S1 / std S2) S2. `rgb` holds unit-mean
    channels (columns R, G, B). The sign is flipped so that, like CHROM, the
    output runs in antiphase to the green channel. One window covers the
    whole signal when it is shorter than a window.
    """
    n = len(rgb)
    win = int(round(window_s * fs_hz))
    if win < 4 or win > n:
        win = n
    # Every window start (Wang et al. slide by one sample): (starts, 3, win).
    segs = np.lib.stride_tricks.sliding_window_view(rgb, win, axis=0)
    mean = segs.mean(axis=2, keepdims=True)
    cn = segs / np.where(np.abs(mean) < 1e-12, 1.0, mean)
    s1 = cn[:, 1] - cn[:, 2]
    s2 = cn[:, 1] + cn[:, 2] - 2 * cn[:, 0]
    alpha = s1.std(axis=1) / np.maximum(s2.std(axis=1), 1e-12)
    h = s1 + alpha[:, None] * s2
    h -= h.mean(axis=1, keepdims=True)
    out = np.zeros(n)
    starts = h.shape[0]
    for k in range(win):  # overlap-add
        out[k : k + starts] += h[:, k]
    return -out


def power_spectrum(x: np.ndarray, fs_hz: float, config: SignalConfig) -> Spectrum:
    """Welch power spectrum, zero-padded to roughly spectrum_resolution_hz bins."""
    nperseg = min(len(x), max(8, int(round(config.welch_window_s * fs_hz))))
    nfft = max(nperseg, 1 << int(math.ceil(math.log2(fs_hz / config.spectrum_resolution_hz))))
    freqs, power = sps.welch(x, fs=fs_hz, window="hann", nperseg=nperseg, nfft=nfft)
    return Spectrum(freqs_hz=freqs, power=power, band_hz=config.band_hz)


def dominant_peak(
    spectrum: Spectrum,
    halfwidth_hz: float,
    picking: str = "argmax",
    harmonic_weight: float = 0.5,
) -> PeakStats:
    """Summarise the strongest in-band spectral peak.

    With picking="harmonic" the peak maximises P(f) + harmonic_weight * P(2f)
    (P(2f) read from the whole spectrum), so a pulse whose second harmonic
    (e.g. from the dicrotic notch) rivals its fundamental is still reported
    at the fundamental. The statistics describe the chosen peak.
    """
    band = spectrum.in_band()
    freqs, power = spectrum.freqs_hz[band], spectrum.power[band]
    if power.size < 3 or not np.isfinite(power).all() or power.sum() <= 0:
        raise SignalRejected("no_spectral_peak", "No usable power in the analysis band")
    if picking == "harmonic":
        second = np.interp(2 * freqs, spectrum.freqs_hz, spectrum.power, right=0.0)
        i = int(np.argmax(power + harmonic_weight * second))
    else:
        i = int(np.argmax(power))
    # Parabolic interpolation refines the peak between zero-padded bins.
    freq = float(freqs[i])
    if 0 < i < power.size - 1:
        a, b, c = power[i - 1], power[i], power[i + 1]
        denom = a - 2 * b + c
        if denom < 0:
            freq += 0.5 * (a - c) / denom * float(freqs[1] - freqs[0])
    near = np.abs(freqs - freqs[i]) <= halfwidth_hz
    # Zero padding at the band edges lets a rival at the edge count as a peak.
    rivals, _ = sps.find_peaks(np.concatenate([[0.0], power, [0.0]]))
    rivals = rivals - 1
    rivals = rivals[~near[rivals]]
    runner_up = float(power[rivals].max()) if rivals.size else 0.0
    prominence = 1.0 - runner_up / float(power[i])
    concentration = float(power[near].sum() / power.sum())
    median = float(np.median(power))
    snr = MAX_SNR_DB if median <= 0 else min(MAX_SNR_DB, 10 * math.log10(power[i] / median))
    return PeakStats(
        frequency_hz=freq,
        prominence=float(np.clip(prominence, 0.0, 1.0)),
        concentration=float(np.clip(concentration, 0.0, 1.0)),
        snr_db=snr,
    )


def window_frequencies(x: np.ndarray, fs_hz: float, config: SignalConfig) -> np.ndarray:
    """Dominant in-band frequency of each sliding window (empty if too short)."""
    win = int(round(config.stability_window_s * fs_hz))
    step = max(1, int(round(config.stability_step_s * fs_hz)))
    if win < 8 or win > len(x):
        return np.zeros(0)
    freqs = []
    for start in range(0, len(x) - win + 1, step):
        seg = x[start : start + win]
        if np.std(seg) < 1e-12:
            continue
        nfft = max(win, 1 << int(math.ceil(math.log2(fs_hz / config.spectrum_resolution_hz))))
        f, p = sps.periodogram(seg, fs=fs_hz, window="hann", nfft=nfft, detrend="linear")
        spec = Spectrum(f, p, config.band_hz)
        try:
            peak = dominant_peak(
                spec, config.peak_halfwidth_hz, config.peak_picking, config.harmonic_weight
            )
            freqs.append(peak.frequency_hz)
        except SignalRejected:
            continue
    return np.asarray(freqs, dtype=np.float64)


def _pulse_from_channels(rel: np.ndarray, method: str, fs: float, config: SignalConfig):
    """Pulse signal from normalised, detrended channels (columns R, G, B or one column)."""
    if method == "green":
        channel = rel[:, 1] if rel.shape[1] == 3 else rel[:, 0]
        return bandpass(channel, fs, config.band_hz, config.filter_order)
    if method == "chrom":
        if rel.shape[1] != 3:
            raise ValueError("CHROM needs three colour channels (R, G, B)")
        r, g, b = (rel + 1.0).T  # back to unit-mean channels
        x = 3 * r - 2 * g
        y = 1.5 * r + g - 1.5 * b
        xy, info = bandpass(np.column_stack([x, y]), fs, config.band_hz, config.filter_order)
        if not info["applied"]:
            xy = sps.detrend(xy, axis=0, type="linear")
        info["chrom_window_s"] = config.chrom_window_s
        info["chrom_edge_handling"] = "Hann overlap-add tapers the first and last half window"
        return chrom_combine(xy[:, 0], xy[:, 1], fs, config.chrom_window_s), info
    if method == "pos":
        if rel.shape[1] != 3:
            raise ValueError("POS needs three colour channels (R, G, B)")
        h, info = bandpass(
            pos_combine(rel + 1.0, fs, config.pos_window_s), fs, config.band_hz, config.filter_order
        )
        info["pos_window_s"] = config.pos_window_s
        info["pos_edge_handling"] = "edge samples are covered by fewer overlapping windows"
        return h, info
    raise ValueError(f"unknown pulse method {method!r}; expected one of {PULSE_METHODS}")


def process_trace(
    timestamps_s: np.ndarray,
    values: np.ndarray,
    method: str = "green",
    config: SignalConfig | None = None,
    span_s: tuple[float, float] | None = None,
    stability: bool = True,
) -> ProcessedSignal:
    """Turn a colour trace into a band-limited pulse signal and its spectrum.

    `values` is (N, 3) RGB with NaN for missing frames, or (N,) for a single
    channel (only the "green" method, which then uses that channel). Invalid
    input is returned as a rejected ProcessedSignal, never raised.
    `stability=False` skips the per-window frequencies (window_freqs_hz empty).
    """
    config = config or SignalConfig()
    if method not in PULSE_METHODS:
        raise ValueError(f"unknown pulse method {method!r}; expected one of {PULSE_METHODS}")
    settings = {"method": method, "config": config.to_dict()}
    quality: dict = {}
    try:
        uniform = resample_uniform(timestamps_s, values, config, span_s)
        quality = uniform.quality
        rel = normalize_detrend(
            uniform.values, config.detrend, uniform.fs_hz, config.detrend_cutoff_hz
        )
        pulse, filter_info = _pulse_from_channels(rel, method, uniform.fs_hz, config)
        settings["filter"] = filter_info
        pulse = _standardize(pulse)
        spectrum = power_spectrum(pulse, uniform.fs_hz, config)
        peak = dominant_peak(
            spectrum, config.peak_halfwidth_hz, config.peak_picking, config.harmonic_weight
        )
    except SignalRejected as exc:
        return ProcessedSignal(
            method=method,
            ok=False,
            reason=exc.reason,
            detail=exc.detail,
            quality={**quality, **exc.quality},
            settings=settings,
        )
    return ProcessedSignal(
        method=method,
        ok=True,
        fs_hz=uniform.fs_hz,
        t=uniform.t,
        signal=pulse,
        spectrum=spectrum,
        peak=peak,
        window_freqs_hz=(
            window_frequencies(pulse, uniform.fs_hz, config) if stability else np.zeros(0)
        ),
        quality=quality,
        settings=settings,
    )


def fuse_spectra(spectra: list[Spectrum]) -> Spectrum:
    """Average of spectra on one frequency grid, each scaled to unit in-band power.

    Combines several ROIs so that no single region dominates by its noise
    level. All spectra must share the frequency grid (same sample rate,
    length and settings).
    """
    if not spectra:
        raise SignalRejected("no_spectral_peak", "No spectrum to fuse")
    base = spectra[0]
    total = np.zeros_like(base.power)
    for spec in spectra:
        if spec.freqs_hz.shape != base.freqs_hz.shape or not np.allclose(
            spec.freqs_hz, base.freqs_hz
        ):
            raise ValueError("spectra to fuse must share a frequency grid")
        in_band = float(spec.power[spec.in_band()].sum())
        if in_band > 0 and math.isfinite(in_band):
            total += spec.power / in_band
    return Spectrum(freqs_hz=base.freqs_hz, power=total / len(spectra), band_hz=base.band_hz)


def snr_at_frequency(spectrum: Spectrum, freq_hz: float, halfwidth_hz: float = 0.1) -> float:
    """SNR (dB) of a known pulse frequency, after de Haan & Jeanne (2013).

    Signal: in-band power within +/- halfwidth of freq_hz and within
    +/- 2 x halfwidth of its second harmonic; noise: all other in-band power.
    """
    band = spectrum.in_band()
    f, p = spectrum.freqs_hz[band], spectrum.power[band]
    signal_bins = (np.abs(f - freq_hz) <= halfwidth_hz) | (
        np.abs(f - 2 * freq_hz) <= 2 * halfwidth_hz
    )
    sig, noise = float(p[signal_bins].sum()), float(p[~signal_bins].sum())
    if sig <= 0:
        return -MAX_SNR_DB
    if noise <= 0:
        return MAX_SNR_DB
    return float(np.clip(10 * math.log10(sig / noise), -MAX_SNR_DB, MAX_SNR_DB))

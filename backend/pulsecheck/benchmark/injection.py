"""Synthetic pulse injection into the skin of real video frames (PRD FR-12).

Model. Every skin pixel of frame k is multiplied, per colour channel c, by

    1 + a * w_c * s(x) * p(t_k)

- p(t) is an intensity waveform with peak-to-peak 1 and zero mean: the
  negative of a blood-volume pulse (more blood absorbs more light, so skin
  gets darker at the systolic peak). The blood-volume pulse is a two-Gaussian
  PPG model per cardiac cycle: a systolic peak and a smaller, later diastolic
  peak, which leaves a dicrotic notch between them and a second harmonic.
- The phase follows an instantaneous heart rate HR(t) = HR0 + 3 sin(2 pi f_v t
  + phi) bpm (slow variability, f_v 0.05-0.15 Hz). The ground-truth HR of a
  clip is the mean of HR(t) over its frames.
- w = (0.43, 1.00, 0.69) for (R, G, B): the blood-volume-pulse signature
  Pbv = (0.33, 0.77, 0.53) of de Haan & van Leest (2014, "Improved motion
  robustness of remote-PPG by using the blood volume pulse signature",
  Physiol. Meas. 35:1913), normalised so green = 1. `a` is therefore the
  peak-to-peak relative change of the green channel.
- s(x) is a spatial weight: 1 on the forehead and cheeks (the "large" ROI
  layout), 0.5 on the rest of the facial skin (nose, upper lip, chin, jaw,
  temples), blurred over ~0.1 interocular distance and zero outside the skin
  mask (face oval minus eyes, eyebrows and lips). Perfusion is highest on
  forehead and cheeks; a spatially uniform pulse would flatter whole-face
  averaging. Real pulses also differ in phase across the face, which this
  model ignores.

Realistic amplitudes. Published camera rPPG work puts the pulsatile change of
the green channel at the order of 0.1-1 % of the skin's DC level in
uncompressed video under good light (Verkruysse, Svaasand & Nelson 2008;
de Haan & Jeanne 2013 and Wang et al. 2017 describe the signal as a fraction
of a percent), lower for dark skin, low light or poor perfusion. The
benchmark sweeps a = 0.05, 0.1, 0.2, 0.5, 1 and 2 % (peak-to-peak) and treats
0.2-1 % as the realistic range for method selection; 0.05-0.1 % is weak and
2 % clearly visible. a = 0 is the null condition.

Quantisation. Decoded frames are 8-bit, and a modulation below half a level
would round away entirely, while a camera digitises the pulse together with
sensor noise that dithers it. Injected frames are therefore rounded
stochastically (floor(x + u), u ~ U[0, 1)), which keeps the expected pixel
value exact and adds at most one level of noise.

The source clip already contains its subject's own pulse. The injected HR is
drawn at least `min_separation_bpm` away from the clip's native HR as
estimated at a = 0 (choose_target_hr), so the two are distinguishable, but
the native pulse remains a structured interferer and its estimate can be
wrong, which this benchmark cannot detect.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from pulsecheck.face.regions import FACE_ROI, RoiConfig, compute_rois, face_frame

# Pbv of de Haan & van Leest (2014), (R, G, B), normalised to green = 1.
BVP_SIGNATURE_RGB = (0.33 / 0.77, 1.0, 0.53 / 0.77)
AMPLITUDES = (0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02)
REALISTIC_AMPLITUDES = (0.002, 0.005, 0.01)
HR_RANGE_BPM = (50.0, 120.0)
HR_VARIABILITY_BPM = 3.0
OTHER_SKIN_WEIGHT = 0.5  # pulse strength off the forehead and cheeks
WEIGHT_BLUR_IOD = 0.1


@dataclass(frozen=True)
class PulseSpec:
    """Ground-truth pulse of one clip."""

    hr_bpm: float  # HR0, the centre heart rate
    variability_bpm: float = HR_VARIABILITY_BPM
    variability_hz: float = 0.1
    variability_phase: float = 0.0
    phase0: float = 0.0  # cardiac phase at t = 0, in cycles

    def instantaneous_hr(self, t: np.ndarray) -> np.ndarray:
        w = 2 * math.pi * self.variability_hz
        return self.hr_bpm + self.variability_bpm * np.sin(w * t + self.variability_phase)

    def cycles(self, t: np.ndarray) -> np.ndarray:
        """Cardiac phase in cycles: the integral of HR(t) / 60."""
        t = np.asarray(t, dtype=np.float64)
        w = 2 * math.pi * self.variability_hz
        vary = (np.cos(self.variability_phase) - np.cos(w * t + self.variability_phase)) / w
        return self.phase0 + (self.hr_bpm * t + self.variability_bpm * vary) / 60.0

    def mean_hr(self, t: np.ndarray) -> float:
        return float(np.mean(self.instantaneous_hr(np.asarray(t, dtype=np.float64))))

    def intensity(self, t: np.ndarray) -> np.ndarray:
        """p(t): zero-mean, peak-to-peak 1, darkest at the systolic peak."""
        return -bvp_waveform(self.cycles(t))

    def to_dict(self) -> dict:
        return asdict(self)


def _bvp_raw(phase: np.ndarray) -> np.ndarray:
    phase = np.mod(phase, 1.0)
    out = np.zeros_like(phase)
    for centre, width, height in ((0.20, 0.07, 1.0), (0.48, 0.10, 0.45)):
        for wrap in (-1.0, 0.0, 1.0):
            out += height * np.exp(-0.5 * ((phase - centre - wrap) / width) ** 2)
    return out


_GRID = _bvp_raw(np.linspace(0, 1, 1000, endpoint=False))
_BVP_MEAN, _BVP_RANGE = float(_GRID.mean()), float(_GRID.max() - _GRID.min())


def bvp_waveform(phase_cycles: np.ndarray) -> np.ndarray:
    """Two-Gaussian blood-volume pulse per cycle, zero mean and peak-to-peak 1."""
    return (_bvp_raw(np.asarray(phase_cycles, dtype=np.float64)) - _BVP_MEAN) / _BVP_RANGE


def random_pulse(rng: np.random.Generator, hr_bpm: float) -> PulseSpec:
    return PulseSpec(
        hr_bpm=float(hr_bpm),
        variability_hz=float(rng.uniform(0.05, 0.15)),
        variability_phase=float(rng.uniform(0, 2 * math.pi)),
        phase0=float(rng.uniform(0, 1)),
    )


def choose_target_hr(
    native_bpm: list[float | None],
    rng: np.random.Generator,
    hr_range: tuple[float, float] = HR_RANGE_BPM,
    min_separation_bpm: float = 15.0,
) -> tuple[float, bool]:
    """A heart rate in `hr_range` at least `min_separation_bpm` from the native estimates.

    Several noisy estimates of one native rate can leave no room in the range.
    Then the least-supported estimate (fewest other estimates within 5 bpm)
    is dropped, repeatedly, until a rate fits. Returns (hr, separated);
    separated is False when even one estimate leaves no room and the rate
    farthest from the estimates was used.
    """
    grid = np.arange(hr_range[0], hr_range[1] + 1e-9, 0.1)
    natives = [float(n) for n in native_bpm if n is not None and math.isfinite(n)]
    if not natives:
        return float(rng.choice(grid)), True
    kept = sorted(natives)
    while kept:
        distance = np.min(np.abs(grid[:, None] - np.asarray(kept)[None, :]), axis=1)
        allowed = grid[distance >= min_separation_bpm]
        if allowed.size:
            return float(rng.choice(allowed)), len(kept) == len(natives)
        support = [sum(abs(n - k) <= 5.0 for n in natives) for k in kept]
        kept.pop(int(np.argmin(support)))
    distance = np.min(np.abs(grid[:, None] - np.asarray(natives)[None, :]), axis=1)
    return float(grid[int(np.argmax(distance))]), False


def skin_weight_map(landmarks: np.ndarray, shape: tuple[int, ...]) -> np.ndarray | None:
    """Per-pixel pulse weight s(x) in [0, 1] (float32, frame-sized), or None without a face."""
    rois = compute_rois(landmarks, shape, RoiConfig(layout="large", face_roi=True))
    skin_roi = rois[FACE_ROI]
    if skin_roi.pixel_count == 0:
        return None
    height, width = shape[:2]
    skin = np.zeros((height, width), np.float32)
    x0, y0, x1, y1 = skin_roi.bbox
    skin[y0:y1, x0:x1] = skin_roi.mask
    weight = skin * OTHER_SKIN_WEIGHT
    for name, roi in rois.items():
        if name == FACE_ROI or roi.pixel_count == 0:
            continue
        x0, y0, x1, y1 = roi.bbox
        weight[y0:y1, x0:x1][roi.mask] = 1.0
    sigma = max(0.5, WEIGHT_BLUR_IOD * face_frame(landmarks).interocular)
    weight = cv2.GaussianBlur(weight, (0, 0), sigma)
    return weight * skin


def inject(
    image_bgr: np.ndarray,
    weight: np.ndarray | None,
    amplitude: float,
    p_t: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """One frame with the pulse value p_t applied, stochastically rounded to uint8."""
    if amplitude == 0 or weight is None:
        return image_bgr
    gain_bgr = np.array(BVP_SIGNATURE_RGB[::-1], dtype=np.float32) * np.float32(amplitude * p_t)
    scaled = image_bgr.astype(np.float32) * (1.0 + weight[..., None] * gain_bgr)
    dither = rng.random(scaled.shape, dtype=np.float32)
    return np.clip(np.floor(scaled + dither), 0, 255).astype(np.uint8)


def fill_missing(landmarks: list[np.ndarray | None]) -> list[np.ndarray | None]:
    """Replace frames without a face by the nearest frame with one (for the injection mask)."""
    have = [i for i, lm in enumerate(landmarks) if lm is not None]
    if not have:
        return list(landmarks)
    out = []
    for i, lm in enumerate(landmarks):
        if lm is not None:
            out.append(lm)
        else:
            nearest = min(have, key=lambda j: abs(j - i))
            out.append(landmarks[nearest])
    return out

# PulseCheck

Experimental rPPG-based deepfake analysis. Results are evidence, not proof.
See [PulseCheck_PRD.md](PulseCheck_PRD.md) for the full requirements.

## Status

- Phase 1: video decoding and metadata inspection (`pulsecheck.video`).
- Phase 2: face landmarks, forehead/cheek ROIs, tracking coverage (`pulsecheck.face`) and
  per-ROI RGB traces (`pulsecheck.signals`).
- Phase 3: pulse signal processing (`pulsecheck.signals.processing`), a per-video feature
  vector (`pulsecheck.features`) and a transparent heuristic baseline (`pulsecheck.models`).
  The heuristic's thresholds are uncalibrated placeholders; its output is a debugging aid,
  not a finding about any video.

## Backend setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; use `source .venv/bin/activate` elsewhere
pip install -e "backend[dev]"
python scripts/download_models.py   # MediaPipe face landmarker -> models/ (not in Git)
```

MediaPipe requires `opencv-contrib-python`, which is the project's only OpenCV dependency.
Do not also install `opencv-python` or `opencv-python-headless`: every OpenCV wheel writes
the same `cv2` package, and mixing them breaks the import. If you have an older environment
with the headless wheel, run `pip uninstall opencv-python-headless opencv-contrib-python`
and then reinstall the project.

Run checks from `backend/`:

```bash
pytest
ruff check .
```

## Inspect a video

```bash
python scripts/inspect_video.py path/to/video.mp4 --extract frames_out --every 30
```

## Visualize face regions and raw traces

```bash
python scripts/visualize_rois.py path/to/video.mp4 --samples 6
```

This writes annotated frames, `traces.png` and `tracking.json` to `data/roi_debug/`. The
annotated frames show the face, so do not commit or share them.

## Pulse signals, features and the heuristic baseline

```bash
python scripts/analyze_signals.py path/to/video.mp4
```

This writes `signals.png` (band-passed green and CHROM pulse signals per ROI), `spectra.png`
and `analysis.json` to `data/signal_debug/`, and prints the feature vector and the heuristic
result. Settings live in `configs/default.yaml` (`signal`, `features`, `heuristic`).

- Each ROI trace is resampled onto a uniform grid shared by all ROIs (missing frames are
  linearly interpolated), normalised by its mean, linearly detrended and band-passed with a
  zero-phase Butterworth filter (default 0.7-3.0 Hz, order 4) when the sample rate and length
  allow it. The filter settings, and whether it ran, are recorded with every signal.
- Signals shorter than 6 s, sampled below 7.5 Hz, with under 70% valid frames or a gap over
  2 s are rejected with a reason code (`too_short`, `invalid_sample_rate`,
  `insufficient_valid_samples`, `gap_too_long`, ...) instead of being analysed.
- On synthetic sinusoids of at least 10 s at 30 fps, the dominant-frequency estimate is within
  0.05 Hz (3 per minute) of the true frequency; the tests check this. Real video is much noisier.
- Features never contain NaN or Infinity: a feature that cannot be computed is `None`, with a
  reason in `missing`.
- The heuristic scores how consistent the pulse is across face regions and returns one of
  "More consistent with real examples", "More consistent with manipulated examples" or
  "Inconclusive". It is inconclusive whenever signal quality, usable regions or face coverage
  are too low. The score is not a probability.

Keep videos and datasets out of Git (see `.gitignore`).

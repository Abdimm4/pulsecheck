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
- Phase 4: FaceForensics++ manifest and group-aware splits (`pulsecheck.dataset`), resumable
  batch feature extraction, logistic-regression and random-forest classifiers and an
  evaluation report (`pulsecheck.models.training`, `.evaluation`, `.report`). See
  [docs/data-card.md](docs/data-card.md). No results are reported yet.

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

## Dataset, training and evaluation (FaceForensics++)

FaceForensics++ may only be used for non-commercial research and education. Keep the
dataset, manifests, feature tables, models and reports under `data/` and `models/` (both
ignored by Git), and never share videos or per-video data.

```bash
# 1. Manifest: one row per video, official splits, leakage checks (exit 2 on leakage).
python scripts/build_manifest.py /path/to/ffpp --splits /path/to/ffpp/splits     --out data/manifests/ffpp_c23.csv
#    pilot run: add --limit-per-class 25 --balance and use another --out name

# 2. Features: resumable (re-run the same command), failures recorded with reason codes.
python scripts/extract_features.py data/manifests/ffpp_c23.csv --workers 4

# 3. Train on train, select on val (test is never read here).
python scripts/train_model.py data/features/ffpp_c23.csv

# 4. Score the test split once, with the heuristic baseline for comparison.
python scripts/evaluate.py data/features/ffpp_c23.csv     --model models/ffpp-c23-logreg-<stamp>.pkl --model models/ffpp-c23-forest-<stamp>.pkl
```

- Manifest fields follow PRD Section 13 (plus `source_video_id`, `group_id`,
  `compression`); a `.csv.json` sidecar records options, counts, warnings and license notes.
- Missing features are imputed with the training median plus a missing-indicator column;
  features missing in over 95% of training videos are dropped. Preprocessing is fitted on
  train only; hyperparameters, the decision threshold and an inconclusive band are chosen on
  validation. A quality gate (the heuristic's thresholds; `--no-quality-gate` to disable)
  makes low-quality videos inconclusive.
- Each model is saved as `models/<name>.pkl` with a `.json` sidecar: config, feature list,
  data version (feature-table and manifest hashes), training dataset and license, and
  train/val metrics. Load only `.pkl` files you created.
- Reports (`data/reports/evaluation-test-<stamp>.md/.json/.predictions.csv`) give accuracy,
  precision, recall, specificity, F1, ROC-AUC, confusion matrices, per-method results,
  group-bootstrap 95% intervals, coverage, inconclusive and failure rates and runtime, and
  state their evaluation conditions. A second test evaluation of a model needs `--force` and
  is flagged.

Keep videos and datasets out of Git (see `.gitignore`).

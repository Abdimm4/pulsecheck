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
- Phase 5: FastAPI service (`pulsecheck.api`), job records and state machine
  (`pulsecheck.jobs`), private upload storage (`pulsecheck.storage`), a Redis/RQ worker
  that runs the real pipeline (`worker/entrypoint.py`), JSON reports (`pulsecheck.reporting`)
  and expiry/cleanup. See [API and worker](#api-and-worker).
- Shortcut checks: media probing, shortcut/combined baselines, a confound audit and video
  normalisation (`pulsecheck.dataset.media`, `.confounds`, `.normalize`). Result: no dataset
  shows pulse signal beyond the shortcut. See [Known issues and findings](#known-issues-and-findings).
- Phase 5A (FR-12): a semi-synthetic pulse-injection benchmark (`pulsecheck.benchmark`,
  `scripts/benchmark_extraction.py`), a UBFC-rPPG ground-truth loader, POS, whole-face and
  large ROIs, robust ROI statistics, landmark smoothing, smoothness-priors detrending and
  harmonic peak picking as config options, and a benchmark-selected pipeline (opt-in). Result:
  at realistic pulse amplitudes the pulse is rarely recovered from these compressed videos,
  with or without the changes. See [Pulse extraction benchmark](#pulse-extraction-benchmark-phase-5a).

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
ruff format --check .
```

The `dev` extra includes the `api` extra (FastAPI, SQLAlchemy, Alembic, Redis, RQ) plus
fakeredis, so the API tests need no database or Redis server. For the research pipeline
alone, `pip install -e backend` is enough.

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
  linearly interpolated), normalised by its mean, detrended (`linear`, or `smoothness_priors`
  after Tarvainen et al. 2002) and band-passed with a zero-phase Butterworth filter
  (0.7-3.0 Hz) when the sample rate and length allow it. The filter settings, and whether it
  ran, are recorded with every signal. The default (`configs/default.yaml` and the code) is
  CHROM per ROI, linear detrending, filter order 4 and a 10 s Welch window. The Phase 5A
  variant (POS on the whole-face skin ROI, smoothness-priors detrending, order 2, 6 s) is
  opt-in: the config comments list the values marked "Phase 5A".
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

### Shortcut checks (source and compression confounds)

If real and fake videos come from different sources, a classifier can tell them apart by
resolution, bitrate, frame rate, duration or brightness without any pulse information.
Every pulse result should be read against such a shortcut baseline.

```bash
# Non-physiological per-video properties (resolution, fps, duration, bitrate, brightness).
python scripts/probe_media.py data/manifests/ffpp_c23.csv --workers 4   # -> data/features/ffpp_c23_media.csv

# Shortcut-only, pulse and pulse + shortcut ("combined") models, gate off so they compare.
for set in shortcut pulse combined; do
  python scripts/train_model.py data/features/ffpp_c23.csv \
      --extra-columns data/features/ffpp_c23_media.csv --feature-set $set --gate off --tag gate-off
done

# One test evaluation of all of them: the report gains a "Shortcut baseline" section.
python scripts/evaluate.py data/features/ffpp_c23.csv --extra-columns data/features/ffpp_c23_media.csv \
    --model models/<each saved .pkl> ... --tag shortcut-check --out-dir data/reports/ffpp_c23

# Which pulse features track the media? (train split; md + json under data/reports/<table>/)
python scripts/audit_confounds.py data/features/ffpp_c23.csv \
    --media data/features/ffpp_c23_media.csv --manifest data/manifests/ffpp_c23.csv
```

- The default `pulse` feature set leaves out tracking/coverage features (`face_coverage`,
  per-ROI `tracking_coverage`, `all_rois_valid_fraction`, `usable_roi_count`), which depend
  on resolution and compression; the quality gate still uses them. `pulse-full` restores the
  old set (model version 0.1.0).
- The "Shortcut baseline" section reports ROC-AUC differences pulse - shortcut and
  combined - shortcut per model family, with paired group-bootstrap 95% intervals. A pulse
  claim needs both differences above 0 with their intervals.
- `scripts/normalize_videos.py` writes normalised copies of a dataset (fixed face-centred
  256x256 crop, constant 30 fps, first 8 s, one brightness gain per clip, identical libx264
  settings via `imageio-ffmpeg`) and a manifest for them (`data/manifests/<name>_norm.csv`),
  so the shortcut test can be rerun with the file properties equalised. Clips that are too
  short or have no face are listed as failed in the manifest sidecar. The copies are derived
  data: keep them under `data/` and do not share them.

Keep videos and datasets out of Git (see `.gitignore`).

## Pulse extraction benchmark (Phase 5A)

Does the pipeline recover a pulse at all? The benchmark injects a known pulse into real
videos and measures how often the heart rate comes back (PRD FR-12).

```bash
# 1. Pick 60 FF++ c23 and 40 normalised NFI real clips (train splits only), split them by
#    source group into dev and held-out halves, inject, re-encode, re-detect and cache the
#    per-frame ROI colour means. Resumable; ~5 h at 4 workers (max 4: ~250-400 MB each).
python scripts/benchmark_extraction.py --workers 4 build
# 2. Diagnose the current pipeline, select a variant on dev, fit the quality ramps on dev,
#    report current vs selected once on held-out (data/reports/benchmark/*.md/.json/.csv).
python scripts/benchmark_extraction.py --workers 4 evaluate
# 3. Optional ground truth: UBFC-rPPG DATASET_2 under data/UBFC-rPPG/ (subject*/vid.avi,
#    ground_truth.txt). Not downloaded automatically: request it from the authors and
#    accept its terms (https://sites.google.com/view/ybenezeth/ubfcrppg). Skipped if absent.
python scripts/benchmark_extraction.py --workers 4 ubfc
```

- Injection (`pulsecheck.benchmark.injection`): every skin pixel (face oval minus eyes,
  brows and lips) is multiplied by `1 + a * w_c * s(x) * p(t)`: blood-volume-pulse signature
  w = (0.43, 1, 0.69) for RGB (de Haan & van Leest 2014), weight 1 on forehead and cheeks and
  0.5 elsewhere, a two-Gaussian PPG waveform with a dicrotic notch, HR 50-120 bpm with
  +/-3 bpm slow variability, stochastic rounding. `a` (peak-to-peak change of green) is 0,
  0.05, 0.1, 0.2, 0.5, 1 and 2 %; 0.2-1 % is treated as realistic. The injected HR is kept
  >= 15 bpm from the clip's own pulse as estimated at a = 0.
- Degradations: libx264 CRF 18/23/28/35 and CRF 23 at half resolution. Per clip the cache
  records skin pixels per ROI, landmark motion, measured bitrate and the native HR.
- Metrics per variant x amplitude x degradation: detection rate (|error| <= 5 bpm), MAE,
  RMSE, SNR at the true frequency, and the false-detection rate on the a = 0 null, with
  group-bootstrap 95% intervals. The report also gives how much of the injected change
  survives each encode.
- Options compared (all in `configs/default.yaml`): `features.pulse_method` green/chrom/pos;
  `color_signal.roi_layout` standard/large, `color_signal.face_roi` and
  `features.hr_source` per_roi/fused/face; `color_signal.statistic` mean/robust;
  `color_signal.landmark_smoothing_frames`; `signal.detrend` linear/smoothness_priors,
  `signal.filter_order`, `signal.welch_window_s`, `signal.peak_picking` argmax/harmonic.
- The cache and reports are derived from non-commercial datasets: keep them under `data/`.

Results (2026-11-01, 99 clips; one FF++ clip was too short):

- **The codec removes much of a weak pulse.** Of the injected change, a median of ~70%
  survives CRF 18, ~62% CRF 23, ~50% CRF 28 and ~32% CRF 35.
- **The current pipeline barely beats chance at realistic amplitudes.** Dev detection is
  4-10% at 0.2-0.5% amplitude against a 2-7% null false-detection rate. It reaches 45-52% only
  at 2% (CRF 18-23). Misses are far off (median 39 bpm), not near-misses. Detection rises
  with skin pixels (3% to 13% across terciles) and bitrate (5% to 16%), but not with motion.
- **Selected on dev:** POS on the whole-face skin ROI, smoothness-priors detrending,
  filter order 2, 6 s Welch window (argmax peak, mean statistic, no landmark smoothing).
- **Held-out, selected vs current:** +3.2 pp [-2.8, +10.6] pooled over 0.2-1% at CRF 23-28,
  so no demonstrated gain at realistic amplitudes. It is clearly better at 1% (CRF 23: 40%
  vs 18%, +21.5 pp [+11.3, +33.3]) and 2% (CRF 23: 73% vs 49%). It is better on FF++ (0.20 vs
  0.10 at realistic amplitudes) but worse on the 256 px NFI clips (0.04 vs 0.11).
- **Quality ramps** (fitted on dev): concentration 0.264-0.309, SNR 7.06-8.22 dB. They
  separate correct from incorrect estimates only weakly (ROC-AUC 0.62 dev, 0.58 held-out;
  the old ramps 0.58 / 0.57). The ramps are narrow, so the score is close to 0 or 1.
- **A pulse is reliably recovered only when strong and lightly compressed.** Held-out, the
  selected pipeline finds 83% at 2% amplitude and CRF 18 and 73% at CRF 23, but 2-12% at
  0.2-0.5% amplitude at any CRF, against a 2-6% null false-detection rate.
- **No ground truth yet.** UBFC-rPPG is not in `data/UBFC-rPPG/`, so the `ubfc` step was
  skipped (loader and parser are tested).

Verdict against the Phase 5A exit criterion: the selected pipeline is **not** a demonstrated
improvement at realistic amplitudes and c23-like compression. It is therefore **not** the
default: the earlier pipeline stays the default (it is also about twice as fast), and the
selected variant is opt-in through the config values marked "Phase 5A" in
`configs/default.yaml`, together with its fitted quality ramps.

### Re-extraction with the selected pipeline (feature schema 2)

```bash
# Use a config with the "Phase 5A" values set (--config path/to/phase5a.yaml, or
# PULSECHECK_CONFIG for the API/worker).
# New tables next to the old ones (the old tables and models stay valid as schema 1).
python scripts/extract_features.py data/manifests/nfi_norm.csv --workers 4 \
    --out data/features/nfi_norm_v2.csv
for set in shortcut pulse combined; do
  python scripts/train_model.py data/features/nfi_norm_v2.csv --cv --manifest data/manifests/nfi_norm.csv \
      --extra-columns data/features/nfi_norm_media.csv --feature-set $set --gate off --tag phase5a-$set-gate-off
done
python scripts/evaluate.py data/features/nfi_norm_v2.csv --model models/<each phase5a .pkl> ... \
    --manifest data/manifests/nfi_norm.csv --extra-columns data/features/nfi_norm_media.csv \
    --tag phase5a --out-dir data/reports/nfi_norm
```

- Schema 2 adds the whole-face ROI (`face.*`) and the primary estimate (`pulse_*`, from
  `features.hr_source`); the pulse classifier set has 43 features.
- A test report lists the earlier evaluations of the same split (`--reports-root`, default
  `data/reports/`), including those of normalised copies of the dataset.

Normalised NFI, test split (scored once, its fourth evaluation; ROC-AUC [95% CI], gate off):

| Family | Shortcut | Pulse | Pulse - shortcut | Combined | Combined - shortcut |
|---|---|---|---|---|---|
| logreg | 0.534 [0.350, 0.772] | 0.745 [0.570, 0.870] | +0.211 [-0.071, 0.434] | 0.605 [0.442, 0.781] | +0.071 [-0.049, 0.168] |
| forest | 0.555 [0.365, 0.743] | 0.709 [0.549, 0.849] | +0.154 [-0.051, 0.364] | 0.634 [0.447, 0.821] | +0.078 [-0.117, 0.252] |

The criterion is not met: every interval includes 0. The higher pulse test scores are not
backed by training cross-validation, where the pulse models scored 0.50 out of fold and the
shortcut models 0.72-0.75. FF++ c23 v2 was not finished (issue 19).

## Known issues and findings

Recorded on 2026-11-01 from the shortcut checks and updated after Phase 5A (the benchmark
and the NFI v2 re-evaluation). Reports are under `data/reports/` (not in Git). Test ROC-AUC
values are for logistic regression and random forest with the quality gate off.

**Results**

1. **NFI-rPPG is confounded by source.** All real clips are at most 500 px tall (median
   856x500, ~975 kbps); fakes have a median of 1280x720 and ~1980 kbps. Height, width and bitrate
   each separate real from fake at AUC ~0.81 on train. A forest on media properties alone
   scored 1.000 test ROC-AUC, against 0.68-0.71 for the pulse models, so those pulse scores
   were mostly the shortcut.
2. **Pulse features do not beat the shortcut on any dataset.** FF++ c23: shortcut
   0.557-0.597, pulse 0.576-0.581, combined 0.596-0.608. Only logreg combined - shortcut is
   above 0 (+0.039 [0.004, 0.072]), so the criterion is not met. Normalised NFI: shortcut
   0.534-0.551, pulse 0.503-0.604, combined 0.540-0.559, with no difference above 0.
   *Phase 5A, normalised NFI v2:* shortcut 0.534-0.555, pulse 0.709-0.745, combined
   0.605-0.634. The pulse - shortcut differences (+0.15, +0.21) still have intervals that
   include 0, and out-of-fold pulse ROC-AUC was 0.50, so the criterion is still not met.
   FF++ c23 was not re-evaluated.
3. **The pipeline barely recovers a pulse, even from real videos.** On real NFI clips the median
   per-ROI SNR is ~9 dB, below the quality ramp's 10 dB floor, and median `signal_quality`
   is ~0. On FF++ the default quality gate decides only ~0.1-0.2% of videos, so all
   comparisons use `--gate off`. This is the main open problem (PRD Phase 5A).
   *Changed in Phase 5A:* the injection benchmark confirms it (issue 17). With the
   recalibrated ramps the default gate now decides 40-44% of normalised NFI test videos, but
   the quality score barely tracks whether the heart rate is right (held-out AUC 0.58).
4. **Normalising NFI did not damage the signal.** On the same clips before and after, the
   real-clip forehead SNR median went from 8.98 to 8.77 dB. Cross-ROI correlation on fakes fell
   from 0.225 to 0.150, which fits it having carried part of the shortcut.

**Confounds still present**

5. **Tracking/coverage features carried the shortcut.** They depend on resolution and
   compression. They now feed the quality gate only, and classifiers use 31 features instead
   of 37 (`pulse-full` keeps the old set, model version 0.1.0). Schema 2 (Phase 5A) has 43.
6. **Clip duration leaks into pulse features on FF++.** The per-ROI `snr_db`,
   `mean_window_freq_std_hz` and `signal_quality` correlate with duration (|rho| 0.34-0.40),
   since longer clips average more. FF++ fakes are slightly shorter (duration AUC 0.44). These
   features are still in the classifier set but flagged by the audit.
7. **Residual brightness and bitrate differences in normalised NFI.** Whole-frame brightness
   still separates real from fake at 0.71 and bitrate at 0.61. In train cross-validation the
   shortcut model still reaches 0.72-0.75. The same holds for the v2 table. The confound
   audit there flags none of the 43 schema-2 features.

**Normalisation caveats**

8. **The first normalisation run clipped skin.** It set the brightness gain on the whole frame,
   which clipped the skin on clips with dark backgrounds, so ROIs were rejected. The gain is now
   measured on the face and capped so bright skin doesn't saturate (the cap applied to 299 of 500
   clips). The cap is why issue 7 remains.
9. **13 of 500 NFI clips fail normalisation:** 6 are shorter than 8 s and 7 have no detectable
   face (their original face coverage was already ~0). *Resolved in Phase 5A:* extraction now
   reads the manifest sidecar and records them as `normalization_too_short` /
   `normalization_no_face` (as in `nfi_norm_v2.csv`). `nfi_norm.csv` still says
   `file_not_found`.
10. **Frame rate resampling repeats the nearest frame.** 24 fps sources therefore repeat about
    one frame in five at 30 fps.
11. **The fixed 8 s length** keeps every fake and loses 6 real clips (10 s would lose 53). At
    8 s the Welch spectrum is a single window and the 6 s stability window yields only 3
    windows. *Changed in Phase 5A:* the default Welch window is now 6 s (chosen on the benchmark's
    dev half); with 50% overlap that is still one segment on 8 s clips.

**Evaluation caveats**

12. **The NFI test split is small:** 82 videos in 14 groups, so the 95% intervals are wide
    (e.g. shortcut logreg [0.358, 0.766]).
13. **The test splits have been reused.** NFI test has been scored four times (default gate,
    gate off + shortcut, normalised, and the Phase 5A v2 evaluation) and FF++ c23 twice.
    Reports now list the earlier evaluations of their split. Decisions use validation/CV
    only. The FF++ c23 test split stays reserved for the deferred v2 evaluation.
14. **The heuristic baseline is degenerate on NFI.** With the gate off it labels every decided
    video as manipulated (specificity 0.000). On NFI v2 with the default gate it is
    still nearly so (specificity 0.040 at 0.44 coverage).

**Environment and data**

15. **No system ffmpeg.** `normalize_videos.py` uses `imageio-ffmpeg` (a dev extra, bundled
    ffmpeg 7.1, libx264 CRF 23). Without it, it falls back to OpenCV's mp4v writer, which uses
    different encoder settings.
16. **The NFI license is unstated.** The real clips derive from Celeb-DF (non-commercial research
    terms). Treat all NFI-derived data as non-commercial and do not share it.

**Phase 5A**

17. **The selected pipeline is not a demonstrated improvement.** Held-out, pooled over
    0.2-1% amplitude at CRF 23-28, selected - current detection is +3.2 pp [-2.8, +10.6].
    A pulse is reliably recovered only at 2% amplitude with light compression. The selected
    pipeline is worse than the current one on the 256 px NFI clips (0.04 vs 0.11 at
    realistic amplitudes).
18. **Extraction is slower with the whole-face skin mask.** On FF++ c23 the median
    per-video time rose from 18 s (schema 1 run) to 39 s (v2 run, 4 workers; ~11.6 s of wall
    time per video). On 256 px NFI clips it rose from 10 s to 12 s.
19. **FF++ c23 v2 is partial and not evaluated.** The re-extraction was stopped at 334 of
    6000 videos (estimated ~18 h remaining). `data/features/ffpp_c23_v2.csv` is kept and
    resumes with the same command. No v2 model was trained on FF++, and its test split was
    not used.
20. **No ground-truth dataset.** UBFC-rPPG must be requested from its authors (terms at
    https://sites.google.com/view/ybenezeth/ubfcrppg) and placed under `data/UBFC-rPPG/`.
    Until then the benchmark is semi-synthetic only.

## API and worker

The web service: a FastAPI app, a job database (PostgreSQL in deployment, SQLite for
local runs), a Redis queue and a background worker that runs the pipeline above.

```bash
pip install -e "backend[dev]"            # or backend[api]; add [postgres] for PostgreSQL
cd backend && alembic upgrade head && cd ..   # create/upgrade the job database

# Redis must be running (e.g. Memurai or WSL on Windows, or docker compose up redis).
uvicorn --factory pulsecheck.api.app:create_app --port 8000   # API docs: /docs
python worker/entrypoint.py worker                            # analysis worker
python worker/entrypoint.py cleanup --loop 300                # expiry/deletion sweep
```

`docker-compose.yml` runs the same stack (postgres, redis, api, worker, cleanup):
`docker compose up --build`, then open http://localhost:8000/docs. Needs Docker Desktop
and a filled-in `.env`.

Environment variables (see [.env.example](.env.example); placeholders only, keep real
values in an untracked `.env` or your host's secret store):

| Variable | Purpose | Default |
|---|---|---|
| `DATABASE_URL` | SQLAlchemy URL, e.g. `postgresql+psycopg://user:pw@host/db` | SQLite at `data/pulsecheck.db` |
| `REDIS_URL` | Queue and rate-limit store | `redis://localhost:6379/0` |
| `PULSECHECK_SECRET_KEY` | Salts the client-IP hash used for limits | random per process (dev only) |
| `PULSECHECK_CORS_ORIGINS` | Comma-separated frontend origins | none |
| `PULSECHECK_TRUST_PROXY` | `1`: client IP from `X-Forwarded-For` | off |
| `PULSECHECK_STORAGE_DIR` | Private upload directory | `data/storage` |
| `PULSECHECK_MODEL_PATH` | Trained model `.pkl` instead of the heuristic | heuristic |
| `PULSECHECK_CONFIG` | Config file | `configs/default.yaml` |

Endpoints (OpenAPI at `/docs` and `/openapi.json`):

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Database and queue reachability (503 if degraded) |
| GET | `/version` | App, API, model, report-schema and feature-schema versions |
| POST | `/v1/jobs` | Create a job: job ID, one-time access token, upload instructions, limits |
| PUT | `/v1/jobs/{job_id}/upload` | Raw video bytes (`application/octet-stream`), capped at the size limit |
| POST | `/v1/jobs/{job_id}/complete-upload` | Validate by decoding and queue the analysis (202) |
| GET | `/v1/jobs/{job_id}` | Status; the result summary once completed |
| GET | `/v1/jobs/{job_id}/report` | The JSON report (FR-10) of a completed job |
| DELETE | `/v1/jobs/{job_id}` | Delete the upload, result and report now (204) |

```bash
curl -X POST localhost:8000/v1/jobs                    # -> job_id, access_token, upload.url
curl -X PUT localhost:8000/v1/jobs/$ID/upload -H "Authorization: Bearer $TOKEN"      -H "Content-Type: application/octet-stream" --data-binary @clip.mp4
curl -X POST localhost:8000/v1/jobs/$ID/complete-upload -H "Authorization: Bearer $TOKEN"
curl localhost:8000/v1/jobs/$ID -H "Authorization: Bearer $TOKEN"          # poll
curl localhost:8000/v1/jobs/$ID/report -H "Authorization: Bearer $TOKEN"
```

- **Jobs** move `created -> uploading -> queued -> processing -> completed | failed`, and
  any status can become `expired`. Transitions are checked in one table
  (`pulsecheck.jobs.models.ALLOWED_TRANSITIONS`) and applied as compare-and-set updates.
  Job rows hold timestamps, expiry, result metadata and deletion status, never video bytes.
- **Access**: every `/v1/jobs/{id}` call needs `Authorization: Bearer <access_token>`. The
  token (32 random bytes) is returned once and stored only as a SHA-256 hash, compared in
  constant time and never logged; tokens in query strings are ignored. A missing, wrong or
  other job's token, a malformed ID and an unknown job all get the same 404
  `job_not_found`, so a job ID alone reveals nothing.
- **Uploads** (UP-1, `upload:` in the config): MP4 or MOV only, recognised from the file's
  first bytes; 100 MB, 60 s, 1920x1080 (either orientation) and 3600 frames by default.
  `complete-upload` decodes the header and first frame; the worker decodes the whole video
  (stopping one frame past the limit) and checks the measured values again. Rejections fail
  the job with a code (`unsupported_format`, `corrupt_video`, `video_too_long`, ...) and
  delete the file. Files live under server-generated random keys in a private directory the
  API never serves.
- **Capacity** (UP-2, `capacity:`): job creation answers 503 `service_busy` (with
  `Retry-After`) when the queue or the global active-job cap is full, and 429 when a
  visitor exceeds the job-creation rate limit or their own active-job cap. Visitors are
  identified by a keyed hash of their IP, deleted when the job expires.
- **Worker**: RQ's `SimpleWorker`, which runs jobs without forking because RQ's default
  worker needs `os.fork()` and does not run on Windows. On Windows, RQ enforces the job
  timeout (300 s) with a timer thread. One worker runs one analysis at a time; start more
  processes for more. It writes the result and the JSON report, then deletes the upload.
- **Retention** (UP-4, `retention:`): uploads are deleted as soon as analysis finishes or
  fails; jobs not queued within 1 h expire and their upload is deleted; results, reports and
  job metadata expire 24 h after creation. Expiry is enforced when a job is read as well as
  by the cleanup sweep, which also fails stuck analyses, retries failed deletions and
  deletes orphaned files. Expired jobs keep a data-free record (status `expired`, reason and
  deletion status) for 7 days, then disappear. Server logs and database backups are outside
  this policy and need their own retention once deployed.
- **Reports** contain versions, timestamps, measured video metadata, configuration, quality
  checks, features, the result and its score semantics, warnings, caveats and expiry. They
  never contain video bytes, frames, file paths, tokens or client data.


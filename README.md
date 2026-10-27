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


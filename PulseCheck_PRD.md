# PulseCheck — Public rPPG Deepfake Detection Web Application

## Product Requirements Document (PRD)

| | |
|---|---|
| **Version** | 2.2 (adds legal/compliance, website design, accessibility, SEO, analytics, paid-service, and regional-availability requirements) |
| **Date** | October 19, 2026 |
| **Product owner** | Solo developer |
| **Product type** | Public web application / computer-vision research tool |
| **Release model** | Paid video analysis with guest checkout (no accounts), rate-limited, automatic deletion, not offered in restricted jurisdictions |

---

## 1. Executive Summary

PulseCheck is a public website where anyone can upload a short video of a face and receive an experimental deepfake analysis based on remote photoplethysmography (rPPG). The detection engine extracts subtle color variations from facial regions, processes their time-series signals, calculates signal-quality and consistency features, and applies a documented baseline or machine-learning classifier.

The product will use a React + TypeScript frontend, a Python FastAPI backend, and a separate background worker for computationally expensive video analysis. Uploaded videos will be held in private temporary storage and automatically deleted according to a published retention policy. Users will not need to create an account for the MVP. PulseCheck is a **paid service** (no free tier; see Section 29) and will **not be offered in jurisdictions where biometric-privacy laws are determined to apply** (see LG-13).

The result is evidence, not proof. A real video can have weak pulse signals, and a manipulated video can contain plausible physiological patterns. The application must include an "inconclusive" outcome and communicate limitations clearly.

---

## 2. Estimated Development Timeline

### Expected solo-developer effort

For a student developer building the project while learning unfamiliar technologies, a realistic estimate is **8–12 weeks at about 8–12 hours per week** for a public MVP, or roughly **65–140 hours**. If you already know React, Python APIs, and deployment, a constrained MVP may be possible in **4–6 weeks**. A robust public service with more rigorous testing and operational safeguards may take **12–16+ weeks**.

These are planning estimates, not guarantees. Dataset access, model quality, deployment troubleshooting, and the amount of frontend polish can change the schedule.

| Phase | Estimated effort | Approximate schedule |
|---|---|---|
| Research, repo, environment, video input | 6–10 hours | Week 1 |
| Face regions and signal extraction | 10–16 hours | Weeks 1–2 |
| Signal processing, features, tests | 10–18 hours | Weeks 2–3 |
| Baseline classifier and evaluation | 10–18 hours | Weeks 3–4 |
| FastAPI service and job lifecycle | 8–14 hours | Weeks 4–5 |
| React interface and result charts | 12–22 hours | Weeks 5–7 |
| Private storage, queue, rate limits, deletion | 8–16 hours | Weeks 7–8 |
| Deployment, end-to-end tests, docs, fixes | 10–20 hours | Weeks 8–10+ |

Some phases can overlap. The first public MVP should prioritize a working, safe vertical slice over a large feature list.

---

## 3. Product Vision and Problem Statement

Convincing synthetic portrait videos can be difficult to assess visually. rPPG research investigates whether small, time-varying color changes associated with blood flow can provide a physiological signal from ordinary video. In principle, signal consistency across multiple facial regions may provide evidence useful for deepfake detection.

In practice, these signals are affected by motion, lighting, camera quality, compression, frame rate, occlusion, makeup, and face tracking. Some generation methods may preserve or synthesize plausible signals. PulseCheck will therefore test a measurable hypothesis and expose evidence and uncertainty rather than make an absolute authenticity claim.

### Product question

> Can pulse-related temporal features and cross-region consistency provide useful evidence for distinguishing authentic videos from manipulated videos in documented test conditions?

### Research references

- Intel Newsroom, "Intel Introduces Real-Time Deepfake Detector" (Nov. 14, 2022): https://www.intel.com/content/www/us/en/newsroom/news/intel-introduces-real-time-deepfake-detector.html
- Çiftçi, Demir, and Yin, "FakeCatcher: Detection of Synthetic Portrait Videos using Biological Signals," IEEE TPAMI (2020): https://doi.org/10.1109/TPAMI.2020.3009287
- Netherlands Forensic Institute rPPG Deepfake Detection research: https://github.com/NetherlandsForensicInstitute/rPPG-Deepfake-Detection
- NFI dataset page: https://huggingface.co/datasets/NetherlandsForensicInstitute/rppg-deepfake-detection

Review the relevant papers and dataset terms before implementation. Historical performance reported by another system is not a performance guarantee for PulseCheck.

---

## 4. Goals

1. Offer a public, mobile-responsive web application with anonymous uploads.
2. Implement the core rPPG signal-extraction and feature pipeline in code the developer understands.
3. Return an explainable result with signal-quality checks and an inconclusive state.
4. Keep uploaded videos private and automatically delete them after analysis or expiry.
5. Prevent uncontrolled resource use through file limits, job limits, and rate limiting.
6. Evaluate models on defensible, held-out data splits.
7. Provide reproducible tests, documentation, and deployment configuration.
8. Demonstrate computer vision, signal processing, ML evaluation, full-stack development, API design, security, and deployment.
9. Operate as a paid service with transparent pricing, secure third-party payment handling, and a clear refund policy.
10. Restrict availability in jurisdictions where applicable biometric-privacy laws create unacceptable legal risk.

---

## 5. Non-Goals for the MVP

- Proving a video's authenticity beyond doubt.
- Identifying or recognizing the person in a video.
- Medical-grade heart-rate measurement or diagnosis.
- Detecting fake audio, images, or all forms of synthetic media.
- Unlimited uploads or guaranteed real-time processing.
- Building user accounts, saved analysis history, or social features. (Payments are **in scope**; see Section 29.)
- Training a large video neural network from scratch before the signal pipeline is validated.
- Making high-stakes legal, employment, identity, or safety decisions based on the result.

---

## 6. Target Users and User Journey

### Target users

- Members of the public who want to explore whether a face video shows signs associated with manipulation.
- Students and researchers interested in rPPG-based analysis.
- Recruiters or engineers reviewing the project as a software portfolio.

### Anonymous user journey

1. Visit the public website without signing in.
2. Read a short explanation of the experimental method and privacy policy.
3. Select a supported video file.
4. Review the file's size and analysis limits.
5. Agree to the upload notice and click **Analyze video**.
6. See a job status such as "Uploading," "Queued," or "Analyzing."
7. View the result, signal-quality indicators, and plots when analysis finishes.
8. Optionally download a JSON report.
9. Leave the page; temporary video data and job artifacts expire automatically.

The MVP does not retain a personal history of past analyses. If a user refreshes or closes the page, the result may be lost unless the temporary job token is still valid. This must be explained in the interface.

> **Paid-service update:** Because PulseCheck is paid, the journey includes a region check, price disclosure, and guest checkout before upload analysis begins. See Section 29 and LG-13.

---

## 7. Product Scope and Release Plan

### Release 0 — Local research prototype

- Video metadata inspection and decoding.
- Face detection and facial-region visualization.
- RGB time-series extraction.
- Signal processing and synthetic-signal tests.
- Initial features and transparent heuristic baseline.

### Release 1 — Public MVP

- React + TypeScript upload/result interface.
- FastAPI backend.
- Anonymous, rate-limited job submission.
- Paid guest checkout through a third-party payment processor, with refund handling (Section 29).
- Regional availability restrictions for blocked jurisdictions (LG-13).
- Background analysis worker.
- Private temporary video storage.
- Automatic deletion and job expiry.
- Signal-quality checks and inconclusive outcome.
- Feature-based baseline or validated classical ML model.
- Results charts and JSON report.
- Basic monitoring, CI, and deployment documentation.

### Release 2 — Reliability and research

- Stronger group-aware dataset evaluation.
- Multiple dataset/source tests and robustness experiments.
- Improved calibration and score interpretation.
- Queue concurrency controls, better abuse prevention, and operational alerts.
- Optional user accounts only if there is a demonstrated need.
- Optional GPU inference if benchmarks justify it.

---

## 8. Recommended Technology Stack

| Layer | Technology | Responsibility |
|---|---|---|
| Frontend | React + TypeScript + Vite | Public website and application state |
| Styling | Tailwind CSS | Responsive layout and consistent UI |
| Charts | Plotly.js | Signal traces, spectra, evaluation visualizations |
| API | Python + FastAPI + Pydantic | Validation, job submission, result retrieval |
| Video processing | OpenCV | Decode frames, inspect metadata, image operations |
| Face landmarks | MediaPipe Face Landmarker or a documented alternative | Define facial ROIs |
| Numerical computing | NumPy | Arrays and time-series operations |
| Signal processing | SciPy | Filtering, spectra, peak analysis |
| Classical ML | scikit-learn | Feature-based baseline and model evaluation |
| Optional later ML | PyTorch | Learned temporal/spatial models after baseline validation |
| Job queue | Redis + RQ or Celery | Background video analysis |
| Metadata database | PostgreSQL | Job state, timestamps, expiry, result metadata |
| Temporary video storage | Private Cloudflare R2 or Amazon S3 bucket | Uploads and short-lived artifacts |
| Upload transport | Presigned upload URL where supported | Upload to private storage without exposing storage credentials |
| Payments | `[PAYMENT PROCESSOR — TO BE CHOSEN]` hosted checkout | Collect payment without the app handling card data |
| Region detection | CDN/edge geolocation headers or a documented geo-IP service | Enforce regional availability restrictions (LG-13) |
| Testing | pytest + Vitest + React Testing Library | Python and frontend tests |
| Code quality | Ruff + TypeScript checks | Formatting, linting, static checks |
| CI | GitHub Actions | Run tests on commits and pull requests |
| Packaging | Docker | Reproducible API and worker environments |
| Hosting | Frontend static hosting plus container-capable API/worker host | Public deployment |

Use one package manager and lock dependencies. Do not add infrastructure only for resume keywords; each service should solve a concrete problem.

---

## 9. High-Level Architecture

```
Browser (React + TypeScript)
        |
        | HTTPS
        v
FastAPI service
  |          |
  |          +--> Validate file metadata / create job
  |
  +--> PostgreSQL (job status, expiry, result metadata)
  |
  +--> Private object storage (temporary video)
  |
  +--> Redis queue
              |
              v
       Background worker
       OpenCV / MediaPipe
       NumPy / SciPy
       feature extraction / ML
              |
              v
       Store result metadata
       and permitted temporary plots
              |
              v
Browser polls job status and
retrieves authorized results
```

The website and worker must not share public file URLs. Use opaque job identifiers and a high-entropy temporary access token or equivalent authorization mechanism. Job IDs alone should not grant access to private results.

---

## 10. Anonymous Upload, Privacy, and Security Requirements

Anonymous access is a product feature, not an absence of security controls.

### UP-1: Upload restrictions

- Accept only explicitly supported formats, initially MP4 and MOV if decoder support is reliable.
- Initial maximum size: 100 MB.
- Initial maximum duration: 60 seconds.
- Set maximum resolution, decoded frame count, and processing time.
- Validate the media by decoding it; never trust its extension or client-supplied MIME type alone.
- Reject corrupt files, empty files, unsupported codecs, and malformed media gracefully.
- Make limits configurable and show them before upload.

These are proposed initial product limits. Adjust them after load and memory testing.

### UP-2: Rate limiting and resource protection

- Rate-limit job creation per IP or other abuse-control signal.
- Apply global concurrency limits and a bounded queue.
- Limit each anonymous visitor to a configurable number of active jobs.
- Add request timeouts, worker timeouts, and memory/CPU limits.
- Return a clear "service busy; try again later" message when capacity is exhausted.
- Add CAPTCHA or other friction only if abuse is observed or threat modeling justifies it.
- The service is paid and has no free usage allowance (Section 29). Do not promise fixed usage quotas until hosting cost and abuse patterns are understood.

IP addresses can be personal data in some jurisdictions. Retain only what is needed for security, restrict access, and document log retention. Do not build invasive fingerprinting into the MVP.

### UP-3: Private storage

- Store uploads in a private bucket with public access disabled.
- Use short-lived presigned URLs if browser-direct upload is implemented.
- Do not expose bucket credentials or long-lived storage keys to the browser.
- Keep upload objects isolated by random server-generated identifiers.
- Do not allow the user to choose storage paths.
- Restrict API and worker access to only the required objects.

### UP-4: Automatic deletion

Define and implement an explicit policy:

- Delete the uploaded video immediately after analysis finishes if it is no longer needed.
- Delete failed and abandoned uploads after a short expiry period (proposed default: 1 hour).
- Expire job metadata and temporary report artifacts (proposed default: 24 hours).
- Use a scheduled cleanup process and storage lifecycle rules where available.
- Record deletion status without retaining the video.
- Test cleanup behavior for success, failure, timeout, and abandoned jobs.

The UI must accurately describe the implemented policy. Do not claim that all data is instantly erased if logs, backups, or storage lifecycle delays may retain limited metadata or copies for a period.

### UP-5: Result access

- Use unguessable temporary tokens or equivalent authorization for job results.
- Set an expiry time for results.
- Do not include the original video in a JSON report.
- Do not publish user uploads, thumbnails, or charts by default.
- Avoid placing sensitive tokens in analytics, referrer headers, or application logs.

### UP-6: Privacy notice and consent

Before upload, state:

- why the video is processed;
- that processing is automated and experimental;
- how long the video and results are retained;
- whether any third-party hosting or processing providers receive the file;
- that the user should only upload media they have permission to use;
- how to report a security or privacy concern.

Do not use uploads for model training without separate, explicit opt-in consent. The MVP must not train on public uploads by default.

### UP-7: Abuse and threat handling

- Keep dependencies updated and scan them where practical.
- Validate request schemas and restrict file handling.
- Protect against path traversal, unsafe filenames, decompression/resource exhaustion, and oversized decoded media.
- Do not execute uploaded content.
- Use HTTPS, secrets management, least-privilege credentials, and separate dev/prod settings.
- Set CORS to the deployed frontend origin rather than wildcard access with credentials.
- Avoid storing raw frames or face embeddings unless required; the MVP should not create identity embeddings.
- Provide a way to report misuse.

---

## 11. Functional Requirements

### FR-1: Upload page

Display:

- product name and purpose;
- supported formats, duration, and size limits;
- drag-and-drop and file picker;
- upload progress;
- privacy notice and deletion policy;
- a clear Analyze button;
- a reminder that results are not proof of authenticity.

### FR-2: Video decoding and metadata

Extract:

- frame dimensions;
- estimated frame rate and timestamps;
- duration;
- number of decoded frames;
- decode errors.

Handle variable frame rates cautiously. Sampling must preserve timestamps; do not assume every video is exactly 30 FPS.

### FR-3: Face detection and region selection

Detect facial landmarks and define regions such as:

- forehead;
- left cheek;
- right cheek.

Exclude eyes, eyebrows, lips, hair, and background where feasible. Track region stability and the percentage of frames with valid regions. If face tracking is unreliable, return a warning or inconclusive result.

### FR-4: Color signal extraction

For each valid frame and ROI:

1. Obtain a robust RGB channel statistic (mean or median).
2. Store the value and timestamp.
3. Record ROI validity and tracking quality.
4. Plot the raw signal for debugging and user explanation.

Start with green-channel extraction as a simple baseline, then compare it with a chrominance-based rPPG method if time allows.

### FR-5: Signal processing

- Handle missing frames and nonuniform timestamps.
- Normalize and detrend signals where appropriate.
- Apply a configurable band-pass filter only when sampling rate and duration support it.
- Record filter settings and edge handling.
- Detect invalid or too-short signals before calculating features.

A starting experimental band might be 0.7–3.0 Hz, but thresholds must be configurable and evaluated. Do not infer a reliable pulse from a clip that is too short or has insufficient temporal resolution.

### FR-6: Feature extraction

Possible features:

- dominant frequency in the selected band;
- peak prominence;
- spectral concentration;
- signal-to-noise proxy;
- stability across windows;
- cross-region correlation/coherence;
- differences in regional dominant frequencies;
- face/ROI tracking coverage;
- signal-quality score.

Features must be robust to invalid inputs and document missing values.

### FR-7: Classification

Build:

1. A transparent heuristic baseline for debugging and comparison.
2. A scikit-learn model trained on labeled examples with group-aware splits.

Start with logistic regression and a tree-based model. Keep training, validation, and test sets separate. Do not fit preprocessing on the test set.

Use three result states:

- **More consistent with real examples**
- **More consistent with manipulated examples**
- **Inconclusive**

These labels describe the model's evidence relative to its training/evaluation scope, not certainty about authenticity.

### FR-8: Results page

Show:

- result category and plain-language explanation;
- model score only with clear wording;
- signal quality and tracking coverage;
- duration and number of usable frames;
- ROI-level feature summary;
- time-series plots;
- frequency spectrum;
- warnings and limitations;
- temporary result expiry;
- JSON report download.

Only call a score a probability if calibration has been evaluated. Do not present a percentage as a probability of deception without evidence that supports that interpretation.

### FR-9: Job status

Use a job state machine:

`created` → `uploading` → `queued` → `processing` → `completed` | `failed` | `expired`

Validate allowed transitions. Provide an endpoint for the browser to check job status. Do not hold an HTTP request open while a long video is processed.

### FR-10: JSON report

Include:

- report schema version;
- app and model version;
- timestamp;
- video metadata needed for reproducibility;
- configuration and quality checks;
- feature values;
- result category and score semantics;
- warnings and caveats;
- expiry information.

Exclude original video bytes, unnecessary personal information, and secret access tokens.

### FR-11: Batch evaluation

Provide a CLI to process a labeled manifest and export per-video predictions plus aggregate metrics. Record failures without silently dropping them.

---

## 12. API Design

Suggested endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Service health |
| GET | `/version` | App and model version |
| POST | `/v1/jobs` | Create a job and return job ID plus upload instructions |
| POST | `/v1/jobs/{job_id}/complete-upload` | Confirm upload and enqueue analysis if presigned upload is used |
| GET | `/v1/jobs/{job_id}` | Return authorized status and, when completed, the result |
| GET | `/v1/jobs/{job_id}/report` | Return authorized JSON report |
| DELETE | `/v1/jobs/{job_id}` | Request deletion of the job's temporary resources |

Protect job endpoints with a temporary, high-entropy access token or another secure capability mechanism. Avoid treating a public job ID as a secret.

Use request-size limits, response schemas, safe errors, and OpenAPI documentation. Do not expose a public arbitrary-file-processing endpoint without capacity controls.

---

## 13. Data and Evaluation Strategy

### Candidate datasets

Investigate access and licensing for:

- FaceForensics++: https://github.com/ondyari/FaceForensics
- Celeb-DF: https://github.com/yuezunli/celeb-deepfakeforensics
- NFI rPPG Deepfake Detection: https://github.com/NetherlandsForensicInstitute/rPPG-Deepfake-Detection
- NFI dataset: https://huggingface.co/datasets/NetherlandsForensicInstitute/rppg-deepfake-detection

Verify current terms and access approval. Keep large datasets outside Git and never redistribute material unless its license explicitly permits it.

### Manifest fields

At minimum:

- `video_id`
- `path` or managed URI
- `label`
- `source_dataset`
- `subject_id` when available
- `manipulation_type`
- `original_video_id` when available
- `split`
- license/usage notes

### Leakage prevention

- Keep source videos and manipulated derivatives in the same split.
- Split by subject/source group where possible.
- Do not split adjacent frames from the same video across training and testing.
- Fit preprocessing and feature selection only on training data.
- Use validation data for thresholds and model selection.
- Evaluate once on the held-out test set.
- Add held-out source or generator evaluation where possible.

### Metrics

Report:

- accuracy;
- precision;
- recall/sensitivity;
- specificity;
- F1;
- ROC-AUC;
- confusion matrix;
- per-source/per-manipulation performance;
- confidence intervals where feasible;
- inconclusive rate and coverage;
- runtime and failure rate.

A target accuracy is not a release requirement until a defensible evaluation protocol is established. Honest, reproducible metrics matter more than matching a historical headline result.

---

## 14. Frontend Design and Accessibility

### Main pages

- **Home / upload:** explanation, uploader, limits, privacy.
- **Analysis:** upload progress, queue status, processing progress.
- **Results:** result state, signal plots, features, caveats, report download.
- **Methodology:** how rPPG works, model scope, evaluation, limitations.
- **Privacy:** storage, retention, deletion, third-party services, contact method.

### UX requirements

- Responsive desktop and mobile layout.
- Accessible labels and keyboard operation.
- Progress feedback for upload and analysis.
- Clear error messages and retry option.
- Use text and icons in addition to colors.
- Charts with labels, units, legends, and short explanations.
- A visible "not forensic proof" notice.
- Avoid frightening or accusatory language about people in videos.

---

## 15. Testing Plan

### Signal unit tests

- Synthetic periodic signals at known frequencies.
- Correct frequency estimate within a documented tolerance.
- Invalid sample rate and too-short signals handled safely.
- Filter stability and edge cases.
- Missing timestamps and nonuniform sampling.
- No NaN/Infinity values in normal feature output.

### Backend tests

- Invalid, corrupt, oversized, and unsupported uploads.
- Job-state transitions.
- Queue failure and timeout handling.
- Authorization checks for result/report endpoints.
- Rate limits and concurrency limits.
- Expired token and expired job behavior.
- Cleanup on success, failure, and abandoned upload.
- JSON schema validation.

### Frontend tests

- Upload form validation.
- Progress and queued states.
- Failed and expired jobs.
- Inconclusive result rendering.
- Accessible keyboard controls and chart labels.

### End-to-end tests

- Upload a permitted test video.
- Observe job status transitions.
- Receive result and report.
- Verify video deletion and result expiry.
- Confirm that one anonymous user cannot access another user's result.

### CI

GitHub Actions should run:

- Ruff;
- pytest;
- TypeScript checks;
- Vitest/React Testing Library;
- lightweight integration tests.

Do not require a large research dataset to run standard pull-request checks.

---

## 16. Repository Structure

```
pulsecheck/
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── vite.config.ts
├── backend/
│   ├── pyproject.toml
│   ├── pulsecheck/
│   │   ├── api/
│   │   ├── jobs/
│   │   ├── storage/
│   │   ├── video/
│   │   ├── face/
│   │   ├── signals/
│   │   ├── features/
│   │   ├── models/
│   │   └── reporting/
│   └── tests/
├── worker/
│   ├── Dockerfile
│   └── entrypoint.py
├── scripts/
│   ├── build_manifest.py
│   ├── extract_features.py
│   ├── train_model.py
│   └── evaluate.py
├── configs/
│   └── default.yaml
├── docs/
│   ├── architecture.md
│   ├── methodology.md
│   ├── model-card.md
│   ├── data-card.md
│   ├── privacy.md
│   └── limitations.md
├── tests/
│   └── e2e/
├── docker-compose.yml
├── .github/workflows/ci.yml
├── .gitignore
└── README.md
```

Keep videos, secrets, environment files, and large model/data files out of Git. Include `.env.example` with placeholders, never real credentials.

---

## 17. Development Roadmap and Milestones

### Phase 1 — Research and setup (Week 1)

- Read the relevant rPPG papers.
- Choose a legally usable small sample set.
- Create the repository, Python environment, and CI skeleton.
- Define input/output schemas and initial product limits.

**Exit criterion:** a script reads video metadata and extracts frames.

### Phase 2 — Face regions and signal extraction (Weeks 1–2)

- Detect landmarks and define forehead/cheek regions.
- Visualize the ROIs.
- Extract RGB time series and timestamps.
- Measure tracking coverage.

**Exit criterion:** sample videos produce region traces and tracking diagnostics.

### Phase 3 — Signal processing and features (Weeks 2–3)

- Implement filtering, spectral analysis, and quality checks.
- Write synthetic-signal unit tests.
- Build the first feature vector and heuristic baseline.

**Exit criterion:** documented signal plots, working tests, and an inconclusive state.

### Phase 4 — Dataset and classifier (Weeks 3–4)

- Build dataset manifest.
- Create group-aware splits.
- Train logistic regression and a tree-based baseline.
- Evaluate on held-out data.

**Exit criterion:** reproducible metrics and documented limitations.

### Phase 5 — API and job model (Weeks 4–5)

- Implement FastAPI endpoints.
- Add PostgreSQL job records.
- Add Redis-backed worker queue.
- Add temporary tokens and job expiry.

**Exit criterion:** a test client can submit a job and retrieve its status/result securely.

### Phase 6 — React frontend (Weeks 5–7)

- Build upload, progress, results, methodology, and privacy pages.
- Add Plotly signal charts.
- Implement job polling and error states.
- Test keyboard navigation and responsive layout.

**Exit criterion:** end-to-end local upload and result display.

### Phase 7 — Storage, security, and deployment (Weeks 7–9)

- Configure private object storage.
- Implement upload limits, rate limits, timeouts, and cleanup.
- Deploy frontend, API, database, queue, and worker.
- Add health checks and basic monitoring.

**Exit criterion:** a public staging deployment with verified private uploads and deletion behavior.

### Phase 8 — Public beta (Weeks 9–12)

- Run end-to-end security and failure tests.
- Benchmark CPU/memory and set safe concurrency.
- Publish privacy, methodology, model, and dataset notes.
- Test from multiple browsers and devices.
- Release with conservative usage limits.

**Exit criterion:** a public beta that works under stated limits and does not overclaim accuracy.

---

## 18. Deployment and Operational Requirements

Deploy these as separate services:

- frontend static hosting;
- FastAPI container;
- worker container;
- PostgreSQL;
- Redis;
- private object storage.

For the first release, use one worker and a small bounded queue. Scale only after measuring resource use. Choose providers based on support for long-running CPU tasks, temporary disk space, network access to object storage, logs, and background processes. Some serverless web platforms are unsuitable for long-running video jobs or persistent workers; verify current limits before committing.

Use environment variables or a secrets manager for credentials. Do not commit secrets. Set health checks, structured logs, error reporting, and a simple alert for repeated worker failures. Logs should not include video contents, tokens, or unnecessary personal data.

---

## 19. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Weak signal due to lighting/motion | Quality checks and inconclusive output |
| New generators synthesize physiological patterns | Multiple datasets and held-out generator/source evaluation |
| Data leakage inflates results | Group-aware splits and provenance tracking |
| Anonymous abuse or excessive cost | Rate limits, bounded queue, file limits, quotas, monitoring |
| Sensitive video exposure | Private storage, short retention, authorization, least privilege |
| Cleanup failure | Scheduled cleanup, storage lifecycle policy, automated deletion tests |
| Deployment cannot run long jobs | Separate background worker; benchmark provider limits |
| Overly broad scope | Complete a working vertical slice before optional features |
| Misleading scores | Document score semantics and calibration |
| Dataset restrictions | Verify terms before download or redistribution |
| Payment fraud and chargebacks | Hosted checkout, processor fraud tools, rate limits, clear pre-purchase terms, dispute runbook |
| Restricted-region users bypass geo-blocking | Server-side checks, attestation, billing cross-check, Terms prohibition, counsel review |

---

## 20. Definition of Done

### Research and algorithm

- [ ] Face ROIs and color signals can be inspected.
- [ ] Signal-processing functions have synthetic-signal tests.
- [ ] Baseline and at least one ML model are compared.
- [ ] Evaluation uses a documented held-out split and avoids obvious leakage.
- [ ] Results include multiple metrics and known failure cases.

### Public product

- [ ] Anonymous visitor can upload a supported video.
- [ ] The app shows progress and useful failure messages.
- [ ] Result displays plots, signal quality, and limitations.
- [ ] Inconclusive results are supported.
- [ ] JSON report export works.
- [ ] Uploads are private.
- [ ] File, duration, concurrency, and rate limits are enforced.
- [ ] Temporary videos and artifacts are automatically deleted.
- [ ] One visitor cannot access another visitor's job result.
- [ ] No uploaded video is used for training without separate explicit opt-in.

### Engineering

- [ ] Setup instructions work from a clean checkout.
- [ ] Dependencies are pinned.
- [ ] Tests and lint checks run in CI.
- [ ] Deployment and secrets are documented.
- [ ] Model card, data card, privacy notice, and limitations are published.
- [ ] No restricted dataset or private video is committed.

---

## 21. Portfolio Presentation

Once the project is implemented and measured, present:

- a live public demo with clear usage limits;
- an architecture diagram;
- a short walkthrough of signal extraction and frequency plots;
- reproducible evaluation commands;
- the model card and data card;
- failure cases and lessons learned;
- unit/integration tests and CI;
- privacy and automatic-deletion behavior.

Potential resume bullet, to update only after the work is complete:

> Built and deployed a public deepfake-analysis web application using React, TypeScript, FastAPI, OpenCV, and SciPy, implementing facial-region signal extraction, feature-based classification, asynchronous video processing, and privacy-conscious anonymous uploads.

Add measured metrics only after you have verified them. Do not claim a particular accuracy, traffic volume, or production status before it is true.

---

## 22. Immediate Next Steps

1. Create the GitHub repository and folder structure.
2. Set up the video pipeline before the public UI.
3. Build signal tests using synthetic periodic data.
4. Choose and verify a small permitted dataset.
5. Implement a baseline and a documented evaluation protocol.
6. Build a local FastAPI endpoint that analyzes one video.
7. Build the React upload/results interface.
8. Add queue, private storage, temporary access, rate limits, and deletion.
9. Deploy to staging and verify end-to-end behavior.
10. Open the public beta only after privacy and resource-limit tests pass.

---

## 23. Legal and Compliance Requirements

> **Important:** This section is a product requirements checklist, not legal advice. The developer is not a lawyer. Laws differ by location and change often, and several items below depend on where PulseCheck is offered, who operates it, and how it is used. Have a qualified attorney review the Privacy Policy, Terms and Conditions, Cookie Policy, consent flows, and the biometric-privacy analysis in LG-11 before public launch. No checklist can guarantee compliance.

These requirements supplement Section 10 (Anonymous Upload, Privacy, and Security). Where they interact with earlier sections, Section 25 records how conflicts are resolved.

### LG-1: Business and operator details

Publish accurate operator details in the footer and on the Privacy, Terms, Cookie, and Contact pages. Use only real, verified information; never invent a company name, address, registration number, or phone number.

| Field | Value (to be completed before launch) |
|---|---|
| Operator / business name | `[TO BE COMPLETED]` |
| Legal entity type (sole proprietor, LLC, etc.) | `[TO BE COMPLETED]` |
| Contact email (privacy, security, misuse reports) | `[TO BE COMPLETED]` |
| Contact phone (optional; only if monitored) | `[TO BE COMPLETED or omit]` |
| Mailing address (if required by applicable law or chosen to publish) | `[TO BE COMPLETED or omit]` |
| Governing law / jurisdiction for Terms | `[TO BE COMPLETED with counsel]` |
| Privacy contact for data-rights requests | `[TO BE COMPLETED]` |
| Policy "last updated" and "effective" dates | Set on every policy change |

- Show a copyright line in the footer: `© [year] [Business name]. All rights reserved.` Generate the year dynamically or update it annually.
- Do not display ® unless a trademark is actually registered. Use ™ only if the mark is genuinely claimed.

### LG-2: Privacy Policy (retain and extend)

Keep the existing privacy requirements (UP-3 through UP-6) and publish them as a dedicated Privacy page. Extend the policy to cover:

- categories of data collected (uploaded video, derived signals and features, job metadata, IP address and request logs, cookie/consent records, analytics data, contact-form data);
- purposes of processing, and, where GDPR/UK GDPR applies, the legal basis for each purpose;
- every third-party processor or recipient (hosting, object storage, database, queue, error reporting, analytics, email) and what each receives;
- international data transfers, if any provider processes data outside the user's region;
- retention periods that match the implemented deletion behavior in UP-4, including log and backup retention;
- how users can exercise rights (access, deletion, correction, opt-out, appeal) where applicable, and how requests are verified given there are no accounts;
- statement that the service is not directed to children (see LG-11);
- how to report a privacy or security concern;
- how policy changes are announced.

Do not use absolute claims such as "we never store anything" unless they are verifiably true for logs, backups, and provider-side copies.

### LG-3: Terms and Conditions page

Add a Terms and Conditions page, linked from the footer and from the upload form. Minimum contents (to be finalized with counsel):

- description of the service as **experimental** and **not forensic proof**;
- acceptable use: upload only media you own or have permission to analyze; no unlawful content; no content depicting minors in sexual or abusive contexts; no attempts to disrupt, scrape, reverse-engineer, or overload the service;
- limited license to process uploaded video solely to produce the analysis;
- no warranty and limitation of liability, written to be enforceable under the chosen governing law;
- prohibition on using results for high-stakes legal, employment, identity, or safety decisions (consistent with Non-Goals);
- rate limiting, suspension, and refusal-of-service rights;
- misuse and takedown reporting contact (see LG-12);
- changes to terms and effective date;
- governing law, dispute resolution, and contact details.

### LG-4: Cookie Policy and cookie consent

- Publish a Cookie Policy listing every cookie, local-storage item, session-storage item, and similar technology: name, purpose, provider, category (strictly necessary / preferences / analytics / marketing), and duration.
- Classify the temporary job access token storage as strictly necessary only if it is used solely to deliver the requested analysis.
- Show a consent banner on first visit **before** any non-essential cookie or script (including Google Analytics) loads.
- Banner requirements: "Accept" and "Reject" options of equal visual prominence; a "Manage preferences" option with per-category toggles; no pre-ticked non-essential categories; no dark patterns.
- Allow users to change or withdraw consent at any time through a persistent footer link.
- Honor browser opt-out preference signals (for example, Global Privacy Control) where applicable.
- Keep a minimal consent record (choice, timestamp, policy version) without storing the video or unnecessary personal data.
- Verify that no non-essential tag fires before consent (test with browser DevTools and a cookie scanner).

### LG-5: Consent on forms

- The upload form requires an **unchecked** checkbox, e.g. "I have permission to upload this video and I have read the Privacy Policy and Terms," with links to both. The Analyze button stays disabled until the box is checked.
- Record the consent text version and timestamp with the job record (not the video).
- The contact/misuse-report form states why data is collected, links to the Privacy Policy, and does not require unnecessary fields.
- Training on uploads remains **off**. Any future training use requires a separate, explicit, optional opt-in (consistent with UP-6).
- Do not pre-tick any consent box and do not bundle optional consents with required ones.
- Include a separate **unchecked** location attestation at checkout/upload (e.g., "I am not located in, and am not a resident of, a jurisdiction where PulseCheck is unavailable"), with a link to the list of unavailable regions (LG-13).

### LG-6: Refund and Cancellation Policy (required)

PulseCheck is a paid service, so a Refund and Cancellation Policy is **mandatory before launch**. Requirements:

- Publish it as its own page, linked from the pricing page, checkout, order confirmation email, and footer.
- State, in plain language: what is refundable, the refund window `[REFUND WINDOW — TO BE COMPLETED]`, how to request a refund, how long refunds take to appear, and which payment method receives the refund.
- State clearly whether analyses that finish with an **Inconclusive** result are billable `[DECISION REQUIRED — see PAY-5]`, and disclose this **before** purchase.
- Provide an automatic refund or re-run when a job fails because of a service error, timeout, or capacity problem.
- Disclose before purchase that results are experimental and not guaranteed to be correct.
- If any recurring billing or expiring credits are offered, add cancellation and renewal terms that comply with applicable state automatic-renewal and consumer-protection laws (PAY-6).
- Have counsel review the policy for consistency with the Terms and consumer-protection rules in the jurisdictions served.

### LG-7: Tracking audit

Before launch, and again on every release that adds scripts, run and document a tracking audit:

- inventory all cookies, local/session storage, pixels, analytics tags, error-reporting SDKs, fonts, and CDN requests (browser DevTools Network/Application tabs plus an automated scanner);
- confirm each item is listed in the Cookie Policy and Privacy Policy;
- confirm no tracking fires before consent where consent is required;
- confirm job IDs and access tokens never appear in analytics events, page URLs sent to analytics, referrer headers, or logs (set `Referrer-Policy: no-referrer` or `strict-origin-when-cross-origin` on result pages);
- remove any tracker that has no documented purpose.

### LG-8: Third-party embed and resource audit

- Avoid third-party embeds (video players, social widgets, maps, chat widgets). If one is necessary, load it only after consent (click-to-load) and disclose it.
- Self-host fonts, icons, and JavaScript libraries (including Plotly.js) instead of loading them from public CDNs, to avoid transmitting visitor IP addresses to third parties without disclosure.
- Disclose the storage provider (Cloudflare R2 or Amazon S3) and other infrastructure providers in the Privacy Policy, since video data is processed or stored by them.
- Use a Content Security Policy that allows only the documented origins; use Subresource Integrity for any unavoidable external script.
- Review each provider's terms and data-processing agreement.

### LG-9: Remove unsupported claims

- No accuracy, detection-rate, or "AI-powered certainty" claims until they come from the documented held-out evaluation in Section 13, and then only with the evaluation conditions stated.
- Do not use phrases such as "100% accurate," "detects all deepfakes," "forensic-grade," "court-ready," "military-grade security," or "trusted by..." unless each is true and provable.
- No testimonials, reviews, logos, awards, press mentions, user counts, or security badges unless real, current, and permitted.
- Maintain a **claims register**: every public claim on the site maps to evidence (test report, dataset result, configuration, or policy) and an owner/date of last verification.
- Refer to Intel, FakeCatcher, and the Netherlands Forensic Institute only as factual citations of prior research. Do not imply affiliation, endorsement, or equivalent performance. Respect third-party trademarks.
- Keep the visible "not forensic proof" notice (Section 14).

### LG-10: Accessibility as a legal and ethical requirement

Target **WCAG 2.2 Level AA**. Detailed implementation requirements are in DS-1 through DS-6. Accessibility complaints and demand letters are a recognized legal risk for public websites, so treat these as release-blocking.

### LG-11: Other applicable laws and risks to review

Review each item below for applicability based on where users are located and who operates the service. Applicability and thresholds vary; confirm with counsel.

| Area | Why it matters for PulseCheck | Required action |
|---|---|---|
| **Biometric privacy laws** (e.g., Illinois BIPA, Texas CUBI, Washington biometric statute, and similar state provisions) | The pipeline detects facial landmarks and analyzes face regions. Some laws treat face-geometry scans as biometric identifiers, with notice, written-consent, retention-schedule, and (in some states) private-right-of-action exposure. | **Decision made: PulseCheck will not be offered in jurisdictions where these laws are determined to apply (LG-13).** Still obtain a legal opinion on which jurisdictions to block and whether landmark-based ROI analysis is covered. Do not store face geometry, templates, or embeddings (already required by UP-7). Do not use the output to identify anyone. Publish a retention and destruction schedule. |
| **U.S. state comprehensive privacy laws** (e.g., California CCPA/CPRA, Minnesota Consumer Data Privacy Act, and others) | Applicability depends on revenue and data-volume thresholds and on small-business exemptions. | Determine whether any threshold applies. If not, still follow the same practices (notice, deletion, opt-out, GPC) as good practice. |
| **GDPR / UK GDPR / ePrivacy** | Applies if the service targets or monitors people in the EU/UK. Face video is personal data; non-essential cookies need prior consent. | Decide whether to geo-restrict or comply fully (legal basis, records of processing, data-subject rights, transfer mechanism, representative if required, DPIA assessment). |
| **Children's privacy** (e.g., COPPA, and age-appropriate design rules) | Videos may show minors, and minors may use the site. | State that the service is not directed to children under 13 (or an age chosen with counsel). Provide a way to request deletion. Do not knowingly collect children's data. Consider an age confirmation on the upload form. |
| **Third-party rights in uploaded videos** (privacy, likeness, publicity, harassment, defamation) | Users may upload videos of other people. Results labeled "more consistent with manipulated" could be used to harass or accuse someone. | Require permission confirmation (LG-5). Use cautious, non-accusatory language (Section 14). Prohibit misuse in the Terms. Provide a misuse-report channel. |
| **Illegal content handling** (e.g., child sexual abuse material, non-consensual intimate imagery) | The service accepts arbitrary user video. | Define an internal procedure: do not retain or redistribute; if apparent child sexual abuse material is discovered, determine reporting obligations under U.S. federal law with counsel. Limit human access to uploads. |
| **Copyright / DMCA** | Users could upload copyrighted video. Temporary, private, automatically deleted processing lowers but may not remove exposure. | Evaluate with counsel whether to designate a DMCA agent and publish a takedown contact. |
| **Consumer-protection and advertising law** (e.g., FTC Act deception rules; rules on fake reviews and testimonials) | Overstated detection claims or fabricated testimonials create deception risk. | Follow LG-9 and the claims register. |
| **Payments, taxes, and consumer-protection law** (sales/VAT collection, automatic-renewal and refund rules, price-disclosure rules, PCI DSS) | PulseCheck is paid. Digital services can be taxable depending on location, and consumer-protection rules govern pricing, refunds, and renewals. | Follow Section 29. Use a hosted checkout so card data never touches the app. Confirm tax registration and collection duties and business-formation requirements with an accountant or attorney. |
| **Accessibility law** (e.g., ADA-based claims, Section 508 where relevant, EU accessibility rules for covered services) | Public websites face accessibility complaints. | Meet WCAG 2.2 AA; publish an accessibility statement and contact method. |
| **Breach notification and security duties** | Private video is sensitive data. | Maintain an incident-response plan, access logging, least privilege, and know applicable state breach-notification deadlines. |
| **Email and phone marketing rules** (CAN-SPAM, TCPA, etc.) | Only relevant if a contact form leads to follow-up messages or a mailing list is added. | No marketing messages without proper consent and unsubscribe handling. |
| **Dataset and software licenses** | Research datasets may prohibit redistribution or commercial use; libraries have license obligations. | Verify dataset terms (Section 13). Keep a third-party license inventory and include required notices. |
| **Export and AI regulation** | Evolving AI transparency rules may apply to some deployments. | Re-check requirements at launch; the product must disclose that results are automated and experimental. |
| **Cost, abuse, and availability risk** | Anonymous public compute can be exploited. | See UP-2 and Section 19. |
| **Reputational and ethical risk** | A false "manipulated" label can harm a person. | Keep the inconclusive state, plain-language limits, and documented false-positive rates. |

### LG-12: Misuse, security, and takedown reporting

- Publish a monitored email address for security, privacy, and misuse reports on the Contact page and footer.
- Provide a response-time commitment per DS-15, but only one the developer can actually meet.
- Document an internal runbook: triage, evidence preservation (without retaining user video beyond policy), user notification, and escalation.

### LG-13: Regional availability restrictions (biometric-privacy jurisdictions)

**Decision:** PulseCheck will not be offered to, sold to, or process uploads from users in jurisdictions where biometric-privacy laws are determined to apply to its face-region analysis.

- **Initial blocked jurisdictions:** `[Illinois, Texas, Washington — CONFIRM WITH COUNSEL BEFORE LAUNCH]`. Counsel may add other states with biometric provisions or adjust the list. Keep the list in configuration (`configs/default.yaml`) so it can change without code changes.
- **Server-side enforcement:** Check region on page load, before checkout, when a payment is confirmed, and at job creation. Hiding the form in the UI alone is not sufficient.
- **Trusted signal only:** Use region data supplied by the CDN or edge provider (or a documented geo-IP service). Accept location headers at the origin only from that provider (for example, by restricting origin access to the CDN) so clients cannot spoof them.
- **Blocked experience:** Show a neutral "PulseCheck is not currently available in your region" page with no upload form and no payment option. Use an appropriate status (HTTP 403 or 451), set `noindex`, and avoid stating legal conclusions on the page. Provide a contact method for users who believe they were blocked in error.
- **User attestation:** Require the unchecked location attestation in LG-5 at checkout and upload.
- **Additional signals:** Where the payment processor supplies billing country or postal code, use it as a cross-check, and cancel and refund any order that conflicts with a blocked region.
- **Terms:** Prohibit use by people located in or resident in blocked jurisdictions and prohibit circumventing the restriction (for example, with a VPN).
- **Privacy:** Use the IP address only to determine region. Store only a coarse result (allowed or blocked, plus region code) with the job. Disclose this in the Privacy Policy and apply the log-retention rules in LG-2.
- **Known limitations:** IP geolocation is imperfect (VPNs, mobile carriers, database errors). The restriction reduces risk but does not eliminate it. Ask counsel whether location-based blocking is sufficient or whether residency-based controls are also needed.
- **Other controls remain:** Blocking does not replace the existing safeguards: no storage of face geometry, templates, or embeddings; no identification of people; published retention and deletion schedule.
- **Review cadence:** Re-review the blocked list with counsel every `[N]` months and whenever relevant laws change.

---

## 24. Website Design, Accessibility, SEO, and Analytics Requirements

### DS-1: Accessibility fixes (WCAG 2.2 AA)

- Run automated checks (axe, Lighthouse) and manual keyboard and screen-reader testing on every page; fix all critical and serious findings before launch.
- Use semantic HTML (headings in order, landmarks, lists, buttons vs. links used correctly).
- Provide a visible keyboard focus indicator on every interactive element.
- Add a "Skip to main content" link.
- Do not rely on color alone to convey status (already required by Section 14).
- Respect `prefers-reduced-motion`.
- Publish a short accessibility statement with a contact method for reporting barriers.

### DS-2: Alt text and chart alternatives

- Every informative image has concise, meaningful `alt` text; purely decorative images use `alt=""`.
- Logo image alt text names the site (e.g., "PulseCheck home").
- Each Plotly chart has an accessible name, a one-sentence summary of what it shows, and a text or table alternative (e.g., key values, dominant frequency per region) so the information is available without sight.
- Social-share images have alt text where the platform supports it.

### DS-3: Color contrast

- Normal text: contrast ratio of at least 4.5:1. Large text and UI components/graphical objects: at least 3:1.
- Verify contrast in default, hover, focus, disabled, error, and chart states, in both light and dark themes if offered.
- Chart series must be distinguishable by more than hue (pattern, marker, label).

### DS-4: Keyboard-friendly forms

- Every input has a programmatically associated visible `<label>`; placeholders are not labels.
- Logical tab order; no keyboard traps; Enter submits where appropriate.
- The drag-and-drop uploader has a fully functional keyboard-accessible file-picker alternative.
- Required fields and formats are stated in text, not only by color.
- Validation errors are shown inline next to the field, linked with `aria-describedby`, announced to assistive technology (e.g., `role="alert"` or a live region), and move focus to the first error on submit.

### DS-5: Clear button and link labels

- Use specific action labels such as "Analyze video," "Choose video file," "Download JSON report," "Delete my video and results," "Accept all cookies," "Reject non-essential cookies."
- Avoid vague labels ("Submit," "Click here," "Learn more") unless the accessible name adds context.
- Icon-only buttons (e.g., menu toggle, close) have accessible names.

### DS-6: Clear error messages

- Provide plain-language messages with a next step for: unsupported format, file too large, video too long, corrupt file, no face detected, service busy, expired job, network failure, and unexpected errors.
- Never expose stack traces, internal IDs, tokens, or file paths in error messages.
- Each error state offers a retry or help link.

### DS-7: Mobile navigation and layout

- Add a responsive **mobile menu** (hamburger button) with `aria-expanded` and `aria-controls`, closing on Escape and on selection, with correct focus management.
- Fix mobile overflow: no horizontal page scroll from 320 px upward; wide content (tables, code, charts) scrolls within its own container; images and charts are fluid (`max-width: 100%`).
- Touch targets are at least 24×24 CSS pixels (aim for 44×44).
- Add a **sticky mobile CTA** ("Analyze a video" / "Upload video") that does not cover form fields, the consent banner, or footer content, and accounts for device safe areas. Hide it on pages where it duplicates an on-screen primary action.
- Test on small and large phones, tablets, and desktop in multiple browsers.

### DS-8: Header, logo, and above-the-fold CTA

- The logo is a link to the home page, with an accessible name.
- Place the primary CTA ("Choose a video to analyze") and the one-line value statement and "experimental, not proof" notice **above the fold** on desktop and mobile, without relying on scrolling.

### DS-9: Contact links

- Email addresses are `mailto:` links; phone numbers (only if published) are `tel:` links in international format.
- Link text or accessible name makes the target clear.

### DS-10: Internal links and breadcrumbs

- Cross-link related pages: Home ↔ Methodology ↔ Privacy ↔ Terms ↔ Cookies ↔ FAQ ↔ Contact, and link the results page to Methodology and Limitations.
- Add **breadcrumbs** on secondary pages (e.g., Home › Methodology), implemented as a labeled `<nav>` with the current page marked `aria-current="page"` and matching `BreadcrumbList` structured data.
- Footer contains links to all policy pages.

### DS-11: Required pages

| Page | Requirement |
|---|---|
| Home / upload | Per FR-1, plus DS-8 above-the-fold CTA |
| Methodology | Per Section 14 |
| Privacy Policy | Per LG-2 |
| Terms and Conditions | Per LG-3 |
| Cookie Policy | Per LG-4 |
| Accessibility statement | Per DS-1 |
| **Pricing** | Price, currency, what is included, tax note, link to Refund and Cancellation Policy (PAY-2) |
| **Checkout / order confirmation / payment failed** | Hosted checkout plus confirmation and failure pages; `noindex`; no order IDs or emails in analytics (PAY-9) |
| **Refund and Cancellation Policy** | Per LG-6 |
| **Region unavailable** | Per LG-13; `noindex` |
| Contact / report misuse | Form or `mailto:` link, consent text per LG-5, response-time promise per DS-15 |
| FAQ | Exactly 5 questions at launch (DS-13) |
| Case studies | Documented examples (DS-14) |
| **Thank-you page** | Shown after the contact/misuse-report form is submitted: confirms receipt, states the response-time promise, links back home, set to `noindex`, contains no tokens or personal data in the URL |
| **Custom 404 page** | Friendly message, search or key links (Home, Methodology, FAQ, Contact), correct HTTP 404 status, `noindex`, matching site design |
| Expired / unavailable job page | Explains expiry and offers a new upload |
| Server-error page | Plain-language message and retry link |

### DS-12: Favicon and branding assets

- Provide a favicon set: `favicon.svg` or `.ico`, a 180×180 Apple touch icon, and PWA-sized icons referenced from the page `<head>` and a web manifest.
- Confirm the favicon appears in major browsers and on iOS/Android home-screen shortcuts.

### DS-13: FAQ (five questions)

Publish exactly five FAQs, written in plain, non-overclaiming language. Suggested questions:

1. **What does PulseCheck do?** Explains the experimental rPPG analysis and the three result states.
2. **Is the result proof that a video is real or fake?** No; it is evidence relative to the model's evaluation scope, and "inconclusive" is a valid outcome.
3. **What happens to my video?** Summarizes the actual implemented retention and deletion behavior (must match UP-4 and the Privacy Policy).
4. **Which videos work best?** Well-lit, front-facing, minimally moving, uncompressed or lightly compressed clips within the stated format, size, and duration limits.
5. **How do I report a problem or misuse?** Points to the Contact page and states the response-time promise.

Mark up the FAQ with `FAQPage` structured data only if the visible page content matches it exactly.

### DS-14: Case studies

PulseCheck has no customers, so "case studies" are **documented analyses of permitted sample videos and evaluation results**, not client stories:

- Use only videos the developer has rights to or that come from datasets whose terms allow showing them; show derived plots rather than raw footage where terms are unclear.
- Include at least: one clear success, one inconclusive example, and one failure/limitation (e.g., low light or heavy compression).
- Report real measured results only; no invented scenarios, clients, or testimonials (LG-9).
- If freelance or client work is later showcased, obtain written permission and keep claims verifiable.

### DS-15: Response-time promise

- State a realistic response-time commitment for contact, privacy, and misuse messages (e.g., "We aim to reply within [N] business days"), using a number the developer can consistently meet.
- Do **not** promise a fixed analysis completion time; show queue position or an estimated range instead, with a note that it may vary (UP-2).

### DS-16: Performance and image compression

- Compress all images; serve modern formats (WebP/AVIF) with fallbacks, responsive `srcset`/`sizes`, explicit `width` and `height` (to prevent layout shift), and `loading="lazy"` for below-the-fold images.
- Lazy-load the Plotly bundle only on the results page.
- Targets (to be measured, not assumed): Largest Contentful Paint under 2.5 s, Cumulative Layout Shift under 0.1, and a Lighthouse performance score recorded in the repo.

### DS-17: SEO and metadata

- Every page has a **unique, descriptive `<title>`** (roughly 50–60 characters) and a **unique meta description** (roughly 140–155 characters).
- One `<h1>` per page; descriptive headings.
- Add a canonical URL for each indexable page.
- Add **`robots.txt`** at the site root: allow public informational pages, disallow API paths (e.g., `/v1/`), job/result pages, and thank-you pages; reference `sitemap.xml`. Remember that `robots.txt` is not an access-control mechanism, so protect results with authorization (UP-5) and `noindex` headers.
- Provide `sitemap.xml` listing only indexable pages.
- Mark job, results, thank-you, 404, and expired-job pages `noindex`.

### DS-18: Social sharing

- Add Open Graph and Twitter/X card tags (`og:title`, `og:description`, `og:image`, `og:url`, `twitter:card`) on public pages.
- Use a 1200×630 share image with legible text and alt text; verify with the platforms' preview tools.
- Never generate share previews from user-uploaded content. Do not add share buttons that load third-party scripts without consent (LG-8).

### DS-19: Structured data and local business schema

- Add `Organization` and `WebSite`/`WebApplication` structured data with accurate details from LG-1.
- Add **`LocalBusiness` schema only if** the operator is a real local business with a genuine address and service area that it chooses to publish. PulseCheck is an online tool, so by default use `Organization` and `WebApplication`. Never publish fabricated address, hours, ratings, or review markup.
- Validate all structured data with Google's Rich Results Test and the Schema Markup Validator.

### DS-20: Google Analytics

- Use Google Analytics 4 only after the visitor consents (LG-4); implement Google Consent Mode so tags respect the choice. Disclose it in the Privacy and Cookie Policies.
- Configure data retention to the shortest practical period, disable advertising features and Google Signals, and do not link to advertising products.
- Send sanitized page paths only: strip job IDs, tokens, query strings, and file names from every URL and event sent to analytics (LG-7, UP-5). Prefer keeping tokens in headers or the URL fragment rather than path/query strings.
- Do not send uploaded-file content, face data, or result details as events.
- Record basic events only (page views, upload started, analysis completed/failed counts) with no personal data.
- If privacy risk or consent friction is a concern, a cookieless, privacy-focused analytics tool is an acceptable alternative; document the choice either way.

---

## 25. Reconciliation with Earlier Requirements

| Earlier requirement | Conflict or interaction | Resolution |
|---|---|---|
| Non-Goals: no payments (v2.0/v2.1) | Product is now paid | Payments moved in scope (Section 29); Refund and Cancellation Policy is mandatory (LG-6) |
| UP-5: tokens must stay out of analytics/logs | Google Analytics requested | DS-20: sanitize URLs; consent first |
| UP-2: IP-based rate limiting | Privacy/tracking | Disclose IP use and log retention in Privacy Policy; minimal retention |
| Section 14: "not forensic proof" notice | Above-the-fold CTA | DS-8: show both together above the fold |
| UP-6: privacy notice before upload | New consent checkbox | LG-5: checkbox links to Privacy and Terms; version recorded |
| Section 8: Plotly.js/CDN | Third-party resource audit | LG-8: self-host libraries and fonts |
| Section 21: portfolio/resume claims | Unsupported-claims rule | LG-9: only verified, measured claims |
| UP-2: "do not promise a free usage allowance" | Product is paid, not free | No free tier or free-allowance wording anywhere; remove "free" from copy and metadata |
| UP-2: IP-based rate limiting | IP also used for region check | Use IP transiently for region only; store coarse result; disclose (LG-13) |
| Anonymous access (Sections 1, 6) | Guest checkout collects an email for receipts | Collect only the email; disclose in Privacy Policy; keep order records separate from video data (PAY-1, PAY-7) |
| UP-5: tokens out of analytics/logs | Payments and order data | Never send order IDs, emails, or tokens to analytics (PAY-9, DS-20) |

---

## 26. Additional Testing and QA for Legal and Design Requirements

- **Consent:** banner appears before non-essential scripts; Reject truly blocks them; preference changes persist; Global Privacy Control is honored where applicable.
- **Tracking:** automated cookie/tracker scan on every release; verify no tokens in analytics requests or `Referer` headers.
- **Forms:** consent box unchecked by default; Analyze disabled until checked; keyboard-only completion of upload and contact forms; error messages announced by screen reader.
- **Accessibility:** axe and Lighthouse with zero critical/serious issues; manual keyboard-only and screen-reader pass (e.g., NVDA/VoiceOver); contrast verification; 200% zoom and 320 px reflow test.
- **Mobile:** no horizontal scroll at 320 px; mobile menu and sticky CTA behavior on iOS Safari and Android Chrome.
- **Pages:** 404 returns HTTP 404; thank-you page is `noindex` and carries no sensitive URL data; all footer and breadcrumb links resolve; `mailto:` and `tel:` links work on desktop and mobile.
- **SEO/social:** unique titles and descriptions verified; `robots.txt` and `sitemap.xml` reachable; structured data validates; Open Graph previews verified.
- **Performance:** Lighthouse/Web Vitals recorded; image sizes and formats checked.
- **Claims:** claims register reviewed line by line against site copy.
- **Legal review:** attorney sign-off recorded (or conscious decision to proceed documented) for Privacy, Terms, Cookies, Refund Policy, and the biometric-law analysis.
- **Regional blocking:** a blocked region is refused at page load, checkout, payment confirmation, and job creation; forged region headers from a client are ignored; a VPN smoke test is documented; the attestation is required; the blocked-region list changes through configuration.
- **Payments:** checkout in the processor's test mode; webhook signature verification; duplicate and replayed webhooks are idempotent; analysis starts only after confirmed payment; failed payment, cancelled checkout, refund, and chargeback flows are tested; no card data appears in logs or the database; price and tax display match what is charged.

---

## 27. Schedule and Roadmap Impact

Add the following to the roadmap:

- **Phase 7 additions (Weeks 7–9):** business-details placeholders resolved; consent banner, cookie inventory, tracker audit, and third-party resource audit; self-hosted fonts and libraries.
- **Phase 8 additions (Weeks 9–12):** write and publish Privacy, Terms, Cookie, FAQ, Contact, accessibility statement, thank-you, 404, and expired-job pages; claims-register review; accessibility, mobile, SEO, and performance QA; attorney review.

- **Payments and regional availability (Weeks 6–9):** payment-processor integration and webhooks, pricing and checkout pages, refund flow, tax setup, edge geolocation enforcement and region-unavailable page.

Estimated added effort (planning estimate, not a guarantee): roughly **25–50 hours** in total for the legal, design, payment, and regional-availability additions, mostly in Phases 6–8, plus time for external legal and tax review. This may extend the total solo-developer estimate in Section 2 by about 2–5 weeks.

---

## 28. Additional Definition of Done

### Legal and privacy

- [ ] Business details completed and displayed accurately (LG-1); footer shows © year and business name.
- [ ] Privacy Policy, Terms and Conditions, and Cookie Policy published and linked from the footer and upload form.
- [ ] Cookie consent banner works; no non-essential tag fires before consent.
- [ ] Upload form has unchecked, versioned consent checkbox.
- [ ] Refund and Cancellation Policy published and linked from pricing, checkout, confirmation emails, and footer (LG-6).
- [ ] Tracking and third-party embed audits completed and documented.
- [ ] Claims register complete; no unsupported claims remain.
- [ ] Biometric-privacy, children's-privacy, GDPR/UK, and state-law applicability reviewed; decisions recorded.
- [ ] Misuse/takedown contact and incident runbook in place.
- [ ] Blocked-jurisdiction list confirmed with counsel and enforced server-side; region-unavailable page live (LG-13).
- [ ] Paid checkout works end to end through a hosted processor; no card data stored; webhooks verified; refunds and failed-job handling tested (Section 29).
- [ ] Pricing, taxes, and business registration confirmed with an accountant or attorney; no "free" wording remains on the site.

### Design, accessibility, and SEO

- [ ] WCAG 2.2 AA checks pass; alt text, contrast, keyboard operation, and clear labels verified.
- [ ] Mobile menu, sticky mobile CTA, and no-overflow layout verified on real devices.
- [ ] Logo links home; email and phone links are clickable; CTA is above the fold.
- [ ] Favicon, unique titles, meta descriptions, canonical tags, `robots.txt`, `sitemap.xml`, social tags, and valid structured data in place.
- [ ] Custom 404, thank-you, expired-job, and error pages implemented.
- [ ] Five FAQs, case studies (real, permitted), breadcrumbs, and internal links published.
- [ ] Images compressed; performance metrics recorded.
- [ ] Google Analytics (or alternative) gated by consent, with sanitized URLs and documented settings.
- [ ] Response-time promise published and achievable.

---

## 29. Commercial Model, Payments, and Regional Availability

### 29.1 Decision

PulseCheck is a **paid service**. There is no free tier and no free-usage allowance. Do not use the word "free" in site copy, metadata, FAQs, social tags, or marketing. The service is also **not offered in blocked jurisdictions** (LG-13).

### 29.2 Commercial placeholders (to be completed before launch)

| Item | Value |
|---|---|
| Pricing model (per analysis, credit pack, or subscription) | `[TO BE DECIDED]` |
| Price and currency | `[PRICE — TO BE COMPLETED]` |
| Payment processor | `[TO BE CHOSEN]` |
| Tax handling approach | `[TO BE COMPLETED with accountant]` |
| Refund window and conditions | `[REFUND WINDOW — TO BE COMPLETED]` |
| Whether Inconclusive results are billable | `[DECISION REQUIRED]` |
| Business registration and tax ID | `[TO BE COMPLETED]` |
| Blocked jurisdictions | `[Illinois, Texas, Washington — CONFIRM WITH COUNSEL]` |

All earlier placeholders (LG-1 business details, governing law, contact details) remain in force.

### 29.3 Payment requirements

**PAY-1: Guest checkout, no accounts.** Use the processor's hosted checkout page. The application must never receive, store, or log card numbers. Collect only an email address for the receipt (personal data, disclosed in the Privacy Policy). Link each payment to a job through an opaque order ID. Confirm payment through a server-side, signature-verified webhook, handled idempotently. Do not start analysis until payment is confirmed (or authorized, if the capture-on-completion model is chosen). Never send the video, access tokens, or result data to the payment processor.

**PAY-2: Price transparency.** Show the price, currency, and any applicable taxes or fees before the visitor commits, on the upload page, Pricing page, and checkout. No hidden fees. State what one purchase includes (for example, one analysis within the stated size and duration limits).

**PAY-3: Taxes and business setup.** Determine whether sales tax, VAT, or similar taxes apply to digital services in the places PulseCheck is offered, and use a tax-calculation tool or accountant. Confirm business formation, registration, and tax-ID requirements before taking payments.

**PAY-4: Policy updates.** Update the Terms (pricing, payment, taxes, refunds, failed payments, chargebacks, purchasers must be adults able to contract), Privacy Policy (payment processor, receipt email, order-record retention), and Cookie Policy (checkout cookies and scripts).

**PAY-5: Refund rules and the Inconclusive result.** Implement the policy in LG-6. Decide and disclose before purchase whether Inconclusive results are billable. Because the product states that results are evidence, not proof, never offer or imply a satisfaction or accuracy guarantee that the evaluation in Section 13 does not support. Automatically refund or re-run failed jobs caused by service errors.

**PAY-6: Recurring billing and credits (only if offered).** Obtain clear affirmative consent, disclose renewal terms and price changes, send reminders where required, and provide easy cancellation. Confirm whether prepaid credits or expiry dates trigger additional state requirements.

**PAY-7: Receipts and order records.** Email a receipt that links to the Refund and Cancellation Policy. Keep order records only as long as tax and accounting rules require, stored separately from video and result data, and disclosed in the Privacy Policy.

**PAY-8: Fraud and disputes.** Enable the processor's fraud tools, keep the rate limits in UP-2, and write a short dispute and chargeback runbook. Keep the pre-purchase terms and consent records (LG-5) as evidence.

**PAY-9: Pages and analytics.** Pricing, checkout, order-confirmation, and payment-failed pages are required (DS-11). Mark checkout and confirmation pages `noindex`, and exclude order IDs, emails, and tokens from analytics URLs and events (DS-20).

### 29.4 Revised user journey (paid)

1. Visit the site; the region check runs (LG-13). Blocked visitors see the region-unavailable page.
2. Read the explanation, price, limits, privacy notice, and Terms.
3. Select a supported video and confirm the consent and location attestation (LG-5).
4. Pay through the hosted checkout; receive an emailed receipt.
5. After payment confirmation, the upload is analyzed and job status is shown.
6. View results and optionally download the JSON report.
7. Temporary video data and job artifacts expire automatically (UP-4).
8. If the job fails because of a service error, the refund or re-run policy applies.

---

## Appendix A: Requirement Traceability

| Requested item | Where addressed |
|---|---|
| Keep the privacy policy | LG-2 |
| Terms and conditions page | LG-3, DS-11 |
| Cookies policy | LG-4, DS-11 |
| Form consent | LG-5 |
| Check cookie consent | LG-4, Section 26 |
| Refund policy | LG-6 |
| Check tracking | LG-7, Section 26 |
| Check third-party embeds | LG-8 |
| Fix accessibility | LG-10, DS-1 |
| Alt text on images | DS-2 |
| Check color contrast | DS-3 |
| Make forms keyboard friendly | DS-4 |
| Use clear button labels | DS-5 |
| Remove unsupported claims | LG-9 |
| Add business details | LG-1 |
| Check other applicable laws and risks | LG-11, LG-12 |
| Mobile menu | DS-7 |
| Favicon | DS-12 |
| Meta descriptions | DS-17 |
| Custom 404 page | DS-11 |
| Copyright and year on business name | LG-1 |
| Compress images | DS-16 |
| Error messages | DS-6 |
| Fix mobile overflow | DS-7 |
| Clickable logo | DS-8 |
| Clickable email and phone | DS-9 |
| CTA above the fold | DS-8 |
| Internal links | DS-10 |
| Thank-you page | DS-11 |
| Breadcrumbs | DS-10 |
| Case studies | DS-14 |
| 5 FAQs | DS-13 |
| Response-time promise | DS-15 |
| Sticky mobile CTA | DS-7 |
| robots.txt | DS-17 |
| Unique page titles | DS-17 |
| Social sharing | DS-18 |
| Local business schema | DS-19 |
| Google Analytics | DS-20 |
| Avoid availability in states where biometric law applies | LG-13, LG-11 |
| Product is not free (paid service) | Section 29, LG-6 |
| Keep placeholders | LG-1, Section 29.2, LG-13 |

---

**Guiding principle:** PulseCheck should be public and easy to try, but the service must be designed so an anonymous visitor cannot expose another person's upload, exhaust unlimited compute, or mistake an experimental model score for proof that a video is authentic or fake.

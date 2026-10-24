# PulseCheck

Experimental rPPG-based deepfake analysis. Results are evidence, not proof.
See [PulseCheck_PRD.md](PulseCheck_PRD.md) for the full requirements.

## Status

- Phase 1: video decoding and metadata inspection (`pulsecheck.video`).
- Phase 2: face landmarks, forehead/cheek ROIs, tracking coverage (`pulsecheck.face`) and
  per-ROI RGB traces (`pulsecheck.signals`).

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

Keep videos and datasets out of Git (see `.gitignore`).

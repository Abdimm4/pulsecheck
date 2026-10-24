# Data card: FaceForensics++ (draft)

Status: draft, written before the dataset was downloaded (access approval pending). Nothing
here reports results. Confirm every fact against the official repository and terms once
access is granted.

## Dataset

- **Name:** FaceForensics++ (FF++), Rössler et al., "FaceForensics++: Learning to Detect
  Manipulated Facial Images", ICCV 2019.
- **Source:** https://github.com/ondyari/FaceForensics (download script issued after a
  signed access request).
- **Content:** 1000 original videos collected from YouTube, mostly one frontal face per
  video, and manipulated versions made by five methods: Deepfakes, FaceSwap and FaceShifter
  (face swapping), and Face2Face and NeuralTextures (facial reenactment). Each method gives
  up to 1000 manipulated videos. FF++ now distributes videos only, not extracted images.
- **Optional extra:** DeepFakeDetection (DFD, contributed by Google/Jigsaw) uses paid actors
  and is distributed alongside FF++. PulseCheck skips it by default (`--include-dfd` adds it).
- **Naming:** a manipulated video `TTT_SSS` is target video TTT with source identity SSS.

## License and permitted use

- FF++ may be used for **non-commercial research and educational purposes only**. Videos,
  frames and derived per-video data (feature tables, predictions) must not be redistributed
  or committed to Git. Re-check the current terms before each use and whenever the
  project's status changes (PRD LG-6).
- PulseCheck is a free, non-commercial project (PRD v2.3, Section 29). Every trained model's
  JSON sidecar records `training_dataset` (name, compression, methods, split source,
  license) and `project_status`, so provenance travels with the model.
- Everything derived from FF++ lives under `data/` and `models/`, which Git ignores. The
  manifest, its sidecar and each feature table carry the license notes.
- The people in the YouTube source videos did not consent to this research use
  individually. Do not publish frames or faces from the dataset, including in debug output.

## What PulseCheck uses

- **Compression: `c23`** (H.264, quantisation parameter 23), the "high quality" setting most
  FF++ work reports alongside raw and c40. Raw is very large; c40 removes much of the
  subtle colour variation that rPPG depends on. c23 is a compromise, and it still smooths
  skin colour, so pulse signals are weaker than in uncompressed video. Real and fake videos
  are both re-encoded at c23, but fakes have been through more processing steps.
- **Manifest:** `scripts/build_manifest.py` writes one row per video with the PRD Section 13
  fields (`video_id`, `path`, `label`, `source_dataset`, `subject_id`, `manipulation_type`,
  `original_video_id`, `split`, `license_notes`) plus `source_video_id`, `group_id` and
  `compression`. `subject_id` is the identity whose face is shown: the source for face-swap
  methods, the target for reenactment methods.
- **Features:** only per-video colour-signal statistics (`pulsecheck.features`) and runtime
  are stored. No frames, landmarks or face geometry.

## Splits

- PulseCheck uses the **official FF++ splits** (`train.json`, `val.json`, `test.json`; 720,
  140 and 140 original videos, listed as 360/70/70 pairs).
- A manipulated video `TTT_SSS` is assigned to the split containing the pair (TTT, SSS) in
  either order, and originals TTT and SSS are in that same split, so an original and
  everything derived from it never cross splits (PRD Section 13 leakage rules).
- The manifest builder, training and evaluation all re-check this: videos linked through
  `original_video_id`, `source_video_id` or `subject_id` form a group, and any group that
  spans splits raises an error. A manipulated video whose pair is not listed but whose two
  originals sit in different splits is also an error.
- Without split files, and always for DFD, linked videos are grouped and each group is
  assigned by a seeded hash in roughly 72/14/14 proportions.
- FF++ ids are YouTube videos, not verified identities. The same person may appear in more
  than one source video, so identity-level leakage across splits cannot be ruled out.
- Hyperparameters, decision thresholds and the inconclusive band are chosen on validation.
  The test split is scored once; model sidecars record each test evaluation, and a repeat
  needs `--force` and is flagged in the report.

## Class balance

- With all five methods there are up to 5000 manipulated videos for 1000 originals (1:5).
- Options: train on the imbalanced set (classifiers use balanced class weights), or build the
  manifest with `--balance`, which downsamples fakes per split to the real count, spread
  evenly across methods. `--limit-per-class N` caps each class per split for pilot runs.
- Accuracy alone is misleading under imbalance; reports also give precision, recall,
  specificity, F1, ROC-AUC and per-method results.

## Known limitations

- **Manipulation methods are from 2016-2020.** Results do not carry over to current face
  generators, full-frame synthesis, or audio-driven video.
- **Partial manipulation.** Reenactment methods change mainly the mouth and expression, and
  swap methods replace the inner face, so forehead and cheek regions may keep original skin
  in some frames. Region-consistency features may therefore behave very differently by
  method; report per-method results.
- **Source material.** YouTube videos with varied resolution, lighting, compression history
  and motion; mostly frontal, mostly news and interview footage. Demographics are not
  documented or balanced, so performance across skin tones, ages and genders is unknown.
  rPPG is known to be harder on darker skin and in low light.
- **Labels are per video.** A manipulated video may contain frames where the manipulation
  failed or the face is absent; no per-frame labels are used.
- **Duration and frame rate vary.** Short or low-frame-rate videos are rejected by the signal
  checks and recorded as low quality or failures rather than dropped.
- **Single dataset.** A model trained and tested on FF++ alone says nothing about other
  datasets or real-world uploads. Held-out-method and cross-dataset evaluation (e.g. Celeb-DF)
  are still to do.

"""NFI rPPG deepfake dataset manifest builder.

Hugging Face: NetherlandsForensicInstitute/rppg-deepfake-detection. Layout:

    <root>/real/CelebDF/id3_0007.mp4              real clip of Celeb-DF identity id3
    <root>/fake/<Tool>_FS/023_<Tool>_fa001.mp4    face swap: target video 023, source face fa001
    <root>/fake/<Tool>_SYN/7.mp4                  fully AI-generated (Luma: deepfake7.mp4)
    <root>/annotations.json                       {"Sora,1.mp4": {gender, skin_color, ...}}

manipulation_type is the tool name ("none" for real clips) and fake_family is
real, face_swap or synthetic. Annotations, where present, go into the
`attributes` column as JSON.

Grouping (leakage). Neither the dataset card nor the NFI repository documents
where the face-swap targets and source faces come from, so the links below
come from the file names plus a visual check of one frame per clip:

- real clips link through their Celeb-DF identity (subject idN);
- a face swap links to its target video NNN and its source face faXXX. The
  same (NNN, faXXX) pairs recur across all five tools, and different NNN
  numbers can show the same volunteer, but those always share a faXXX, so
  linking both ids merges them. No face-swap person appeared among the
  Celeb-DF identities, so face swaps are not linked to real clips;
- Runway, Pika, PixVerse and KlingAI clip N were generated from the same
  prompt (matching annotations, matching scene descriptions), so they are
  linked through prompt N;
- consecutive Sora clips with identical annotations often show the same
  generated character, so such runs are linked;
- Luma clips have no annotations and some neighbours are near duplicates, so
  each block of LUMA_BLOCK consecutive clips is linked.

Splits. With about 500 videos the groups are split into N_FOLDS + 1 grouped
folds, stratified by manipulation type. One fold is the held-out test split;
the others form the train split with cv_fold 0..N_FOLDS-1 for cross-validated
model, threshold and quality-gate selection. No group may span folds.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from pulsecheck import __version__
from pulsecheck.dataset.manifest import (
    LeakageError,
    ManifestRow,
    SplitRecord,
    check_leakage,
    link_groups,
)

SOURCE_DATASET = "NFI-rPPG"
REAL = "real"
FACE_SWAP = "face_swap"
SYNTHETIC = "synthetic"
FAMILIES = (REAL, FACE_SWAP, SYNTHETIC)
N_FOLDS = 5
LUMA_BLOCK = 3
# Tools whose clip N shares a prompt with the other tools' clip N.
SHARED_PROMPT_TOOLS = frozenset({"Runway", "Pika", "PixVerse", "KlingAI"})

LICENSE_NOTES = (
    "NFI rPPG deepfake dataset (huggingface.co/datasets/NetherlandsForensicInstitute/"
    "rppg-deepfake-detection): no license stated. Real clips derive from Celeb-DF "
    "(non-commercial research terms). Treat as non-commercial research data; do not "
    "redistribute videos, frames or derived per-video data."
)

_REAL = re.compile(r"^(id\d+)_(\d+)$")
_FS = re.compile(r"^(\d{3})_([A-Za-z0-9]+)_(fa\d+)$")
_SYN = re.compile(r"^(?:deepfake)?(\d+)$")
_FOLDER = re.compile(r"^(.+)_(FS|SYN)$")


@dataclass
class NfiOptions:
    n_folds: int = N_FOLDS
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2")


@dataclass
class NfiResult:
    rows: list[ManifestRow]
    meta: dict
    warnings: list[str] = field(default_factory=list)


def load_annotations(root: Path) -> dict[tuple[str, str], dict]:
    """(tool, file name) -> annotation dict; empty if the file is absent."""
    path = root / "annotations.json"
    if not path.is_file():
        return {}
    out = {}
    for key, value in json.loads(path.read_text(encoding="utf-8")).items():
        tool, _, name = key.partition(",")
        out[(tool.strip(), name.strip())] = value
    return out


def _videos(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*.mp4") if p.is_file()) if folder.is_dir() else []


def scan(root: str | Path) -> tuple[list[ManifestRow], list[str]]:
    """Every recognised NFI video (split unassigned, synthetic groups not yet linked)."""
    root = Path(root)
    annotations = load_annotations(root)
    rows: list[ManifestRow] = []
    warnings: list[str] = []
    common = {"source_dataset": SOURCE_DATASET, "split": "unassigned"}

    def rel(path: Path) -> str:
        return path.relative_to(root).as_posix()

    real_dir = root / "real" / "CelebDF"
    if not real_dir.is_dir():
        warnings.append(f"no real clips folder at {real_dir}")
    for path in _videos(real_dir):
        m = _REAL.match(path.stem)
        if not m:
            warnings.append(f"skipped unrecognised real file {rel(path)}")
            continue
        rows.append(
            ManifestRow(
                video_id=f"real_{path.stem}",
                path=rel(path),
                label="real",
                subject_id=f"celebdf:{m.group(1)}",
                manipulation_type="none",
                original_video_id=f"celebdf:{path.stem}",
                license_notes=LICENSE_NOTES,
                fake_family=REAL,
                **common,
            )
        )

    fake_dir = root / "fake"
    folders = sorted(p for p in fake_dir.iterdir() if p.is_dir()) if fake_dir.is_dir() else []
    if not folders:
        warnings.append(f"no fake folders under {fake_dir}")
    used_annotations = set()
    for folder in folders:
        fm = _FOLDER.match(folder.name)
        if not fm:
            warnings.append(f"skipped unrecognised fake folder {rel(folder)}")
            continue
        tool, kind = fm.groups()
        for path in _videos(folder):
            if kind == "FS":
                m = _FS.match(path.stem)
                if not m or m.group(2) != tool:
                    warnings.append(f"skipped unrecognised face-swap file {rel(path)}")
                    continue
                target, _, face = m.groups()
                links = {
                    "original_video_id": f"fs_target:{target}",
                    "source_video_id": f"fs_face:{face}",
                    "subject_id": f"fs_face:{face}",  # a swap shows the source face
                }
                family = FACE_SWAP
            else:
                m = _SYN.match(path.stem)
                if not m:
                    warnings.append(f"skipped unrecognised synthetic file {rel(path)}")
                    continue
                links = {
                    "original_video_id": f"syn:{tool}:{int(m.group(1))}",
                    "subject_id": "",  # set by link_synthetic
                }
                family = SYNTHETIC
            note = annotations.get((tool, path.name))
            if note is not None:
                used_annotations.add((tool, path.name))
            rows.append(
                ManifestRow(
                    video_id=f"{tool}_{kind}_{path.stem}",
                    path=rel(path),
                    label="fake",
                    manipulation_type=tool,
                    license_notes=LICENSE_NOTES,
                    fake_family=family,
                    attributes=json.dumps(note, sort_keys=True) if note else "",
                    **links,
                    **common,
                )
            )
    unused = set(annotations) - used_annotations
    if unused:
        warnings.append(f"{len(unused)} annotations match no video, e.g. {sorted(unused)[0]}")
    return link_synthetic(rows), warnings


def _syn_index(row: ManifestRow) -> int:
    return int(row.original_video_id.rsplit(":", 1)[1])


def link_synthetic(rows: list[ManifestRow]) -> list[ManifestRow]:
    """Set subject_id on synthetic rows to the shared-prompt / near-duplicate link key."""
    sora = sorted(
        (r for r in rows if r.fake_family == SYNTHETIC and r.manipulation_type == "Sora"),
        key=_syn_index,
    )
    # Runs of consecutive Sora clips with identical annotations share a key.
    sora_key: dict[str, str] = {}
    previous: tuple[int, str] | None = None
    prev_id = ""
    for row in sora:
        index = _syn_index(row)
        if (
            previous is not None
            and row.attributes
            and index == previous[0] + 1
            and row.attributes == previous[1]
        ):
            sora_key[row.video_id] = sora_key[prev_id]
        else:
            sora_key[row.video_id] = f"sora_run:{index}"
        previous, prev_id = (index, row.attributes), row.video_id

    out = []
    for row in rows:
        if row.fake_family != SYNTHETIC:
            out.append(row)
            continue
        tool, index = row.manipulation_type, _syn_index(row)
        if tool in SHARED_PROMPT_TOOLS:
            key = f"prompt:{index}"
        elif tool == "Sora":
            key = sora_key[row.video_id]
        elif tool == "Luma":
            key = f"luma_block:{(index - 1) // LUMA_BLOCK}"
        else:
            key = f"syn:{tool}:{index}"  # unknown tool: the clip alone
        out.append(replace(row, subject_id=key))
    return out


def group_names(rows: list[ManifestRow]) -> list[str]:
    """Linked-group name per row: the smallest link key in its component."""
    components = link_groups(r.link_keys() for r in rows)
    names: dict[str, str] = {}
    for row, comp in zip(rows, components, strict=True):
        key = min(row.link_keys())
        names[comp] = min(names.get(comp, key), key)
    return [names[c] for c in components]


def assign_folds(rows: list[ManifestRow], options: NfiOptions) -> list[ManifestRow]:
    """Grouped folds stratified by manipulation type; fold 0 becomes the test split."""
    groups = group_names(rows)
    strata = [r.manipulation_type for r in rows]
    n_groups = len(set(groups))
    k = options.n_folds + 1
    if n_groups < k:
        raise ValueError(f"only {n_groups} linked groups; cannot make {k} grouped folds")
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=options.seed)
    fold_of = np.empty(len(rows), dtype=int)
    for fold, (_, idx) in enumerate(splitter.split(np.zeros(len(rows)), strata, groups)):
        fold_of[idx] = fold
    out = []
    for row, group, fold in zip(rows, groups, fold_of, strict=True):
        if fold == 0:
            out.append(replace(row, split="test", cv_fold="", group_id=f"link:{group}"))
        else:
            out.append(replace(row, split="train", cv_fold=str(fold - 1), group_id=f"link:{group}"))
    return out


def check_fold_leakage(rows: list[ManifestRow]) -> dict:
    """check_leakage over splits and, within train, over CV folds (raises LeakageError)."""
    split_summary = check_leakage(rows)
    train = [r for r in rows if r.split == "train"]
    if any(not r.cv_fold for r in train):
        raise LeakageError("train rows without a cv_fold")
    folds = check_leakage(
        [SplitRecord(r.video_id, f"fold{r.cv_fold}", tuple(r.link_keys())) for r in train]
    )
    return {"splits": split_summary, "cv_folds": folds}


def _key(row: ManifestRow) -> tuple:
    return (
        row.split != "train",
        row.cv_fold,
        FAMILIES.index(row.fake_family),
        row.manipulation_type,
        row.video_id,
    )


def build_nfi_manifest(root: str | Path, options: NfiOptions | None = None) -> NfiResult:
    options = options or NfiOptions()
    root = Path(root).resolve()
    rows, warnings = scan(root)
    if not rows:
        raise FileNotFoundError(f"no NFI videos found under {root}")
    rows = sorted(assign_folds(rows, options), key=_key)
    leakage = check_fold_leakage(rows)
    for split in ("train", "test"):
        families = {r.fake_family for r in rows if r.split == split}
        if families != set(FAMILIES):
            warnings.append(f"{split} lacks families {sorted(set(FAMILIES) - families)}")

    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        fold = "test" if row.split == "test" else f"fold{row.cv_fold}"
        counts[fold][row.fake_family] += 1
    meta = {
        "dataset": SOURCE_DATASET,
        "root": str(root),
        "compression": None,
        "methods": sorted({r.manipulation_type for r in rows} - {"none"}),
        "include_dfd": False,
        "split_source": (
            f"StratifiedGroupKFold(n_splits={options.n_folds + 1}, seed={options.seed}) over "
            "linked groups, stratified by manipulation type; fold 0 = test, the rest = train "
            f"with cv_fold 0..{options.n_folds - 1}"
        ),
        "sampling": {"limit_per_class_per_split": None, "balance": False, "seed": options.seed},
        "n_folds": options.n_folds,
        "grouping": {
            "real": "Celeb-DF identity (idN)",
            "face_swap": "union of target video NNN and source face faXXX",
            "synthetic": (
                f"prompt index for {sorted(SHARED_PROMPT_TOOLS)}; runs of consecutive Sora "
                f"clips with identical annotations; Luma blocks of {LUMA_BLOCK} clips"
            ),
            "assumption": (
                "Face-swap people are not Celeb-DF identities (visual check; undocumented "
                "by the dataset). See pulsecheck/dataset/nfi.py."
            ),
        },
        "groups": leakage,
        "fold_counts": {k: dict(sorted(v.items())) for k, v in sorted(counts.items())},
        "annotated_videos": sum(bool(r.attributes) for r in rows),
        "license_notes": LICENSE_NOTES,
        "non_commercial_use_only": True,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "pulsecheck_version": __version__,
        "warnings": warnings[:200],
        "warning_count": len(warnings),
    }
    return NfiResult(rows, meta, warnings)

"""Source clips for the injection benchmark (PRD FR-12, Section 13).

Only real (pristine) videos from the deepfake datasets' **train** splits are
used: FaceForensics++ c23 originals and the normalised NFI-rPPG real clips.
Nothing from a validation or test split is read. Clips are drawn with a fixed
seed, spread over as many source groups as possible, and the groups are then
divided into a **dev** half (method selection, quality calibration) and a
**held-out** half (reported once for the current and the selected pipeline).
A group never spans both halves.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from pulsecheck.dataset import read_manifest, read_sidecar

DEV = "dev"
HELDOUT = "heldout"
BENCH_SPLITS = (DEV, HELDOUT)


@dataclass(frozen=True)
class BenchmarkClip:
    clip_id: str  # "<dataset>:<video_id>"
    dataset: str  # short dataset key, e.g. ffpp or nfi
    video_id: str
    path: str  # absolute path of the source video
    group: str  # leakage group from the manifest (dataset-prefixed)
    split: str  # dev or heldout
    max_duration_s: float  # analysed span from the first frame

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SourceSpec:
    dataset: str
    manifest: Path
    count: int
    max_duration_s: float
    root: Path | None = None  # default: the manifest sidecar's root


def select_clips(spec: SourceSpec, seed: int = 0) -> list[BenchmarkClip]:
    """`spec.count` real train clips, round-robin over shuffled groups, split dev/held-out."""
    root = spec.root or Path(read_sidecar(spec.manifest).get("root", "."))
    rows = [
        r
        for r in read_manifest(spec.manifest)
        if r.label == "real" and r.split == "train" and (root / r.path).is_file()
    ]
    by_group: dict[str, list] = defaultdict(list)
    for row in sorted(rows, key=lambda r: r.video_id):
        by_group[row.group_id or row.original_video_id].append(row)
    rng = random.Random(f"{seed}:{spec.dataset}")
    groups = sorted(by_group)
    rng.shuffle(groups)
    for group in groups:
        rng.shuffle(by_group[group])
    picked = []
    depth = 0
    while len(picked) < spec.count and any(len(by_group[g]) > depth for g in groups):
        for group in groups:
            if len(picked) < spec.count and len(by_group[group]) > depth:
                picked.append(by_group[group][depth])
        depth += 1
    used = sorted({r.group_id or r.original_video_id for r in picked})
    rng.shuffle(used)
    dev_groups = set(used[: (len(used) + 1) // 2])
    return [
        BenchmarkClip(
            clip_id=f"{spec.dataset}:{r.video_id}",
            dataset=spec.dataset,
            video_id=r.video_id,
            path=str((root / r.path).resolve()),
            group=f"{spec.dataset}:{r.group_id or r.original_video_id}",
            split=DEV if (r.group_id or r.original_video_id) in dev_groups else HELDOUT,
            max_duration_s=spec.max_duration_s,
        )
        for r in sorted(picked, key=lambda r: r.video_id)
    ]

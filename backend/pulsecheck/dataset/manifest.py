"""Dataset manifests and leakage checks (PRD Section 13).

A manifest is a CSV with one row per video plus a JSON sidecar
(`<manifest>.json`) recording the dataset root, build options, counts and
license notes. Paths in the CSV are relative to the dataset root, so the
manifest itself holds no video data and can be rebuilt anywhere.

Leakage rule: every video is linked to the source videos it was made from
(`original_video_id`, `source_video_id`) and to its `subject_id`. Videos
linked directly or transitively form one group, and a group must never span
more than one split. `check_leakage` recomputes the groups from those links
rather than trusting the `group_id` column, and raises LeakageError on any
violation.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path

SPLITS = ("train", "val", "test")
UNASSIGNED = "unassigned"
LABELS = ("real", "fake")
REAL_MANIPULATION = "none"


class LeakageError(ValueError):
    """Raised when a source group (or a video) appears in more than one split."""


@dataclass(frozen=True)
class ManifestRow:
    # PRD Section 13 fields.
    video_id: str
    path: str  # relative to the dataset root, POSIX separators
    label: str  # real or fake
    source_dataset: str
    subject_id: str
    manipulation_type: str  # "none" for real videos
    original_video_id: str  # the video this one is (or derives from); itself for real videos
    split: str  # train, val, test or unassigned
    license_notes: str
    # Extra fields used for grouping and reporting.
    source_video_id: str = ""  # the second video a manipulation draws on, if any
    group_id: str = ""  # split group key (informational; check_leakage recomputes)
    compression: str = ""

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ValueError(f"{self.video_id}: label must be one of {LABELS}")
        if self.split not in (*SPLITS, UNASSIGNED):
            raise ValueError(f"{self.video_id}: unknown split {self.split!r}")

    def link_keys(self) -> list[str]:
        """Identity/source nodes this video is tied to, namespaced by dataset."""
        keys = [self.original_video_id, self.source_video_id, self.subject_id]
        return [f"{self.source_dataset}:{k}" for k in keys if k]


MANIFEST_FIELDS = tuple(f.name for f in fields(ManifestRow))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sidecar_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".json")


def write_manifest(rows: Iterable[ManifestRow], path: str | Path, meta: Mapping) -> Path:
    """Write the CSV and its JSON sidecar; returns the sidecar path."""
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    sidecar = {
        **meta,
        "manifest_sha256": file_sha256(path),
        "counts": manifest_counts(rows),
    }
    side = sidecar_path(path)
    side.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return side


def read_manifest(path: str | Path) -> list[ManifestRow]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(MANIFEST_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"manifest {path} lacks columns {sorted(missing)}")
        return [ManifestRow(**{k: row[k] for k in MANIFEST_FIELDS}) for row in reader]


def read_sidecar(path: str | Path) -> dict:
    side = sidecar_path(path)
    return json.loads(side.read_text(encoding="utf-8")) if side.is_file() else {}


def manifest_counts(rows: Iterable[ManifestRow]) -> dict:
    by_split: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        by_split[row.split][row.label] += 1
        by_split[row.split][f"method:{row.manipulation_type}"] += 1
    return {split: dict(sorted(c.items())) for split, c in sorted(by_split.items())}


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, key: str) -> str:
        self.parent.setdefault(key, key)
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != root:
            self.parent[key], key = root, self.parent[key]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def link_groups(link_keys: Iterable[list[str]]) -> list[str]:
    """Connected-component id for each item, given each item's link keys."""
    link_keys = [list(k) for k in link_keys]
    uf = _UnionFind()
    for i, keys in enumerate(link_keys):
        node = f"__item:{i}"
        uf.find(node)
        for key in keys:
            uf.union(node, key)
    return [uf.find(f"__item:{i}") for i in range(len(link_keys))]


@dataclass(frozen=True)
class SplitRecord:
    """Minimal view of a row for leakage checks (manifest or feature table)."""

    video_id: str
    split: str
    links: tuple[str, ...]


def check_leakage(records: Iterable[SplitRecord | ManifestRow], ignore_unassigned=True) -> dict:
    """Raise LeakageError if a video id or a linked group spans several splits.

    Returns a summary with the number of groups per split.
    """
    recs = [
        r if isinstance(r, SplitRecord) else SplitRecord(r.video_id, r.split, tuple(r.link_keys()))
        for r in records
    ]
    if ignore_unassigned:
        recs = [r for r in recs if r.split != UNASSIGNED]

    problems = []
    splits_by_video: dict[str, set[str]] = defaultdict(set)
    for r in recs:
        splits_by_video[r.video_id].add(r.split)
    for video_id, splits in splits_by_video.items():
        if len(splits) > 1:
            problems.append(f"video {video_id} appears in splits {sorted(splits)}")

    groups = link_groups(r.links for r in recs)
    members: dict[str, list[SplitRecord]] = defaultdict(list)
    for r, g in zip(recs, groups, strict=True):
        members[g].append(r)
    for group in members.values():
        splits = {r.split for r in group}
        if len(splits) > 1:
            sample = ", ".join(f"{r.video_id}->{r.split}" for r in group[:6])
            problems.append(
                f"group of {len(group)} linked videos spans splits {sorted(splits)} ({sample})"
            )
    if problems:
        shown = "\n  ".join(problems[:20])
        more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        raise LeakageError(f"split leakage detected ({len(problems)} problems):\n  {shown}{more}")

    per_split = Counter(next(iter({r.split for r in g})) for g in members.values())
    return {"groups": len(members), "groups_per_split": dict(sorted(per_split.items()))}

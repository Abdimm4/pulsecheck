"""FaceForensics++ (FF++) manifest builder.

Expected layout (official download script):

    <root>/original_sequences/youtube/<compression>/videos/071.mp4
    <root>/manipulated_sequences/<Method>/<compression>/videos/071_054.mp4

`TTT_SSS` is target video TTT with source identity SSS. DeepFakeDetection
(DFD) folders are skipped unless `include_dfd` is set:

    <root>/original_sequences/actors/<compression>/videos/01__scene.mp4
    <root>/manipulated_sequences/DeepFakeDetection/<compression>/videos/01_02__scene__ID.mp4

Splits: the official train/val/test JSON files list pairs such as
["071", "054"]. A manipulated video TTT_SSS goes to the split holding the pair
(TTT, SSS) in either order, and originals TTT and SSS go to that same split.
Without split files, and always for DFD (which the official splits do not
cover), linked videos are grouped and each group is assigned by a seeded hash
in roughly the official 72/14/14 proportions.

subject_id is the identity whose face is shown: the source SSS for face-swap
methods (Deepfakes, FaceSwap, FaceShifter, DFD) and the target TTT for
reenactment methods (Face2Face, NeuralTextures), which keep the target's face
and transfer the source's expressions. FF++ ids are YouTube videos, not
verified person identities; the same person may appear in several videos.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

from pulsecheck import __version__
from pulsecheck.dataset.manifest import (
    LABELS,
    SPLITS,
    UNASSIGNED,
    LeakageError,
    ManifestRow,
    check_leakage,
    link_groups,
)

SOURCE_DATASET = "FaceForensics++"
METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter")
SWAP_METHODS = frozenset({"Deepfakes", "FaceSwap", "FaceShifter", "DeepFakeDetection"})
DFD_METHOD = "DeepFakeDetection"
COMPRESSIONS = ("raw", "c23", "c40")
HASH_SPLIT_FRACTIONS = {"train": 0.72, "val": 0.14, "test": 0.14}

LICENSE_NOTES = (
    "FaceForensics++ terms of use: non-commercial research and educational use only; "
    "do not redistribute videos, frames or derived per-video data. "
    "https://github.com/ondyari/FaceForensics (re-check current terms)"
)
DFD_LICENSE_NOTES = (
    "DeepFakeDetection (Google/Jigsaw), distributed with FaceForensics++ under its "
    "non-commercial research terms; re-check the DFD terms before use"
)

_YT_ORIGINAL = re.compile(r"^(\d{3})$")
_YT_FAKE = re.compile(r"^(\d{3})_(\d{3})$")
_DFD_ORIGINAL = re.compile(r"^(\d{2})__(.+)$")
_DFD_FAKE = re.compile(r"^(\d{2})_(\d{2})__(.+?)(?:__[A-Za-z0-9]+)?$")


@dataclass
class ManifestOptions:
    compression: str = "c23"
    methods: tuple[str, ...] = METHODS
    splits_dir: Path | None = None
    include_dfd: bool = False
    limit_per_class: int | None = None  # per class, in each split
    balance: bool = False  # downsample fakes to the real count in each split
    seed: int = 0

    def __post_init__(self) -> None:
        if self.compression not in COMPRESSIONS:
            raise ValueError(f"compression must be one of {COMPRESSIONS}")
        unknown = set(self.methods) - set(METHODS)
        if unknown:
            raise ValueError(f"unknown methods {sorted(unknown)}; known: {METHODS}")
        if self.limit_per_class is not None and self.limit_per_class < 1:
            raise ValueError("limit_per_class must be positive")


@dataclass
class ManifestResult:
    rows: list[ManifestRow]
    meta: dict
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class OfficialSplits:
    by_pair: dict[frozenset[str], str]
    by_id: dict[str, str]
    pair_of: dict[str, frozenset[str]]

    def group_key(self, *videos: str) -> str:
        """Sorted pair key; for one original, the pair it is listed in."""
        ids = set(videos)
        if len(ids) == 1:
            ids = set(self.pair_of.get(videos[0], ids))
        return "_".join(sorted(ids))

    def for_original(self, video: str) -> str | None:
        return self.by_id.get(video)

    def for_pair(self, target: str, source: str) -> str | None:
        split = self.by_pair.get(frozenset((target, source)))
        if split is None and self.by_id.get(target) == self.by_id.get(source):
            split = self.by_id.get(target)
        return split


def load_official_splits(splits_dir: str | Path) -> OfficialSplits:
    """Read train/val/test.json; raise LeakageError if an id is in several splits."""
    splits_dir = Path(splits_dir)
    by_pair: dict[frozenset[str], str] = {}
    by_id: dict[str, str] = {}
    pair_of: dict[str, frozenset[str]] = {}
    conflicts = []
    for split in SPLITS:
        path = splits_dir / f"{split}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing split file {path}")
        for pair in json.loads(path.read_text(encoding="utf-8")):
            if len(pair) != 2:
                raise ValueError(f"{path}: expected pairs, got {pair!r}")
            ids = [str(x) for x in pair]
            by_pair[frozenset(ids)] = split
            for vid in ids:
                pair_of[vid] = frozenset(ids)
                if by_id.setdefault(vid, split) != split:
                    conflicts.append(f"{vid} in {by_id[vid]} and {split}")
    if conflicts:
        raise LeakageError("official split files overlap: " + "; ".join(conflicts[:20]))
    return OfficialSplits(by_pair, by_id, pair_of)


def _videos(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*.mp4") if p.is_file()) if folder.is_dir() else []


def scan(root: str | Path, options: ManifestOptions) -> tuple[list[ManifestRow], list[str]]:
    """List every recognised video under root (split still unassigned)."""
    root = Path(root)
    comp = options.compression
    rows: list[ManifestRow] = []
    warnings: list[str] = []

    def rel(path: Path) -> str:
        return path.relative_to(root).as_posix()

    def add(**kw) -> None:
        rows.append(
            ManifestRow(
                source_dataset=SOURCE_DATASET,
                split=UNASSIGNED,
                compression=comp,
                **kw,
            )
        )

    originals = root / "original_sequences" / "youtube" / comp / "videos"
    if not originals.is_dir():
        warnings.append(f"no original videos folder at {originals}")
    for path in _videos(originals):
        m = _YT_ORIGINAL.match(path.stem)
        if not m:
            warnings.append(f"skipped unrecognised original file {rel(path)}")
            continue
        vid = m.group(1)
        add(
            video_id=f"original_{vid}",
            path=rel(path),
            label="real",
            subject_id=vid,
            manipulation_type="none",
            original_video_id=vid,
            license_notes=LICENSE_NOTES,
        )

    for method in options.methods:
        folder = root / "manipulated_sequences" / method / comp / "videos"
        if not folder.is_dir():
            warnings.append(f"no {method} folder at {folder}")
        for path in _videos(folder):
            m = _YT_FAKE.match(path.stem)
            if not m:
                warnings.append(f"skipped unrecognised {method} file {rel(path)}")
                continue
            target, source = m.groups()
            add(
                video_id=f"{method}_{target}_{source}",
                path=rel(path),
                label="fake",
                subject_id=source if method in SWAP_METHODS else target,
                manipulation_type=method,
                original_video_id=target,
                source_video_id=source,
                license_notes=LICENSE_NOTES,
            )

    if options.include_dfd:
        rows.extend(_scan_dfd(root, comp, warnings, rel))
    return rows, warnings


def _scan_dfd(root: Path, comp: str, warnings: list[str], rel) -> list[ManifestRow]:
    rows = []
    common = {
        "source_dataset": SOURCE_DATASET,
        "split": UNASSIGNED,
        "compression": comp,
        "license_notes": DFD_LICENSE_NOTES,
    }
    folder = root / "original_sequences" / "actors" / comp / "videos"
    if not folder.is_dir():
        warnings.append(f"no DFD original folder at {folder}")
    for path in _videos(folder):
        m = _DFD_ORIGINAL.match(path.stem)
        if not m:
            warnings.append(f"skipped unrecognised DFD original file {rel(path)}")
            continue
        actor, scene = m.groups()
        rows.append(
            ManifestRow(
                video_id=f"dfd_original_{path.stem}",
                path=rel(path),
                label="real",
                subject_id=f"actor:{actor}",
                manipulation_type="none",
                original_video_id=f"dfd:{actor}__{scene}",
                **common,
            )
        )
    folder = root / "manipulated_sequences" / DFD_METHOD / comp / "videos"
    if not folder.is_dir():
        warnings.append(f"no {DFD_METHOD} folder at {folder}")
    for path in _videos(folder):
        m = _DFD_FAKE.match(path.stem)
        if not m:
            warnings.append(f"skipped unrecognised {DFD_METHOD} file {rel(path)}")
            continue
        target, source, scene = m.groups()
        rows.append(
            ManifestRow(
                video_id=f"{DFD_METHOD}_{path.stem}",
                path=rel(path),
                label="fake",
                subject_id=f"actor:{source}",
                manipulation_type=DFD_METHOD,
                # The target actor's original of the same scene; linking it also
                # links the target actor's other videos through subject_id.
                original_video_id=f"dfd:{target}__{scene}",
                source_video_id=f"actor:{target}",
                **common,
            )
        )
    return rows


def _hash_split(key: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    edge = 0.0
    for split, fraction in HASH_SPLIT_FRACTIONS.items():
        edge += fraction
        if u < edge:
            return split
    return SPLITS[-1]


def _is_dfd(row: ManifestRow) -> bool:
    return row.manipulation_type == DFD_METHOD or row.video_id.startswith("dfd_")


def assign_splits(
    rows: list[ManifestRow], official: OfficialSplits | None, seed: int = 0
) -> tuple[list[ManifestRow], list[str]]:
    """Set split and group_id on each row; see the module docstring for the rules."""
    warnings = []
    out = list(rows)
    if official is not None:
        for i, row in enumerate(out):
            if _is_dfd(row):
                continue
            if row.label == "real":
                split = official.for_original(row.original_video_id)
                group = official.group_key(row.original_video_id)
            else:
                target, source = row.original_video_id, row.source_video_id
                split = official.for_pair(target, source)
                group = official.group_key(target, source)
                known = (official.for_original(target), official.for_original(source))
                if split is None and None not in known:
                    raise LeakageError(
                        f"split leakage: {row.video_id} combines videos from the {known[0]} and "
                        f"{known[1]} splits; a group spans splits"
                    )
            if split is None:
                warnings.append(f"{row.video_id}: not covered by the split files; unassigned")
                out[i] = replace(row, split=UNASSIGNED, group_id=f"pair:{group}")
            else:
                out[i] = replace(row, split=split, group_id=f"pair:{group}")

    pending = [i for i, row in enumerate(out) if official is None or _is_dfd(row)]
    if pending:
        components = link_groups(out[i].link_keys() for i in pending)
        # Name each group after its smallest link key so the hash, and hence the
        # split, does not depend on which other videos happen to be present.
        names: dict[str, str] = {}
        for i, comp in zip(pending, components, strict=True):
            key = min(out[i].link_keys())
            names[comp] = min(names.get(comp, key), key)
        for i, comp in zip(pending, components, strict=True):
            name = names[comp]
            out[i] = replace(out[i], split=_hash_split(name, seed), group_id=f"link:{name}")
    return out, warnings


def _sample(rows: list[ManifestRow], n: int, rng: random.Random) -> list[ManifestRow]:
    """Up to n rows, spread evenly across manipulation types (round robin)."""
    if n >= len(rows):
        return list(rows)
    by_type: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in rows:
        by_type[row.manipulation_type].append(row)
    queues = []
    for key in sorted(by_type):
        group = sorted(by_type[key], key=lambda r: r.video_id)
        rng.shuffle(group)
        queues.append(group)
    picked: list[ManifestRow] = []
    while len(picked) < n:
        for queue in queues:
            if queue and len(picked) < n:
                picked.append(queue.pop())
    return picked


def sample_rows(rows: list[ManifestRow], options: ManifestOptions) -> list[ManifestRow]:
    """Apply limit_per_class and balance within each split (seeded)."""
    if options.limit_per_class is None and not options.balance:
        return rows
    rng = random.Random(options.seed)
    out = []
    for split in (*SPLITS, UNASSIGNED):
        real = sorted((r for r in rows if r.split == split and r.label == "real"), key=_key)
        fake = sorted((r for r in rows if r.split == split and r.label == "fake"), key=_key)
        limit = options.limit_per_class
        if limit is not None:
            real = _sample(real, limit, rng)
        fake_n = len(fake) if limit is None else min(limit, len(fake))
        if options.balance:
            fake_n = min(fake_n, len(real))
        out.extend(real)
        out.extend(_sample(fake, fake_n, rng))
    return out


def _key(row: ManifestRow) -> tuple:
    split = (*SPLITS, UNASSIGNED).index(row.split)
    return split, LABELS.index(row.label), row.manipulation_type, row.video_id


def build_manifest(root: str | Path, options: ManifestOptions | None = None) -> ManifestResult:
    options = options or ManifestOptions()
    root = Path(root).resolve()
    rows, warnings = scan(root, options)
    if not rows:
        raise FileNotFoundError(
            f"no FF++ videos found under {root} for compression {options.compression}"
        )
    official = load_official_splits(options.splits_dir) if options.splits_dir else None
    rows, split_warnings = assign_splits(rows, official, options.seed)
    warnings += split_warnings
    leakage = check_leakage(rows)  # raises LeakageError
    rows = sorted(sample_rows(rows, options), key=_key)
    check_leakage(rows)
    meta = {
        "dataset": SOURCE_DATASET,
        "root": str(root),
        "compression": options.compression,
        "methods": list(options.methods),
        "include_dfd": options.include_dfd,
        "split_source": (
            f"official:{Path(options.splits_dir).resolve()}"
            if options.splits_dir
            else f"hash(seed={options.seed}, fractions={HASH_SPLIT_FRACTIONS})"
        ),
        "sampling": {
            "limit_per_class_per_split": options.limit_per_class,
            "balance": options.balance,
            "seed": options.seed,
        },
        "groups_before_sampling": leakage,
        "license_notes": LICENSE_NOTES,
        "non_commercial_use_only": True,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "pulsecheck_version": __version__,
        "warnings": warnings[:200],
        "warning_count": len(warnings),
    }
    return ManifestResult(rows, meta, warnings)

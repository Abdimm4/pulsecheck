"""The per-video feature table written by batch extraction (PRD FR-11).

One CSV row per video: manifest metadata, extraction status, runtime, the
FEATURE_NAMES values (empty cell = missing) and a JSON column with the reason
for every missing feature. Failed videos keep a row with status "failed" and
a reason code, so they are counted rather than silently dropped. No frames,
landmarks or face geometry are stored.

Rows are appended as videos finish, so a file can hold several rows for one
video after a retry; readers keep the last one.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path

from pulsecheck.dataset.manifest import SplitRecord
from pulsecheck.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, FeatureVector

META_COLUMNS = (
    "video_id",
    "label",
    "split",
    "manipulation_type",
    "source_dataset",
    "subject_id",
    "original_video_id",
    "source_video_id",
    "group_id",
)
STATUS_COLUMNS = (
    "status",  # ok or failed
    "failure_reason",  # reason code when failed
    "failure_detail",
    "runtime_s",
    "n_frames",
    "duration_s",
    "fps",
    "feature_schema_version",
    "pulse_method",
)
TABLE_COLUMNS = (*META_COLUMNS, *STATUS_COLUMNS, *FEATURE_NAMES, "missing_json")
STATUS_OK = "ok"
STATUS_FAILED = "failed"


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value) if math.isfinite(value) else ""
    return str(value)


def format_row(meta: Mapping, status: Mapping, features: FeatureVector | None) -> dict[str, str]:
    row = {c: _cell(meta.get(c)) for c in META_COLUMNS}
    row.update({c: _cell(status.get(c)) for c in STATUS_COLUMNS})
    values = features.values if features else {}
    row.update({name: _cell(values.get(name)) for name in FEATURE_NAMES})
    row["missing_json"] = json.dumps(features.missing, sort_keys=True) if features else ""
    return row


def append_rows(path: str | Path, rows: Iterable[Mapping[str, str]]) -> None:
    """Append rows, writing the header first if the file is new or empty."""
    path = Path(path)
    new = not path.is_file() or path.stat().st_size == 0
    if not new:
        check_header(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS)
        if new:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
        handle.flush()


def check_header(path: str | Path) -> None:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle), [])
    if tuple(header) != TABLE_COLUMNS:
        raise ValueError(
            f"{path} has a different column layout (feature schema changed?); "
            "write to a new file or pass --overwrite"
        )


def read_table(path: str | Path) -> list[dict[str, str]]:
    """All rows, one per video (the last row written wins), in first-seen order."""
    by_id: dict[str, dict[str, str]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(TABLE_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"feature table {path} lacks columns {sorted(missing)[:5]}...")
        for row in reader:
            by_id[row["video_id"]] = row
    return list(by_id.values())


def parse_float(cell: str) -> float | None:
    if cell is None or cell == "":
        return None
    value = float(cell)
    return value if math.isfinite(value) else None


def row_features(row: Mapping[str, str]) -> FeatureVector:
    """Rebuild the FeatureVector stored in a table row (status ok only)."""
    values = {name: parse_float(row[name]) for name in FEATURE_NAMES}
    missing = json.loads(row["missing_json"]) if row.get("missing_json") else {}
    for name, value in values.items():
        if value is None:
            missing.setdefault(name, "not_computed")
    version = int(row["feature_schema_version"] or FEATURE_SCHEMA_VERSION)
    return FeatureVector(values, missing, row["pulse_method"] or "chrom", version)


def split_records(rows: Iterable[Mapping[str, str]]) -> list[SplitRecord]:
    """Leakage-check view of table rows (same link keys as ManifestRow)."""
    out = []
    for row in rows:
        keys = [row["original_video_id"], row["source_video_id"], row["subject_id"]]
        links = tuple(f"{row['source_dataset']}:{k}" for k in keys if k)
        out.append(SplitRecord(row["video_id"], row["split"], links))
    return out

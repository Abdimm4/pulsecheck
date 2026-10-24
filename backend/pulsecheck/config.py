"""Load configs/default.yaml (or $PULSECHECK_CONFIG).

Only the sections the code reads are interpreted here; each module turns its
section into a typed config (e.g. SignalConfig.from_mapping).
"""

from __future__ import annotations

import os
from pathlib import Path

from ruamel.yaml import YAML

REPO_ROOT = Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    return Path(os.environ.get("PULSECHECK_CONFIG", REPO_ROOT / "configs" / "default.yaml"))


def load_config(path: str | Path | None = None) -> dict:
    """Parse the YAML config; an empty file gives an empty dict."""
    path = Path(path) if path is not None else default_config_path()
    with path.open(encoding="utf-8") as handle:
        return YAML(typ="safe").load(handle) or {}

"""The per-job JSON report (PRD FR-10).

Contains versions, timestamps, measured video metadata, configuration,
quality checks, features, the result with its score semantics, warnings,
caveats and expiry. It never contains video bytes, frames, face geometry,
file paths, the client's IP (or its hash) or the access token.
"""

from __future__ import annotations

import math
from datetime import datetime

from pulsecheck import __version__
from pulsecheck.features import FEATURE_SCHEMA_VERSION
from pulsecheck.jobs.pipeline import AnalysisOutput

REPORT_SCHEMA_VERSION = 1
APP_NAME = "PulseCheck"

CAVEATS = (
    "PulseCheck is an experimental research tool. Its result is evidence, not proof, "
    "and does not establish whether a video is authentic.",
    "Result states describe how consistent the measured signals are with the examples "
    "the method was built or trained on; they do not generalise beyond that scope.",
    "Scores are not probabilities unless the report says the model is calibrated.",
    "Compression, lighting, motion, make-up, skin tone and camera processing all affect "
    "remote pulse signals and can make the result inconclusive or wrong.",
)


def _json_safe(value):
    """Replace non-finite floats with None, recursively (reports never hold NaN)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    return value


def _iso(moment: datetime | None) -> str | None:
    return moment.isoformat(timespec="seconds") if moment else None


def build_report(
    *,
    job_id: str,
    created_at: datetime,
    completed_at: datetime,
    expires_at: datetime,
    container: str | None,
    output: AnalysisOutput,
    model_name: str,
    model_version: str,
    upload_limits: dict,
    retention_policy: str,
) -> dict:
    video = output.video.to_dict()
    video.pop("path", None)  # server-side storage path; not the user's business
    video["container"] = container
    features = output.features
    warnings = list(output.video.warnings)
    if output.tracking is not None:
        warnings += output.tracking.warnings
    if output.result.get("state") == "inconclusive":
        warnings += output.result.get("reasons", [])
    report = {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": _iso(completed_at),
        "app": {"name": APP_NAME, "version": __version__},
        "model": {
            "name": model_name,
            "version": model_version,
            "calibrated": bool(output.result.get("calibrated", False)),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
        },
        "job": {
            "id": job_id,
            "created_at": _iso(created_at),
            "completed_at": _iso(completed_at),
        },
        "expiry": {"expires_at": _iso(expires_at), "policy": retention_policy},
        "video": video,
        "configuration": {"upload_limits": upload_limits, **output.settings},
        "quality_checks": {
            "tracking": output.tracking.to_dict() if output.tracking else None,
            "signals": output.signal_summary,
        },
        "features": features.to_dict() if features else None,
        "result": output.result,
        "warnings": warnings,
        "caveats": list(CAVEATS),
    }
    return _json_safe(report)

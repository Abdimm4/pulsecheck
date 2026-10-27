"""Request and response schemas (OpenAPI at /docs and /openapi.json)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ErrorDetail(BaseModel):
    code: str = Field(description="Stable machine-readable error code")
    message: str = Field(description="Human-readable explanation, safe to show to users")
    fields: list[dict[str, Any]] | None = Field(
        None, description="Invalid request fields (validation errors only)"
    )


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: str = Field(description='"ok" or "degraded"')
    database: str
    queue: str


class ModelInfo(BaseModel):
    name: str
    version: str


class VersionResponse(BaseModel):
    app: str
    version: str
    api_version: str
    model: ModelInfo
    report_schema_version: int
    feature_schema_version: int


class CreateJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    size_bytes: int | None = Field(
        None, ge=1, description="Declared file size, for an early rejection; checked again later"
    )


class UploadLimitsInfo(BaseModel):
    allowed_formats: list[str]
    max_size_bytes: int
    max_duration_s: float
    max_resolution: list[int] = Field(description="[long side, short side] in pixels")
    max_frames: int


class UploadInstructions(BaseModel):
    method: str
    url: str = Field(description="Upload URL; relative URLs are on this API")
    headers: dict[str, str]
    max_bytes: int
    expires_at: datetime
    requires_access_token: bool = Field(
        description="Send `Authorization: Bearer <access_token>` with the upload"
    )


class CreateJobResponse(BaseModel):
    job_id: str
    access_token: str = Field(
        description=(
            "Shown once and stored only as a hash. Send it as `Authorization: Bearer` on "
            "every request for this job; never put it in a URL."
        )
    )
    status: str
    created_at: datetime
    expires_at: datetime
    upload: UploadInstructions
    limits: UploadLimitsInfo


class JobError(BaseModel):
    code: str
    message: str | None


class ResultSummary(BaseModel):
    state: str = Field(
        description="more_consistent_with_real, more_consistent_with_manipulated or inconclusive"
    )
    label: str
    score: float | None
    score_semantics: str | None
    calibrated: bool
    reasons: list[str]
    signal_quality: float | None
    face_coverage: float | None
    usable_roi_count: float | None
    duration_s: float | None
    decoded_frame_count: int | None


class JobStatusResponse(BaseModel):
    job_id: str
    status: str = Field(
        description="created, uploading, queued, processing, completed, failed or expired"
    )
    created_at: datetime
    updated_at: datetime
    expires_at: datetime
    upload_expires_at: datetime | None = Field(
        description="Deadline to finish the upload (unfinished jobs only)"
    )
    queued_at: datetime | None
    started_at: datetime | None
    finished_at: datetime | None
    deletion_status: str
    expiry_reason: str | None
    error: JobError | None
    model: ModelInfo | None
    result: ResultSummary | None
    report_url: str | None


class ReportModelInfo(BaseModel):
    name: str
    version: str
    calibrated: bool
    feature_schema_version: int


class ReportExpiry(BaseModel):
    expires_at: str
    policy: str


class Report(BaseModel):
    """The FR-10 JSON report. Never contains video bytes, tokens or client data."""

    model_config = ConfigDict(extra="forbid")

    report_schema_version: int
    generated_at: str
    app: dict[str, str]
    model: ReportModelInfo
    job: dict[str, str | None]
    expiry: ReportExpiry
    video: dict[str, Any]
    configuration: dict[str, Any]
    quality_checks: dict[str, Any]
    features: dict[str, Any] | None
    result: dict[str, Any]
    warnings: list[str]
    caveats: list[str]

"""FastAPI service (PRD Section 12).

Run with `uvicorn --factory pulsecheck.api.app:create_app` (see README). The flow is:

1. POST /v1/jobs -> job ID, a one-time access token and upload instructions;
2. PUT the video bytes to the upload URL (here: PUT /v1/jobs/{id}/upload);
3. POST /v1/jobs/{id}/complete-upload -> validated and queued;
4. poll GET /v1/jobs/{id} until completed or failed, then GET .../report;
5. DELETE /v1/jobs/{id} to delete everything early.

Every /v1/jobs/{id} endpoint needs `Authorization: Bearer <access_token>`.
Unknown jobs, malformed IDs and missing or wrong tokens all get the same 404,
so a job ID alone reveals nothing, not even whether the job exists.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import text
from starlette.requests import ClientDisconnect

from pulsecheck import __version__
from pulsecheck.api.errors import ApiError, error_response, install_error_handlers, job_not_found
from pulsecheck.api.ratelimit import RateLimiter
from pulsecheck.api.schemas import (
    CreateJobRequest,
    CreateJobResponse,
    ErrorResponse,
    HealthResponse,
    JobStatusResponse,
    ModelInfo,
    Report,
    UploadInstructions,
    UploadLimitsInfo,
    VersionResponse,
)
from pulsecheck.features import FEATURE_SCHEMA_VERSION
from pulsecheck.jobs.auth import client_hash
from pulsecheck.jobs.models import ExpiryReason, Job, JobStatus
from pulsecheck.jobs.queue import QueueUnavailable
from pulsecheck.jobs.runtime import Runtime
from pulsecheck.jobs.service import (
    JobConflict,
    JobNotFound,
    JobService,
    ServiceBusy,
    TooManyActiveJobs,
)
from pulsecheck.jobs.settings import Settings
from pulsecheck.jobs.validation import FILE_TOO_LARGE, UploadRejected
from pulsecheck.reporting import REPORT_SCHEMA_VERSION
from pulsecheck.storage import ObjectTooLarge

log = logging.getLogger("pulsecheck.api")

API_VERSION = "v1"
# JSON request bodies are tiny; only the upload endpoint streams large bodies.
MAX_JSON_BODY_BYTES = 16 * 1024
BINARY_SCHEMA = {"type": "string", "format": "binary"}
BUSY_MESSAGE = "The service is busy. Please try again later."

bearer = HTTPBearer(auto_error=False, description="The access token returned at job creation")

ERRORS = {
    404: {"model": ErrorResponse, "description": "Unknown job, or missing or wrong token"},
    409: {"model": ErrorResponse, "description": "Not possible in the job's current status"},
}


def get_runtime(request: Request) -> Runtime:
    return request.app.state.runtime


def get_service(request: Request) -> JobService:
    return JobService(get_runtime(request))


ServiceDep = Annotated[JobService, Depends(get_service)]


def authorized_job(
    job_id: str,
    service: ServiceDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> Job:
    token = credentials.credentials if credentials else None
    try:
        return service.authorize(job_id, token)
    except JobNotFound:
        raise job_not_found() from None


JobDep = Annotated[Job, Depends(authorized_job)]


def client_ip(request: Request, settings: Settings) -> str:
    if settings.trust_proxy:
        # The proxy appends the address it saw, so only the last entry is trustworthy;
        # earlier entries come from the client and would let it dodge rate limits.
        forwarded = request.headers.get("x-forwarded-for", "")
        last = forwarded.split(",")[-1].strip()
        if last:
            return last
    return request.client.host if request.client else "unknown"


def busy(settings: Settings) -> ApiError:
    retry = {"Retry-After": str(settings.capacity.retry_after_s)}
    return ApiError(503, "service_busy", BUSY_MESSAGE, retry)


def conflict(exc: JobConflict) -> ApiError:
    return ApiError(409, exc.code, f"{exc.message} (status: {exc.status.value})")


def rejected(exc: UploadRejected) -> ApiError:
    return ApiError(413 if exc.code == FILE_TOO_LARGE else 422, exc.code, exc.message)


def job_response(job: Job) -> JobStatusResponse:
    status = job.job_status
    unfinished = status in (JobStatus.CREATED, JobStatus.UPLOADING)
    return JobStatusResponse(
        job_id=job.id,
        status=status.value,
        created_at=job.created_at,
        updated_at=job.updated_at,
        expires_at=job.expires_at,
        upload_expires_at=job.upload_expires_at if unfinished else None,
        queued_at=job.queued_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        deletion_status=job.deletion_status,
        expiry_reason=job.expiry_reason,
        error=(
            {"code": job.error_code, "message": job.error_message}
            if status == JobStatus.FAILED
            else None
        ),
        model=(
            ModelInfo(name=job.model_name, version=job.model_version) if job.model_name else None
        ),
        result=job.result if status == JobStatus.COMPLETED else None,
        report_url=f"/v1/jobs/{job.id}/report" if status == JobStatus.COMPLETED else None,
    )


router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["service"])
def health(request: Request, response: Response) -> HealthResponse:
    """Service health: database and queue reachability."""
    rt = get_runtime(request)
    database = queue = "ok"
    try:
        with rt.store.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001 - reported as a status, not raised
        database = "unavailable"
    try:
        rt.queue.depth()
    except QueueUnavailable:
        queue = "unavailable"
    ok = database == queue == "ok"
    if not ok:
        response.status_code = 503
    return HealthResponse(status="ok" if ok else "degraded", database=database, queue=queue)


@router.get("/version", response_model=VersionResponse, tags=["service"])
def version(request: Request) -> VersionResponse:
    """App, API and model versions."""
    name, model_version = get_runtime(request).model()
    return VersionResponse(
        app="PulseCheck",
        version=__version__,
        api_version=API_VERSION,
        model=ModelInfo(name=name, version=model_version),
        report_schema_version=REPORT_SCHEMA_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
    )


@router.post(
    "/v1/jobs",
    status_code=201,
    response_model=CreateJobResponse,
    tags=["jobs"],
    responses={
        413: {"model": ErrorResponse, "description": "Declared size over the limit"},
        429: {"model": ErrorResponse, "description": "Rate limit or per-visitor job limit"},
        503: {"model": ErrorResponse, "description": "Service busy"},
    },
)
def create_job(
    request: Request, service: ServiceDep, body: CreateJobRequest | None = None
) -> CreateJobResponse:
    """Create a job. The access token in the response is shown only once."""
    rt = service.rt
    settings = rt.settings
    if body and body.size_bytes and body.size_bytes > settings.upload.max_bytes:
        mb = settings.upload.max_size_mb
        raise ApiError(413, FILE_TOO_LARGE, f"The file is larger than {mb:g} MB.")
    client = client_hash(client_ip(request, settings), settings.secret_key)
    limiter = RateLimiter(
        rt.redis, settings.capacity.job_creations_per_window, settings.capacity.rate_window_s
    )
    try:
        retry_after = limiter.hit(client)
    except QueueUnavailable:
        raise busy(settings) from None
    if retry_after is not None:
        raise ApiError(
            429,
            "rate_limited",
            "Too many jobs were created from your connection. Please try again later.",
            {"Retry-After": str(retry_after)},
        )
    try:
        job, token = service.create(client)
    except ServiceBusy:
        raise busy(settings) from None
    except TooManyActiveJobs:
        raise ApiError(
            429,
            "too_many_active_jobs",
            "You already have the maximum number of unfinished jobs. Wait for one to finish "
            "or delete it.",
            {"Retry-After": str(settings.capacity.retry_after_s)},
        ) from None
    target = rt.storage.upload_target(
        job.upload_key, job.id, job.upload_expires_at, settings.upload.max_bytes
    )
    return CreateJobResponse(
        job_id=job.id,
        access_token=token,
        status=job.status,
        created_at=job.created_at,
        expires_at=job.expires_at,
        upload=UploadInstructions(**asdict(target)),
        limits=UploadLimitsInfo(**settings.upload.to_public()),
    )


@router.put(
    "/v1/jobs/{job_id}/upload",
    response_model=JobStatusResponse,
    tags=["jobs"],
    responses={
        **ERRORS,
        413: {"model": ErrorResponse, "description": "File over the size limit"},
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {"application/octet-stream": {"schema": BINARY_SCHEMA}},
        }
    },
)
async def upload_video(request: Request, job: JobDep, service: ServiceDep) -> JobStatusResponse:
    """Stream the raw video bytes (no multipart) into private storage, with a hard size cap."""
    storage = service.storage
    if not storage.supports_direct_upload:
        raise ApiError(404, "not_found", "Not found.")
    declared = request.headers.get("content-length")
    try:
        declared_size = int(declared) if declared is not None else None
    except ValueError:
        raise ApiError(400, "bad_request", "Invalid Content-Length header.") from None
    try:
        job = await run_in_threadpool(service.begin_upload, job, declared_size)
    except JobConflict as exc:
        raise conflict(exc) from None
    except UploadRejected as exc:
        raise rejected(exc) from None
    max_bytes = service.settings.upload.max_bytes
    try:
        size = await storage.receive(job.upload_key, request.stream(), max_bytes)
    except ObjectTooLarge:
        exc = await run_in_threadpool(service.upload_too_large, job)
        raise rejected(exc) from None
    except ClientDisconnect:
        await run_in_threadpool(service.upload_interrupted, job)
        raise ApiError(400, "upload_incomplete", "The upload did not finish.") from None
    except BaseException:
        await run_in_threadpool(service.upload_interrupted, job)
        raise
    job = await run_in_threadpool(service.finish_upload, job, size)
    return job_response(job)


@router.post(
    "/v1/jobs/{job_id}/complete-upload",
    status_code=202,
    response_model=JobStatusResponse,
    tags=["jobs"],
    responses={
        **ERRORS,
        413: {"model": ErrorResponse, "description": "File over the size limit"},
        422: {"model": ErrorResponse, "description": "Unsupported, corrupt or too-long video"},
        503: {"model": ErrorResponse, "description": "Service busy; the upload is kept"},
    },
)
def complete_upload(job: JobDep, service: ServiceDep) -> JobStatusResponse:
    """Validate the uploaded video by decoding it and queue the analysis."""
    try:
        job = service.complete_upload(job)
    except JobConflict as exc:
        raise conflict(exc) from None
    except UploadRejected as exc:
        raise rejected(exc) from None
    except ServiceBusy:
        raise busy(service.settings) from None
    return job_response(job)


@router.get("/v1/jobs/{job_id}", response_model=JobStatusResponse, tags=["jobs"], responses=ERRORS)
def get_job(job: JobDep) -> JobStatusResponse:
    """Job status, and the result summary once completed. Poll this; it never blocks."""
    return job_response(job)


@router.get(
    "/v1/jobs/{job_id}/report",
    response_model=Report,
    tags=["jobs"],
    responses={**ERRORS, 410: {"model": ErrorResponse, "description": "Job expired"}},
)
def get_report(job: JobDep) -> Report:
    """The FR-10 JSON report of a completed job."""
    status = job.job_status
    if status == JobStatus.EXPIRED:
        raise ApiError(410, "job_expired", "This job has expired and its report was deleted.")
    if status == JobStatus.FAILED:
        raise ApiError(409, "report_unavailable", "The analysis failed; there is no report.")
    if status != JobStatus.COMPLETED or job.report is None:
        raise ApiError(409, "report_not_ready", f"The job is not completed (status: {status}).")
    return Report.model_validate(job.report)


@router.delete("/v1/jobs/{job_id}", status_code=204, tags=["jobs"], responses=ERRORS)
def delete_job(job: JobDep, service: ServiceDep) -> Response:
    """Delete the upload, result and report now. The job then reports status "expired"."""
    service.expire(job, ExpiryReason.DELETED_ON_REQUEST)
    return Response(status_code=204)


def create_app(runtime: Runtime | None = None) -> FastAPI:
    runtime = runtime or Runtime.from_settings(Settings.load())
    app = FastAPI(
        title="PulseCheck API",
        version=__version__,
        description=(
            "Experimental rPPG-based deepfake analysis. Results are evidence, not proof. "
            "Free and non-commercial."
        ),
        redoc_url=None,
    )
    app.state.runtime = runtime
    install_error_handlers(app)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        # Small JSON bodies only, except on the streaming upload endpoint.
        if request.method in ("POST", "PUT", "PATCH") and not request.url.path.endswith("/upload"):
            length = request.headers.get("content-length")
            chunked = "chunked" in request.headers.get("transfer-encoding", "").lower()
            if chunked:
                return error_response(411, "length_required", "Send a Content-Length header.")
            if length is not None and (not length.isdigit() or int(length) > MAX_JSON_BODY_BYTES):
                return error_response(413, "request_too_large", "The request body is too large.")
        response = await call_next(request)
        # Job responses carry tokens or private results: never cache, never leak referrers.
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    if runtime.settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(runtime.settings.cors_origins),
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["Authorization", "Content-Type"],
            allow_credentials=False,
        )
    app.include_router(router)
    return app

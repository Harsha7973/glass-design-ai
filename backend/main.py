"""FastAPI entrypoint.

Endpoints so far: GET /health, POST /process (upload + validation),
GET /status/{job_id}. The AI pipeline is wired in during later phases.
"""
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from schemas.job import HealthResponse, JobStatus
from services import image_service
from services.image_service import UploadError
from services.job_store import job_store
from utils.cleanup import cleanup_old_jobs
from utils.logging_setup import setup_logging

setup_logging(settings.log_level)
logger = logging.getLogger("glass_design_ai")

VERSION = "0.4.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_old_jobs(settings.upload_dir, settings.cleanup_after_hours)
    cleanup_old_jobs(settings.output_dir, settings.cleanup_after_hours)
    logger.info("GLASS DESIGN AI backend started (v%s)", VERSION)
    yield


app = FastAPI(title="GLASS DESIGN AI", version=VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(UploadError)
async def upload_error_handler(_: Request, exc: UploadError) -> JSONResponse:
    logger.info("Upload rejected (%d): %s", exc.status_code, exc.message)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.exception_handler(Exception)
async def unhandled_error_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error: %s", type(exc).__name__)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="glass-design-ai",
        version=VERSION,
        groq_configured=settings.groq_configured,
        groq_model=settings.effective_groq_model,
    )


@app.post("/process", response_model=JobStatus, status_code=202)
async def process(file: UploadFile = File(...)) -> JobStatus:
    """Validate and store an uploaded image, returning a job id.

    Phase 2: the image is validated and saved only. Processing starts here
    once the agents exist.
    """
    image_service.check_declared_type(file.filename, file.content_type)
    data = await image_service.read_limited(file)
    info = image_service.validate_image_bytes(data)

    job_id = uuid.uuid4().hex
    image_service.save_upload(job_id, data, info)
    job = job_store.create(job_id, info.format, info.width, info.height)
    logger.info("Job %s accepted: %s %dx%d, %d bytes", job_id, info.format, info.width, info.height, len(data))
    return job


@app.get("/status/{job_id}", response_model=JobStatus)
def status(job_id: str) -> JobStatus:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job

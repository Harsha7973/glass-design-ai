"""FastAPI entrypoint.

Endpoints: GET /health, POST /process (validate, save, start the pipeline in the
background), GET /status/{job_id} (real step messages and the result summary),
GET /result/{job_id}/{name} (the generated PNGs). PNG/SVG export arrives in Phase 9.
"""
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from agents.orchestrator import PUBLIC_FILES, Orchestrator
from config import settings
from schemas.job import HealthResponse, JobStatus
from services import image_service
from services.image_service import UploadError
from services.job_store import job_store
from utils.cleanup import cleanup_old_jobs
from utils.logging_setup import setup_logging

setup_logging(settings.log_level)
logger = logging.getLogger("glass_design_ai")

VERSION = "0.5.0"


def build_orchestrator() -> Orchestrator:
    return Orchestrator()  # tests replace this to avoid real Groq calls


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
async def process(background: BackgroundTasks, file: UploadFile = File(...)) -> JobStatus:
    """Validate and store an uploaded image, then process it in the background.

    Poll GET /status/{job_id} for the real progress."""
    image_service.check_declared_type(file.filename, file.content_type)
    data = await image_service.read_limited(file)
    info = image_service.validate_image_bytes(data)

    job_id = uuid.uuid4().hex
    image_service.save_upload(job_id, data, info)
    job = job_store.create(job_id, info.format, info.width, info.height)
    logger.info("Job %s accepted: %s %dx%d, %d bytes", job_id, info.format, info.width, info.height, len(data))
    if settings.auto_process:
        background.add_task(build_orchestrator().run, job_id)
    return job


@app.get("/status/{job_id}", response_model=JobStatus)
def status(job_id: str) -> JobStatus:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.get("/result/{job_id}/{name}")
def result_file(job_id: str, name: str) -> FileResponse:
    """Serve one generated PNG. Only known file names are allowed, and the job id must
    be a job this server created, so no other path can be reached."""
    if name not in PUBLIC_FILES or job_store.get(job_id) is None:
        raise HTTPException(status_code=404, detail="File not found.")
    path = settings.output_dir / job_id / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-store"})

"""Orchestrator: runs the agents in order for one uploaded photo and reports every real
step to the job store. No agent framework; each agent stays a plain class so this can
later be moved onto LangGraph without touching the agents.

Order: image understanding (Groq, 1 call) -> [isolate -> outline -> evaluate] with capped
retries. Export (PNG/SVG) is added in Phase 9."""
import logging
import threading
import time
from pathlib import Path

from agents.image_understanding_agent import ImageUnderstandingAgent
from agents.quality_evaluation_agent import QualityEvaluationAgent
from config import settings
from schemas.analysis import ImageAnalysis
from schemas.job import JobResult
from services.groq_service import GroqError, GroqService, get_groq_service
from services.job_store import JobStore, job_store

logger = logging.getLogger(__name__)

# Only these files can be downloaded through the API (also stops path tricks).
PUBLIC_FILES = (
    "combined_outline.png", "outer_boundary.png", "internal_design.png", "centerlines.png",
    "outline_preview.png", "crop.png", "isolated_crop.png", "debug_overlay.png",
)

_gate_lock = threading.Lock()
_gate: threading.Semaphore | None = None


def _concurrency_gate() -> threading.Semaphore:
    global _gate
    with _gate_lock:
        if _gate is None:
            _gate = threading.Semaphore(max(1, settings.max_concurrent_jobs))
        return _gate


def find_upload(job_id: str) -> Path | None:
    d = settings.upload_dir / job_id
    if not d.is_dir():
        return None
    return next((p for p in sorted(d.iterdir()) if p.stem == "original"), None)


class Orchestrator:
    def __init__(self, store: JobStore | None = None, groq: GroqService | None = None) -> None:
        self.store = store or job_store
        self._groq = groq

    def _groq_service(self) -> GroqService:
        return self._groq or get_groq_service()

    def run(self, job_id: str) -> None:
        """Never raises: any failure ends as status 'failed' with a plain message."""
        try:
            with _gate_notice(self.store, job_id):
                self._run(job_id)
        except Exception as exc:  # noqa: BLE001 - last line of defence, logged by class only
            logger.exception("Job %s crashed: %s", job_id, type(exc).__name__)
            self.store.update(job_id, status="failed", stage="done",
                              message="Processing failed unexpectedly. Try another photo or try again.",
                              error=f"Unexpected {type(exc).__name__}")

    def _run(self, job_id: str) -> None:
        store = self.store
        started = time.monotonic()
        image_path = find_upload(job_id)
        if image_path is None:
            store.update(job_id, status="failed", stage="done", message="The uploaded photo could not be found.",
                         error="upload missing")
            return
        store.update(job_id, status="processing", stage="analyzing",
                     message="Looking at the photo to find the main design")

        # 1) Image understanding. If Groq is unavailable the pipeline still runs, without a location hint.
        analysis, ai_used = None, False
        try:
            analysis = ImageUnderstandingAgent(self._groq).run(job_id, image_path)
            ai_used = True
            store.update(job_id, message=f"AI found: {analysis.primary_object} (confidence {analysis.confidence:.0%})")
        except GroqError as exc:
            logger.warning("Job %s: Groq analysis unavailable: %s", job_id, type(exc).__name__)
            store.update(job_id, warning=f"AI analysis unavailable ({exc}); the whole photo was searched instead.")
            store.update(job_id, message="AI analysis unavailable; searching the whole photo for the design")
            analysis = ImageAnalysis.model_validate({"primary_object": "unknown", "confidence": 0.0})

        # 2) isolate -> outline -> evaluate, with capped retries and a time limit
        quality = QualityEvaluationAgent(self._groq_service() if settings.groq_quality_opinion and ai_used else None)
        limit = settings.processing_timeout_seconds
        refinement = quality.refine(
            job_id, image_path, analysis,
            use_groq=settings.groq_quality_opinion and ai_used,
            on_progress=lambda stage, msg: store.update(job_id, stage=stage, message=msg),
            should_stop=lambda: time.monotonic() - started > limit,
        )

        # 3) publish the result
        out_dir = settings.output_dir / job_id
        files = {n: f"/result/{job_id}/{n}" for n in PUBLIC_FILES if (out_dir / n).is_file()}
        best = refinement.attempts[refinement.best_attempt - 1].report
        decision = refinement.decision
        result = JobResult(
            decision="accept" if decision == "accept" else ("needs_review" if decision == "needs_review" else "failed"),
            score=best.score, attempts=len(refinement.attempts), best_attempt=refinement.best_attempt,
            summary=refinement.message, problems=best.problems, primary_object=analysis.primary_object,
            ai_analysis_used=ai_used, files=files if decision != "failed" else {k: v for k, v in files.items() if k == "debug_overlay.png"},
        )
        status = {"accept": "completed", "needs_review": "needs_review"}.get(decision, "failed")
        store.update(job_id, status=status, stage="done", message=refinement.message, result=result,
                     error=None if status != "failed" else "no usable outline was produced")


class _gate_notice:
    """Wait for a free processing slot, telling the job it is queued if it has to wait."""

    def __init__(self, store: JobStore, job_id: str) -> None:
        self.store, self.job_id, self.gate = store, job_id, _concurrency_gate()

    def __enter__(self):
        if not self.gate.acquire(blocking=False):
            self.store.update(self.job_id, status="processing", stage="queued",
                              message="Waiting for another photo to finish processing")
            self.gate.acquire()
        return self

    def __exit__(self, *exc):
        self.gate.release()
        return False

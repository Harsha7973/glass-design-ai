"""Agent 1: Image Understanding.

Sends the photo to Groq once, validates the structured answer and saves it as
analysis.json in the job's output folder for the later agents.
"""
import logging
from pathlib import Path

from config import settings
from schemas.analysis import ExtractionInstructions, ImageAnalysis, PrimaryObject
from services.groq_service import GroqService, get_groq_service

logger = logging.getLogger(__name__)


class ImageUnderstandingAgent:
    name = "image_understanding"

    def __init__(self, groq: GroqService | None = None) -> None:
        self._groq = groq

    @property
    def groq(self) -> GroqService:
        return self._groq or get_groq_service()

    def run(self, job_id: str, image_path: Path) -> ImageAnalysis:
        """Analyse the photo. Raises GroqError subclasses on failure; the
        orchestrator (Phase 8) decides how to report them."""
        analysis = self.groq.analyze_image(image_path)
        logger.info(
            "Job %s: primary object %r (confidence %.2f, category %s)",
            job_id, analysis.primary_object, analysis.confidence, analysis.object_category,
        )
        if analysis.bounding_box is None:
            logger.warning("Job %s: no usable bounding box; later stages must locate the object themselves", job_id)
        self._save(job_id, analysis)
        return analysis

    def primary_object(self, analysis: ImageAnalysis) -> PrimaryObject:
        return self.groq.identify_primary_object(analysis)

    def instructions(self, analysis: ImageAnalysis) -> ExtractionInstructions:
        return self.groq.generate_extraction_instructions(analysis)

    @staticmethod
    def _save(job_id: str, analysis: ImageAnalysis) -> Path:
        out_dir = settings.output_dir / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "analysis.json"
        path.write_text(analysis.model_dump_json(indent=2), encoding="utf-8")
        return path

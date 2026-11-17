"""Agent 4: Quality Evaluation and the capped retry loop.

Scores an outline with deterministic measurements against the photo (optionally
adding Groq's visual opinion, one call per attempt), and, if the result is weak,
re-runs segmentation + outline with adjusted settings. The number of attempts is
capped; the best attempt is kept; a weak result is reported as such, never hidden."""
import logging
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from agents.outline_extraction_agent import OutlineExtractionAgent
from agents.segmentation_agent import SegmentationAgent
from config import settings
from schemas.analysis import ImageAnalysis
from schemas.outline import OutlineParams, OutlineResult
from schemas.quality import (
    ACCEPT_SCORE,
    MIN_USABLE_SCORE,
    AttemptRecord,
    QualityReport,
    RefinementResult,
)
from schemas.segmentation import SegmentationParams
from services import quality_service as qs
from services.groq_service import GroqError, GroqService

logger = logging.getLogger(__name__)
GROQ_WEIGHT = 0.3  # Groq's opinion is advisory: local measurements carry 70%


def _load(path: Path, mode: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert(mode))


class QualityEvaluationAgent:
    name = "quality_evaluation"

    def __init__(self, groq: GroqService | None = None) -> None:
        self._groq = groq

    def evaluate(self, job_id: str, outline: OutlineResult, use_groq: bool = False) -> QualityReport:
        d = settings.output_dir / job_id
        if not outline.success:
            return QualityReport(score=0.0, local_score=0.0, decision="failed",
                                 problems=["no reliable line structure was found"] + list(outline.warnings))
        crop = _load(d / "crop.png", "RGB")
        lines = 255 - _load(d / "centerlines.png", "L")
        outer = 255 - _load(d / "outer_boundary.png", "L")
        metrics = qs.measure(crop, lines, outer)
        local = qs.local_score(metrics)
        problems = qs.diagnose(metrics)

        score, groq_eval, status = local, None, "not_used"
        if use_groq and self._groq is not None:
            try:
                groq_eval = self._groq.evaluate_extraction_result(d / "crop.png", d / "combined_outline.png")
                score = round((1 - GROQ_WEIGHT) * local + GROQ_WEIGHT * groq_eval.quality_score, 4)
                status = "used"
                problems += [f"Groq: {i}" for i in groq_eval.issues[:3]]
            except GroqError as exc:
                status = f"unavailable: {type(exc).__name__}"  # advisory only: carry on without it
        decision = "accept" if score >= ACCEPT_SCORE else "retry"
        return QualityReport(score=score, local_score=local, decision=decision, metrics=metrics,
                             problems=problems, groq=groq_eval, groq_status=status)

    def refine(
        self,
        job_id: str,
        image_path: Path,
        analysis: ImageAnalysis,
        outline_params: OutlineParams | None = None,
        seg_params: SegmentationParams | None = None,
        max_attempts: int | None = None,
        use_groq: bool = False,
        on_progress: Callable[[str, str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> RefinementResult:
        cap = max(1, max_attempts if max_attempts is not None else settings.max_retries + 1)
        op = outline_params or OutlineParams()
        sp = seg_params or SegmentationParams()
        seg_agent, out_agent = SegmentationAgent(), OutlineExtractionAgent()
        records: list[AttemptRecord] = []
        notes: list[str] = []

        say = on_progress or (lambda stage, msg: None)
        for n in range(1, cap + 1):
            if n > 1 and should_stop is not None and should_stop():
                notes.append("stopped early: the time limit was reached")
                break
            sub = f"{job_id}/attempt{n}"
            say("segmenting", f"Attempt {n} of {cap}: isolating the design from the background")
            seg = seg_agent.run(sub, image_path, analysis, sp)
            if seg.success:
                mask = _load(settings.output_dir / sub / "mask.png", "L")
                say("extracting", f"Attempt {n} of {cap}: extracting the outer boundary and inner lines")
                outline = out_agent.run(sub, image_path, mask, analysis, op)
                say("evaluating", f"Attempt {n} of {cap}: checking the outline quality")
                report = self.evaluate(sub, outline, use_groq)
            else:
                report = QualityReport(score=0.0, local_score=0.0, decision="failed",
                                       problems=["the design could not be isolated"] + list(seg.warnings))
            records.append(AttemptRecord(attempt=n, outline_params=op, segmentation_params=sp, report=report))
            logger.info("Attempt %d/%d score %.2f (%s)", n, cap, report.score, report.decision)
            say("evaluating", f"Attempt {n} of {cap} scored {report.score:.2f}" + (" (good enough)" if report.decision == "accept" else ""))
            if report.decision == "accept":
                break
            if n < cap:
                unwanted = bool(report.groq and report.groq.unwanted_elements_present)
                op, sp, note = qs.next_params(op, sp, report.metrics, unwanted)
                notes.append(f"attempt {n + 1}: {note}")

        best = max(records, key=lambda r: r.report.score)
        files = self._publish_best(job_id, best.attempt)
        if best.report.score >= ACCEPT_SCORE:
            decision, msg = "accept", f"Accepted after {len(records)} attempt(s), score {best.report.score:.2f}."
        elif best.report.score >= MIN_USABLE_SCORE:
            decision = "needs_review"
            msg = (f"Best of {len(records)} attempt(s) scored {best.report.score:.2f}, below the "
                   f"{ACCEPT_SCORE:.2f} target. Usable as a draft; check it before relying on it.")
        else:
            decision = "failed"
            msg = f"No attempt reached a usable result (best score {best.report.score:.2f})."
        result = RefinementResult(decision=decision, best_attempt=best.attempt, attempts=records,
                                  max_attempts=cap, message=msg + (" Adjustments: " + "; ".join(notes) if notes else ""),
                                  files=files)
        (settings.output_dir / job_id / "refinement.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result

    @staticmethod
    def _publish_best(job_id: str, attempt: int) -> dict[str, str]:
        """Copy the winning attempt's files up to the job folder so later phases find them there."""
        src = settings.output_dir / job_id / f"attempt{attempt}"
        dst = settings.output_dir / job_id
        files = {}
        for f in src.iterdir():
            if f.is_file():
                shutil.copy2(f, dst / f.name)
                files[f.name] = f.name
        return files

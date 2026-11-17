"""Quality agent tests: synthetic scenes plus a fake Groq. They prove the scoring and the
capped retry logic behave as designed, not that scores match human judgement on real photos."""
import json

import numpy as np
import pytest
from PIL import Image

import agents.quality_evaluation_agent as qa
from agents.quality_evaluation_agent import QualityEvaluationAgent
from config import settings
from schemas.analysis import GroqEvaluation, ImageAnalysis
from schemas.outline import OutlineParams
from schemas.quality import ACCEPT_SCORE, QualityMetrics
from schemas.segmentation import SegmentationParams
from services import quality_service as qs
from services.groq_service import GroqTimeoutError
from synth import etched_scene

BOX = (0.35, 0.0, 0.55, 0.85)


def analysis(box=BOX):
    return ImageAnalysis.model_validate({"primary_object": "x", "confidence": 0.9, "bounding_box": box})


@pytest.fixture(autouse=True)
def out_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "output_dir", tmp_path / "out")
    monkeypatch.setattr(settings, "segmentation_backend", "classical")


@pytest.fixture()
def photo(tmp_path):
    p = tmp_path / "s.png"
    Image.fromarray(etched_scene()[0]).save(p)
    return p


@pytest.fixture()
def plain(tmp_path):
    p = tmp_path / "plain.png"
    Image.fromarray(etched_scene(with_person=False, with_ornament=False)[0]).save(p)
    return p


class FakeGroq:
    def __init__(self, score=0.9, unwanted=False, error=None):
        self.score, self.unwanted, self.error, self.calls = score, unwanted, error, 0

    def evaluate_extraction_result(self, original, outline):
        self.calls += 1
        if self.error:
            raise self.error
        return GroqEvaluation(quality_score=self.score, unwanted_elements_present=self.unwanted, issues=["x"])


def test_clean_synthetic_design_is_accepted_first_try(photo):
    r = QualityEvaluationAgent().refine("a", photo, analysis())
    assert r.decision == "accept" and len(r.attempts) == 1 and r.attempts[0].report.score >= ACCEPT_SCORE
    d = settings.output_dir / "a"
    assert (d / "combined_outline.png").is_file() and (d / "refinement.json").is_file()


def test_failure_is_capped_and_reported(plain):
    r = QualityEvaluationAgent().refine("b", plain, analysis((0.3, 0.1, 0.6, 0.8)), max_attempts=3)
    assert r.decision == "failed" and len(r.attempts) == 3 and r.max_attempts == 3
    assert all(a.report.score == 0.0 for a in r.attempts)


def test_retry_loop_never_exceeds_cap_and_keeps_best(photo, monkeypatch):
    monkeypatch.setattr(qa, "ACCEPT_SCORE", 1.1)  # nothing is ever good enough
    r = QualityEvaluationAgent().refine("c", photo, analysis(), max_attempts=2)
    assert len(r.attempts) == 2
    assert r.attempts[r.best_attempt - 1].report.score == max(a.report.score for a in r.attempts)
    assert r.decision in ("needs_review", "failed") and "below" in r.message or r.decision == "failed"


def test_groq_opinion_is_blended_and_called_once_per_attempt(photo):
    g = FakeGroq(score=0.2)
    r = QualityEvaluationAgent(g).refine("d", photo, analysis(), max_attempts=1, use_groq=True)
    rep = r.attempts[0].report
    assert g.calls == 1 and rep.groq_status == "used"
    assert rep.score == pytest.approx(0.7 * rep.local_score + 0.3 * 0.2, abs=1e-3)


def test_groq_failure_is_non_fatal(photo):
    g = FakeGroq(error=GroqTimeoutError("slow"))
    r = QualityEvaluationAgent(g).refine("e", photo, analysis(), max_attempts=1, use_groq=True)
    rep = r.attempts[0].report
    assert rep.groq_status.startswith("unavailable") and rep.score == rep.local_score


def test_groq_not_called_unless_requested(photo):
    g = FakeGroq()
    QualityEvaluationAgent(g).refine("f", photo, analysis(), max_attempts=1)
    assert g.calls == 0


def test_next_params_noisy_vs_sparse():
    op, sp = OutlineParams(sensitivity="medium"), SegmentationParams()
    noisy = QualityMetrics(contrast=0.5, recall=0.9, boundary_fit=0.9, continuity=0.9, clutter=0.9, line_density=0.1)
    o2, s2, _ = qs.next_params(op, sp, noisy)
    assert o2.sensitivity == "low" and s2.density_scale > sp.density_scale
    sparse = QualityMetrics(contrast=0.95, recall=0.1, boundary_fit=0.9, continuity=0.9, clutter=0.9, line_density=0.01)
    o3, _, _ = qs.next_params(op, sp, sparse)
    assert o3.bright_lines_only is False  # first remedy for missing detail
    o4, _, _ = qs.next_params(o3, sp, sparse)
    assert o4.sensitivity == "high"


def test_next_params_stay_inside_limits():
    op, sp = OutlineParams(sensitivity="low"), SegmentationParams(density_scale=3.0, grow_reach=0.0)
    bad = QualityMetrics(contrast=0.1, recall=0.1, boundary_fit=0.1, continuity=0.1, clutter=0.0, line_density=0.5)
    o2, s2, _ = qs.next_params(op, sp, bad, groq_unwanted=True)
    OutlineParams.model_validate(o2.model_dump()); SegmentationParams.model_validate(s2.model_dump())
    assert o2.sensitivity == "low" and s2.density_scale <= 3.0


def test_refinement_json_is_valid(photo):
    QualityEvaluationAgent().refine("g", photo, analysis(), max_attempts=1)
    data = json.loads((settings.output_dir / "g" / "refinement.json").read_text())
    assert data["decision"] and data["attempts"][0]["report"]["metrics"]["clutter"] >= 0

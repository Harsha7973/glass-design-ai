"""End-to-end pipeline tests through the API, with a fake Groq (no network).
Synthetic scenes only: they prove the wiring, status reporting and failure handling."""
import io

import pytest
from PIL import Image

import agents.orchestrator as orch
import main
from agents.orchestrator import Orchestrator
from config import settings
from schemas.analysis import ImageAnalysis
from services.groq_service import GroqConfigError
from synth import etched_scene


class FakeGroq:
    def __init__(self, box=(0.35, 0.0, 0.55, 0.85), error=None):
        self.box, self.error, self.calls = box, error, 0

    def analyze_image(self, path):
        self.calls += 1
        if self.error:
            raise self.error
        return ImageAnalysis.model_validate(
            {"primary_object": "etched ornament", "confidence": 0.9, "bounding_box": self.box})


def png(rgb) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def run(client, monkeypatch):
    monkeypatch.setattr(settings, "auto_process", True)
    monkeypatch.setattr(settings, "segmentation_backend", "classical")

    def go(rgb, groq=None):
        groq = groq or FakeGroq()
        monkeypatch.setattr(main, "build_orchestrator", lambda: Orchestrator(groq=groq))
        r = client.post("/process", files={"file": ("p.png", png(rgb), "image/png")})
        assert r.status_code == 202 and r.json()["status"] == "uploaded"
        job = client.get(f"/status/{r.json()['job_id']}").json()  # background task has finished
        return job, groq

    return go


def test_full_pipeline_completes_and_serves_results(client, run):
    job, groq = run(etched_scene()[0])
    assert job["status"] == "completed" and job["stage"] == "done" and job["error"] is None
    res = job["result"]
    assert res["decision"] == "accept" and res["score"] >= 0.8 and res["ai_analysis_used"] is True
    assert res["primary_object"] == "etched ornament" and groq.calls == 1
    assert "combined_outline.png" in res["files"]
    r = client.get(f"/result/{job['job_id']}/combined_outline.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(r.content)).format == "PNG"


def test_status_reports_real_steps_in_order(run):
    job, _ = run(etched_scene()[0])
    stages = [e["stage"] for e in job["events"]]
    assert stages[0] == "uploaded" and stages[-1] == "done"
    for s in ("analyzing", "segmenting", "extracting", "evaluating"):
        assert s in stages, s
    assert stages.index("analyzing") < stages.index("segmenting") < stages.index("extracting")
    times = [e["at"] for e in job["events"]]
    assert times == sorted(times)


def test_groq_unavailable_still_runs_with_warning(run):
    job, _ = run(etched_scene()[0], FakeGroq(error=GroqConfigError("no key")))
    assert job["status"] in ("completed", "needs_review")
    assert job["result"]["ai_analysis_used"] is False
    assert any("AI analysis unavailable" in w for w in job["warnings"])
    assert job["result"]["files"]


def test_plain_frost_fails_honestly_without_outline_files(run):
    plain = etched_scene(with_person=False, with_ornament=False)[0]
    job, _ = run(plain, FakeGroq(box=(0.3, 0.1, 0.6, 0.8)))
    assert job["status"] == "failed" and job["error"]
    assert job["result"]["decision"] == "failed"
    assert "combined_outline.png" not in job["result"]["files"]
    assert len(job["result"]["summary"]) > 10


def test_time_limit_stops_retries(run, monkeypatch):
    monkeypatch.setattr(settings, "processing_timeout_seconds", 0)
    plain = etched_scene(with_person=False, with_ornament=False)[0]
    job, _ = run(plain, FakeGroq(box=(0.3, 0.1, 0.6, 0.8)))
    assert job["result"]["attempts"] == 1
    assert "time limit" in job["result"]["summary"]


def test_unexpected_crash_becomes_failed_status(client, monkeypatch):
    monkeypatch.setattr(settings, "auto_process", True)

    def boom(self, *a, **k):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(orch.QualityEvaluationAgent, "refine", boom)
    monkeypatch.setattr(main, "build_orchestrator", lambda: Orchestrator(groq=FakeGroq()))
    r = client.post("/process", files={"file": ("p.png", png(etched_scene()[0]), "image/png")})
    job = client.get(f"/status/{r.json()['job_id']}").json()
    assert job["status"] == "failed" and "unexpected" in job["message"].lower()
    assert "secret internal detail" not in str(job)


def test_missing_upload_is_reported(client):
    from services.job_store import job_store
    job_store.create("ghost", "PNG", 10, 10)
    Orchestrator().run("ghost")
    assert job_store.get("ghost").status == "failed"


@pytest.mark.parametrize("name", ["original.png", "../../config.py", "mask.png", "analysis.json", "..%2Fx.png"])
def test_result_route_only_serves_whitelisted_files(client, run, name):
    job, _ = run(etched_scene()[0])
    assert client.get(f"/result/{job['job_id']}/{name}").status_code == 404


def test_result_route_unknown_job(client):
    assert client.get("/result/nope/combined_outline.png").status_code == 404


def test_auto_process_off_leaves_job_uploaded(client, monkeypatch):
    monkeypatch.setattr(settings, "auto_process", False)
    r = client.post("/process", files={"file": ("p.png", png(etched_scene()[0]), "image/png")})
    assert client.get(f"/status/{r.json()['job_id']}").json()["status"] == "uploaded"

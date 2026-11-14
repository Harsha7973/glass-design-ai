"""Groq integration tests. All HTTP is mocked with httpx.MockTransport, so these
prove our request building, parsing, error handling and caching, but NOT that
the real Groq API accepts the request. Use backend/scripts/check_groq.py for that."""
import json
import logging

import httpx
import pytest
from PIL import Image

from agents.image_understanding_agent import ImageUnderstandingAgent
from config import settings
from schemas.analysis import ImageAnalysis, normalize_bbox
from services.groq_service import (
    GroqAPIError,
    GroqAuthError,
    GroqConfigError,
    GroqInvalidResponseError,
    GroqModelUnavailableError,
    GroqRateLimitError,
    GroqService,
    GroqTimeoutError,
    extract_json,
)

SECRET = "gsk_SECRET_TEST_KEY_123"

GOOD = {
    "primary_object": "Etched glass floral ornament",
    "object_category": "glass",
    "confidence": 0.9,
    "location": "right",
    "bounding_box": [0.5, 0.02, 0.9, 0.9],
    "candidate_designs": [],
    "unwanted_objects": ["person", "door frame"],
    "is_transparent_or_reflective": True,
    "background_description": "grey frosted glass",
    "segmentation_required": True,
    "outer_boundary_required": True,
    "internal_details_required": True,
    "extraction_hints": {"edge_sensitivity": "high", "contrast_enhance": True, "line_style": "thin", "notes": "faint"},
}


def chat_reply(content, finish="stop"):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": finish}]})


class Recorder:
    """MockTransport handler that records requests and replays queued responses."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request):
        self.requests.append(request)
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item


def make_service(*responses, **kw):
    rec = Recorder(*responses)
    sleeps: list[float] = []
    svc = GroqService(
        api_key=kw.pop("api_key", SECRET),
        model="test-model",
        max_retries=kw.pop("max_retries", 1),
        transport=httpx.MockTransport(rec),
        sleep=sleeps.append,
        **kw,
    )
    return svc, rec, sleeps


@pytest.fixture()
def photo(tmp_path):
    p = tmp_path / "photo.jpg"
    Image.new("RGB", (2000, 1500), (120, 130, 140)).save(p, "JPEG")
    return p


# ------------------------------------------------------------ request/parse
def test_analyze_success_and_request_shape(photo):
    svc, rec, _ = make_service(chat_reply(json.dumps(GOOD)))
    result = svc.analyze_image(photo)
    assert result.primary_object.startswith("Etched glass")
    assert result.bounding_box == (0.5, 0.02, 0.9, 0.9)

    req = rec.requests[0]
    assert req.url.path.endswith("/chat/completions")
    assert req.headers["authorization"] == f"Bearer {SECRET}"
    body = json.loads(req.content)
    assert body["model"] == "test-model"
    assert body["response_format"] == {"type": "json_object"}
    parts = body["messages"][1]["content"]
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "JSON" in body["messages"][0]["content"]  # JSON mode needs the word JSON


def test_large_image_is_downscaled_for_groq(photo):
    from services import image_service

    data, mime = image_service.prepare_for_vision(photo, max_side=1024)
    assert mime == "image/jpeg"
    img = Image.open(__import__("io").BytesIO(data))
    assert max(img.size) == 1024 and img.size == (1024, 768)  # aspect ratio kept


def test_tolerates_fences_and_think_blocks(photo):
    wrapped = "<think>hmm</think>\n```json\n" + json.dumps(GOOD) + "\n```"
    svc, _, _ = make_service(chat_reply(wrapped))
    assert svc.analyze_image(photo).confidence == 0.9


def test_extract_json_variants():
    assert extract_json('noise {"a": 1} tail') == {"a": 1}
    for bad in ["no json here", "{broken", "[1, 2]", ""]:
        with pytest.raises(GroqInvalidResponseError):
            extract_json(bad)


def test_invalid_reply_raises(photo):
    svc, _, _ = make_service(chat_reply("sorry, I cannot help"))
    with pytest.raises(GroqInvalidResponseError):
        svc.analyze_image(photo)


def test_missing_required_field_raises(photo):
    svc, _, _ = make_service(chat_reply(json.dumps({"confidence": 0.5})))
    with pytest.raises(GroqInvalidResponseError, match="primary_object"):
        svc.analyze_image(photo)


def test_truncated_reply_raises(photo):
    svc, _, _ = make_service(chat_reply(json.dumps(GOOD)[:20], finish="length"))
    with pytest.raises(GroqInvalidResponseError, match="cut off"):
        svc.analyze_image(photo)


def test_unexpected_response_shape_raises(photo):
    svc, _, _ = make_service(httpx.Response(200, json={"nope": 1}))
    with pytest.raises(GroqInvalidResponseError):
        svc.analyze_image(photo)


# ------------------------------------------------------------------ errors
def test_missing_key_raises_before_any_request(photo):
    svc, rec, _ = make_service(chat_reply("{}"), api_key="")
    with pytest.raises(GroqConfigError, match="GROQ_API_KEY"):
        svc.analyze_image(photo)
    assert rec.requests == []


@pytest.mark.parametrize("status", [401, 403])
def test_auth_error_no_retry(photo, status):
    svc, rec, _ = make_service(httpx.Response(status, json={"error": {"message": "bad key"}}))
    with pytest.raises(GroqAuthError):
        svc.analyze_image(photo)
    assert len(rec.requests) == 1


def test_model_unavailable(photo):
    svc, rec, _ = make_service(httpx.Response(404, json={"error": {"code": "model_not_found", "message": "x"}}))
    with pytest.raises(GroqModelUnavailableError, match="test-model"):
        svc.analyze_image(photo)
    assert len(rec.requests) == 1


def test_model_decommissioned_400(photo):
    svc, _, _ = make_service(httpx.Response(400, json={"error": {"code": "model_decommissioned", "message": "x"}}))
    with pytest.raises(GroqModelUnavailableError):
        svc.analyze_image(photo)


def test_rate_limit_retries_then_succeeds(photo):
    svc, rec, sleeps = make_service(
        httpx.Response(429, headers={"retry-after": "3"}, json={"error": {"message": "slow down"}}),
        chat_reply(json.dumps(GOOD)),
    )
    assert svc.analyze_image(photo).confidence == 0.9
    assert len(rec.requests) == 2
    assert sleeps == [3.0]


def test_rate_limit_gives_up_after_limit(photo):
    svc, rec, _ = make_service(httpx.Response(429, json={"error": {"message": "x"}}))
    with pytest.raises(GroqRateLimitError):
        svc.analyze_image(photo)
    assert len(rec.requests) == 2  # 1 try + 1 retry, never infinite


def test_retry_after_is_capped(photo):
    svc, _, sleeps = make_service(
        httpx.Response(429, headers={"retry-after": "9999"}, json={}), chat_reply(json.dumps(GOOD))
    )
    svc.analyze_image(photo)
    assert sleeps == [10.0]


def test_timeout_retries_then_raises(photo):
    svc, rec, _ = make_service(httpx.ReadTimeout("slow"))
    with pytest.raises(GroqTimeoutError):
        svc.analyze_image(photo)
    assert len(rec.requests) == 2


def test_server_error_retries_then_raises(photo):
    svc, rec, _ = make_service(httpx.Response(503, json={}))
    with pytest.raises(GroqAPIError, match="503"):
        svc.analyze_image(photo)
    assert len(rec.requests) == 2


def test_network_error_message_is_friendly(photo):
    svc, _, _ = make_service(httpx.ConnectError("dns"), max_retries=0)
    with pytest.raises(GroqAPIError, match=r"ConnectError.*internet"):
        svc.analyze_image(photo)


def test_certificate_error_gets_hint(photo):
    svc, _, _ = make_service(httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED] self-signed"), max_retries=0)
    with pytest.raises(GroqAPIError, match="intercepting HTTPS"):
        svc.analyze_image(photo)


# ------------------------------------------------------- caching / secrets
def test_same_image_is_cached(photo):
    svc, rec, _ = make_service(chat_reply(json.dumps(GOOD)))
    a = svc.analyze_image(photo)
    b = svc.analyze_image(photo)
    assert a == b and len(rec.requests) == 1 and svc.api_calls == 1


def test_different_image_is_not_cached(photo, tmp_path):
    other = tmp_path / "other.jpg"
    Image.new("RGB", (300, 300), (5, 5, 5)).save(other, "JPEG")
    svc, rec, _ = make_service(chat_reply(json.dumps(GOOD)))
    svc.analyze_image(photo)
    svc.analyze_image(other)
    assert len(rec.requests) == 2


def test_cached_result_is_a_copy(photo):
    svc, _, _ = make_service(chat_reply(json.dumps(GOOD)))
    a = svc.analyze_image(photo)
    a.primary_object = "mutated"
    assert svc.analyze_image(photo).primary_object != "mutated"


def test_key_never_logged_or_in_errors(photo, caplog):
    caplog.set_level(logging.DEBUG)
    svc, _, _ = make_service(httpx.Response(401, json={"error": {"message": f"key {SECRET} bad"}}))
    with pytest.raises(GroqAuthError) as info:
        svc.analyze_image(photo)
    assert SECRET not in str(info.value)
    assert SECRET not in caplog.text
    assert SECRET not in repr(svc)


def test_error_body_is_truncated(photo):
    svc, _, _ = make_service(httpx.Response(422, json={"error": {"message": "x" * 1000}}))
    with pytest.raises(GroqAPIError) as info:
        svc.analyze_image(photo)
    assert len(str(info.value)) < 300


# ----------------------------------------------------------- schema logic
def test_bbox_normalisation():
    assert normalize_bbox([0.1, 0.2, 0.5, 0.9]) == (0.1, 0.2, 0.5, 0.9)
    assert normalize_bbox([100, 200, 500, 900]) == (0.1, 0.2, 0.5, 0.9)  # 0-1000 grid
    assert normalize_bbox([0.5, 0.5, 0.5, 0.5]) is None  # degenerate
    assert normalize_bbox([5000, 0, 6000, 10]) is None  # nonsense
    assert normalize_bbox([-1, 0, 0.5, 0.5]) is None
    assert normalize_bbox("junk") is None and normalize_bbox(None) is None


def test_schema_normalises_odd_values():
    a = ImageAnalysis.model_validate(
        {"primary_object": " Carving ", "confidence": 85, "object_category": "Marble statue",
         "location": "  RIGHT ", "bounding_box": [100, 100, 600, 800], "unwanted_objects": None}
    )
    assert a.primary_object == "Carving" and a.confidence == 0.85
    assert a.object_category == "other" and a.location == "right"
    assert a.bounding_box == (0.1, 0.1, 0.6, 0.8) and a.unwanted_objects == []


def test_derived_functions_make_no_api_calls(photo):
    svc, rec, _ = make_service(chat_reply(json.dumps(GOOD)))
    analysis = svc.analyze_image(photo)
    primary = svc.identify_primary_object(analysis)
    inst = svc.generate_extraction_instructions(analysis)
    assert len(rec.requests) == 1
    assert primary.category == "glass" and not primary.multiple_candidates
    assert inst.region_of_interest == analysis.bounding_box
    assert inst.transparent_material and inst.edge_sensitivity == "high"
    assert "person" in inst.objects_to_suppress


def test_glass_never_gets_low_sensitivity(photo):
    data = {**GOOD, "extraction_hints": {**GOOD["extraction_hints"], "edge_sensitivity": "low", "contrast_enhance": False}}
    svc, _, _ = make_service(chat_reply(json.dumps(data)))
    inst = svc.generate_extraction_instructions(svc.analyze_image(photo))
    assert inst.edge_sensitivity == "medium" and inst.contrast_enhance


# --------------------------------------------------------------- evaluation
def test_evaluate_sends_two_images(photo, tmp_path):
    outline = tmp_path / "outline.png"
    Image.new("L", (500, 400), 255).save(outline)
    reply = {"quality_score": 80, "boundary_complete": True, "issues": ["noise"], "recommendations": ["blur more"]}
    svc, rec, _ = make_service(chat_reply(json.dumps(reply)))
    ev = svc.evaluate_extraction_result(photo, outline)
    assert ev.quality_score == 0.8 and ev.boundary_complete and ev.issues == ["noise"]
    parts = json.loads(rec.requests[0].content)["messages"][1]["content"]
    uris = [p["image_url"]["url"] for p in parts if p["type"] == "image_url"]
    assert len(uris) == 2 and uris[1].startswith("data:image/png;base64,")


# -------------------------------------------------------------------- agent
def test_agent_saves_analysis_json(photo, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "output_dir", tmp_path / "out")
    svc, _, _ = make_service(chat_reply(json.dumps(GOOD)))
    agent = ImageUnderstandingAgent(groq=svc)
    analysis = agent.run("job123", photo)
    saved = json.loads((tmp_path / "out" / "job123" / "analysis.json").read_text())
    assert saved["primary_object"] == analysis.primary_object
    assert agent.primary_object(analysis).name == analysis.primary_object
    assert agent.instructions(analysis).transparent_material


def test_exif_rotation_is_applied(tmp_path):
    from services import image_service

    p = tmp_path / "rot.jpg"
    img = Image.new("RGB", (200, 100), (1, 2, 3))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise when displayed
    img.save(p, "JPEG", exif=exif)
    assert image_service.load_oriented_image(p).size == (100, 200)


# ------------------------------------------------------------------- health
def test_health_reports_groq_status_without_key(client, monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", SECRET)
    monkeypatch.setattr(settings, "groq_model", "")
    body = client.get("/health").json()
    assert body["groq_configured"] is True
    assert body["groq_model"] == "qwen/qwen3.8-27b"
    assert SECRET not in json.dumps(body)


# ----------------------------------------------------------- TLS trust setup
def test_verify_uses_system_cert_store_by_default():
    import ssl

    import truststore

    verify = GroqService._build_verify()
    assert isinstance(verify, truststore.SSLContext) and isinstance(verify, ssl.SSLContext)


def test_verify_falls_back_when_system_certs_off(monkeypatch):
    monkeypatch.setattr(settings, "groq_use_system_certs", False)
    assert GroqService._build_verify() is True  # normal verification, never disabled


def test_ca_bundle_missing_file_is_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "groq_ca_bundle", str(tmp_path / "nope.pem"))
    with pytest.raises(GroqConfigError, match="GROQ_CA_BUNDLE"):
        GroqService._build_verify()


def test_ca_bundle_file_is_loaded(monkeypatch, tmp_path):
    import ssl
    import subprocess

    pem = tmp_path / "ca.pem"
    try:
        import certifi

        pem.write_bytes(open(certifi.where(), "rb").read())
    except ImportError:  # certifi ships with httpx; skip if absent
        pytest.skip("certifi not available")
    monkeypatch.setattr(settings, "groq_ca_bundle", f'"{pem}"')  # surrounding quotes tolerated
    assert isinstance(GroqService._build_verify(), ssl.SSLContext)

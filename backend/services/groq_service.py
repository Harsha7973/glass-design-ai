"""Groq client (OpenAI-compatible chat completions API) for image understanding.

Design choices
- Talks to Groq with httpx directly; no extra SDK to keep compatible on Windows.
- ONE vision call per image (analyze_image). identify_primary_object() and
  generate_extraction_instructions() are derived locally from that result, so
  the free tier is not spent on three calls per photo.
- Results are cached by (image bytes, model, prompt version).
- Retries are limited (settings.groq_max_retries) and only for 429/5xx/timeouts.
- The API key is only ever placed in the Authorization header. It is never
  logged, never put in exception text and never included in repr().
"""
import base64
import hashlib
import json
import logging
import re
import ssl
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import httpx
from pydantic import BaseModel, ValidationError

from config import settings
from schemas.analysis import (
    ExtractionInstructions,
    GroqEvaluation,
    ImageAnalysis,
    PrimaryObject,
)
from services import image_service
from services.groq_prompts import (
    ANALYZE_SYSTEM,
    ANALYZE_USER,
    EVALUATE_SYSTEM,
    EVALUATE_USER,
    PROMPT_VERSION,
)

logger = logging.getLogger(__name__)

MAX_BACKOFF_SECONDS = 10.0
CACHE_SIZE = 64


# ---------------------------------------------------------------- errors
class GroqError(Exception):
    """Base class. `str(exc)` is always safe to show to a user."""


class GroqConfigError(GroqError):
    """API key missing or otherwise not configured."""


class GroqAuthError(GroqError):
    """Key rejected (HTTP 401/403)."""


class GroqRateLimitError(GroqError):
    """Rate limit hit (HTTP 429) even after the allowed retries."""


class GroqTimeoutError(GroqError):
    """Request timed out even after the allowed retries."""


class GroqModelUnavailableError(GroqError):
    """Model does not exist, was renamed or was decommissioned."""


class GroqInvalidResponseError(GroqError):
    """The model answered, but not with usable, valid JSON."""


class GroqAPIError(GroqError):
    """Any other API or network failure."""


# --------------------------------------------------------------- helpers
def extract_json(text: str) -> dict:
    """Pull one JSON object out of a model reply.

    Tolerates <think>...</think> blocks and ```json fences, which some models add
    even when asked for JSON only.
    """
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise GroqInvalidResponseError("The AI did not return JSON.") from None
        try:
            value = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            raise GroqInvalidResponseError("The AI returned malformed JSON.") from None
    if not isinstance(value, dict):
        raise GroqInvalidResponseError("The AI returned JSON that is not an object.")
    return value


def _data_uri(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _api_error_message(response: httpx.Response) -> tuple[str, str]:
    """Return (code, message) from a Groq error body, truncated and key-free."""
    try:
        err = response.json().get("error", {})
        return str(err.get("code") or ""), str(err.get("message") or "")[:200]
    except (ValueError, AttributeError):
        return "", ""


class GroqService:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._api_key = (settings.groq_api_key if api_key is None else api_key).strip()
        self.model = (model or settings.effective_groq_model).strip()
        self._base_url = base_url or settings.groq_base_url
        self._timeout = settings.groq_timeout_seconds if timeout is None else timeout
        self._max_retries = settings.groq_max_retries if max_retries is None else max_retries
        self._transport = transport
        self._sleep = sleep
        self._client: httpx.Client | None = None
        self._cache: OrderedDict[str, ImageAnalysis] = OrderedDict()
        self._lock = threading.Lock()
        self.api_calls = 0  # real HTTP requests made (useful for tests and free-tier tracking)

    def __repr__(self) -> str:  # never include the key
        return f"GroqService(model={self.model!r}, configured={bool(self._api_key)})"

    # ------------------------------------------------------------ transport
    def _require_config(self) -> None:
        if not self._api_key:
            raise GroqConfigError(
                "GROQ_API_KEY is not set. Add it to backend/.env and restart the backend."
            )
        if not self.model:
            raise GroqConfigError("GROQ_MODEL is empty. Set a vision model in backend/.env.")

    @staticmethod
    def _build_verify() -> ssl.SSLContext | bool:
        """TLS trust settings. Verification is never disabled."""
        bundle = settings.groq_ca_bundle.strip().strip('"')
        if bundle:
            path = Path(bundle)
            if not path.is_file():
                raise GroqConfigError(f"GROQ_CA_BUNDLE file not found: {path}")
            return ssl.create_default_context(cafile=str(path))
        if settings.groq_use_system_certs:
            try:
                import truststore

                return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            except ImportError:
                logger.warning(
                    "truststore is not installed, so Python's bundled certificates are used. "
                    "On a company network run: pip install truststore"
                )
        return True

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self._base_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout,
                transport=self._transport,
                verify=self._build_verify(),
            )
        return self._client

    def _backoff(self, attempt: int, retry_after: str | None = None) -> None:
        delay = 2.0**attempt
        if retry_after:
            try:
                delay = float(retry_after)
            except ValueError:
                pass
        self._sleep(min(delay, MAX_BACKOFF_SECONDS))

    def _chat(self, messages: list[dict], max_tokens: int = 2048) -> str:
        """POST one chat completion and return the reply text."""
        self._require_config()
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},  # JSON mode (documented with images)
        }
        attempts = self._max_retries + 1
        for attempt in range(attempts):
            last = attempt == attempts - 1
            started = time.monotonic()
            self.api_calls += 1
            try:
                response = self._get_client().post("/chat/completions", json=payload)
            except httpx.TimeoutException:
                if last:
                    raise GroqTimeoutError("The AI service timed out. Please try again.") from None
                logger.warning("Groq timeout, retrying (attempt %d/%d)", attempt + 1, attempts)
                self._backoff(attempt)
                continue
            except httpx.HTTPError as exc:
                reason = type(exc).__name__
                detail = str(exc)[:200]  # httpx network errors never contain request headers
                logger.warning("Groq network error: %s: %s", reason, detail)
                if last:
                    hint = ""
                    if "certificate" in detail.lower():
                        hint = (
                            " A company network or antivirus may be intercepting HTTPS."
                            " Run 'pip install truststore' or set GROQ_CA_BUNDLE in backend/.env."
                        )
                    raise GroqAPIError(
                        f"Could not reach the AI service ({reason}). Check your internet connection.{hint}"
                    ) from None
                self._backoff(attempt)
                continue

            status = response.status_code
            if status == 200:
                logger.info("Groq call ok: model=%s, %.1fs", self.model, time.monotonic() - started)
                return self._reply_text(response)

            code, message = _api_error_message(response)
            if status in (401, 403):
                raise GroqAuthError("Groq rejected the API key. Check GROQ_API_KEY in backend/.env.")
            if status == 404 or code in {"model_not_found", "model_decommissioned", "model_not_active"}:
                raise GroqModelUnavailableError(
                    f"Model '{self.model}' is unavailable. Check GROQ_MODEL against Groq's current model list."
                )
            if status == 429:
                if last:
                    raise GroqRateLimitError("Groq rate limit reached. Wait a bit and try again.")
                logger.warning("Groq rate limited, retrying (attempt %d/%d)", attempt + 1, attempts)
                self._backoff(attempt, response.headers.get("retry-after"))
                continue
            if status >= 500:
                if last:
                    raise GroqAPIError(f"Groq service error (HTTP {status}). Please try again later.")
                self._backoff(attempt)
                continue
            raise GroqAPIError(f"Groq rejected the request (HTTP {status}): {message}")
        raise GroqAPIError("Groq request failed.")  # unreachable; keeps type checkers happy

    @staticmethod
    def _reply_text(response: httpx.Response) -> str:
        try:
            choice = response.json()["choices"][0]
            content = choice["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise GroqInvalidResponseError("Unexpected response format from the AI service.") from None
        if choice.get("finish_reason") == "length":
            raise GroqInvalidResponseError("The AI reply was cut off before it finished.")
        if not isinstance(content, str) or not content.strip():
            raise GroqInvalidResponseError("The AI returned an empty reply.")
        return content

    def _ask(self, schema: type[BaseModel], system: str, user: str, images: list[tuple[bytes, str]]):
        content: list[dict] = [{"type": "text", "text": user}]
        for data, mime in images:
            content.append({"type": "image_url", "image_url": {"url": _data_uri(data, mime)}})
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        obj = extract_json(self._chat(messages))
        try:
            return schema.model_validate(obj)
        except ValidationError as exc:
            fields = ", ".join(sorted({str(e["loc"][0]) for e in exc.errors() if e["loc"]}))
            logger.warning("Groq reply failed validation (fields: %s)", fields)
            raise GroqInvalidResponseError(f"The AI reply was missing or had invalid fields: {fields}.") from None

    # ----------------------------------------------------------- public API
    def analyze_image(self, image_path: Path) -> ImageAnalysis:
        """The single vision call per photo. Cached by image content."""
        data, mime = image_service.prepare_for_vision(image_path)
        key = hashlib.sha256(data + self.model.encode() + PROMPT_VERSION.encode()).hexdigest()
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                logger.info("Groq analysis served from cache")
                return cached.model_copy(deep=True)
        analysis = self._ask(ImageAnalysis, ANALYZE_SYSTEM, ANALYZE_USER, [(data, mime)])
        with self._lock:
            self._cache[key] = analysis
            while len(self._cache) > CACHE_SIZE:
                self._cache.popitem(last=False)
        return analysis.model_copy(deep=True)

    def identify_primary_object(self, analysis: ImageAnalysis) -> PrimaryObject:
        """Local view of the analysis (no API call)."""
        return PrimaryObject(
            name=analysis.primary_object,
            category=analysis.object_category,
            confidence=analysis.confidence,
            location=analysis.location,
            bounding_box=analysis.bounding_box,
            multiple_candidates=len(analysis.candidate_designs) > 1,
        )

    def generate_extraction_instructions(self, analysis: ImageAnalysis) -> ExtractionInstructions:
        """Turn the analysis into parameters for later phases (no API call)."""
        hints = analysis.extraction_hints
        sensitivity = hints.edge_sensitivity
        if analysis.is_transparent_or_reflective and sensitivity == "low":
            sensitivity = "medium"  # glass edges are faint; never go below medium
        return ExtractionInstructions(
            region_of_interest=analysis.bounding_box,
            segmentation_required=analysis.segmentation_required,
            outer_boundary_required=analysis.outer_boundary_required,
            internal_details_required=analysis.internal_details_required,
            transparent_material=analysis.is_transparent_or_reflective,
            edge_sensitivity=sensitivity,
            contrast_enhance=hints.contrast_enhance or analysis.is_transparent_or_reflective,
            line_style=hints.line_style,
            objects_to_suppress=analysis.unwanted_objects,
            notes=hints.notes,
        )

    def evaluate_extraction_result(self, original_path: Path, outline_path: Path) -> GroqEvaluation:
        """Optional second opinion (used in Phase 7). Costs one API call; the
        caller decides whether it is worth it. Advisory only."""
        original = image_service.prepare_for_vision(original_path)
        outline = image_service.prepare_for_vision(outline_path, fmt="PNG")
        return self._ask(GroqEvaluation, EVALUATE_SYSTEM, EVALUATE_USER, [original, outline])


_service: GroqService | None = None
_service_lock = threading.Lock()


def get_groq_service() -> GroqService:
    """Shared instance, so the cache and HTTP connection are reused."""
    global _service
    with _service_lock:
        if _service is None:
            _service = GroqService()
        return _service

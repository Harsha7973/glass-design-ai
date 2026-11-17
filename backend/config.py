"""Central configuration, loaded from backend/.env (never from the frontend)."""
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent

# Verified in Groq's docs on 2026-11-14: image input + JSON mode supported.
# It is listed as a *preview* model, so Groq may change or remove it.
DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    groq_api_key: str = Field(default="", repr=False)  # repr=False: never logged
    groq_model: str = ""  # empty -> DEFAULT_GROQ_MODEL
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_timeout_seconds: float = 30.0
    groq_max_retries: int = 1  # retries on 429/5xx/timeout; keeps free-tier usage low
    groq_vision_max_side: int = 1024  # longest image side sent to Groq (pixels)
    # Corporate networks often re-sign HTTPS with a company certificate that Python's
    # bundled certificates do not know. Using the OS certificate store fixes that
    # without disabling verification.
    groq_use_system_certs: bool = True
    groq_ca_bundle: str = ""  # optional path to a .pem/.crt file; overrides the above

    # Segmentation (Phase 5). "classical" = OpenCV only, "sam2" = require SAM 2
    # (falls back to classical if unavailable), "auto" = use SAM 2 only if installed.
    segmentation_backend: str = "auto"
    sam2_checkpoint: str = ""  # path to a downloaded SAM 2.1 .pt file
    sam2_model_cfg: str = "configs/sam2.1/sam2.1_hiera_t.yaml"  # must match the checkpoint
    sam2_device: str = "cpu"

    max_upload_mb: int = 15
    max_image_pixels: int = 40_000_000  # decompression-bomb guard (width * height)
    max_retries: int = 2
    processing_timeout_seconds: int = 180
    auto_process: bool = True  # start the pipeline right after a valid upload
    groq_quality_opinion: bool = False  # extra Groq call per attempt; off to save free-tier quota
    max_concurrent_jobs: int = 1  # the image work is CPU heavy; others wait in 'queued'
    cleanup_after_hours: int = 24
    cors_origins: str = (
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:5174,http://127.0.0.1:5174"
    )
    log_level: str = "INFO"

    upload_dir: Path = BASE_DIR / "uploads"
    output_dir: Path = BASE_DIR / "outputs"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def effective_groq_model(self) -> str:
        return self.groq_model.strip() or DEFAULT_GROQ_MODEL

    @property
    def groq_configured(self) -> bool:
        return bool(self.groq_api_key.strip())

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings()

"""Upload validation and secure storage.

Security notes:
- The client's filename is never used on disk; we store `original.<ext>` where
  the extension comes from the *detected* image format.
- Size is enforced while streaming, so oversized uploads are not fully buffered.
- Pillow decodes the bytes, so a renamed text/exe file is rejected even if the
  extension and Content-Type claim to be an image.
"""
import io
import logging
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from config import settings

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
FORMAT_TO_EXT = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
CHUNK_SIZE = 1024 * 1024


class UploadError(Exception):
    """A user-correctable upload problem, mapped to an HTTP status."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@dataclass(frozen=True)
class ImageInfo:
    format: str  # "JPEG" | "PNG" | "WEBP"
    width: int
    height: int
    extension: str  # ".jpg" | ".png" | ".webp"


def check_declared_type(filename: str | None, content_type: str | None) -> None:
    """Cheap pre-check on the claimed name/type. Real validation is Pillow's."""
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise UploadError(415, "Unsupported file type. Please upload a JPG, JPEG, PNG or WEBP image.")
    if content_type and content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise UploadError(415, "Unsupported content type. Please upload a JPG, PNG or WEBP image.")


async def read_limited(file: UploadFile, max_bytes: int | None = None) -> bytes:
    """Read the upload in chunks, aborting as soon as the size limit is exceeded."""
    limit = settings.max_upload_bytes if max_bytes is None else max_bytes
    buf = bytearray()
    while True:
        chunk = await file.read(CHUNK_SIZE)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > limit:
            raise UploadError(413, f"File too large. Maximum size is {settings.max_upload_mb} MB.")
    if not buf:
        raise UploadError(400, "The uploaded file is empty.")
    return bytes(buf)


def validate_image_bytes(data: bytes) -> ImageInfo:
    """Confirm the bytes are a real, supported, reasonably sized image."""
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            width, height = probe.size
            if fmt not in FORMAT_TO_EXT:
                raise UploadError(415, "Unsupported image format. Please upload a JPG, PNG or WEBP image.")
            if width * height > settings.max_image_pixels:
                raise UploadError(
                    413,
                    f"Image dimensions too large ({width}x{height}). "
                    f"Maximum is {settings.max_image_pixels:,} pixels.",
                )
            probe.verify()  # integrity check without a full decode
    except UploadError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        raise UploadError(400, "The file is not a valid image or is corrupted.") from None
    if width < 16 or height < 16:
        raise UploadError(400, "Image is too small to process (minimum 16x16 pixels).")
    return ImageInfo(format=fmt, width=width, height=height, extension=FORMAT_TO_EXT[fmt])


def save_upload(job_id: str, data: bytes, info: ImageInfo) -> Path:
    """Write the validated bytes to uploads/<job_id>/original<ext>."""
    job_dir = settings.upload_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    path = job_dir / f"original{info.extension}"
    path.write_bytes(data)
    return path


def load_oriented_image(path: Path) -> Image.Image:
    """Open an image as RGB with EXIF rotation applied.

    Phone photos are often stored sideways with an EXIF flag. Every stage that
    works with coordinates (Groq boxes, masks, outlines) must load images through
    this function so they all share the same orientation.
    """
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def prepare_for_vision(path: Path, max_side: int | None = None, fmt: str = "JPEG") -> tuple[bytes, str]:
    """Downscale (never upscale) for the vision API. Returns (bytes, mime type).

    Smaller images cost fewer tokens and stay under request-size limits. Boxes
    returned by the model are normalised (0..1), so the downscale does not matter.
    """
    limit = max_side or settings.groq_vision_max_side
    img = load_oriented_image(path)
    w, h = img.size
    scale = limit / max(w, h)
    if scale < 1:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    if fmt.upper() == "PNG":
        img.save(buf, format="PNG")
        return buf.getvalue(), "image/png"
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), "image/jpeg"

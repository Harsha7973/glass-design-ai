import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from config import settings


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Test client that writes uploads/outputs to a temp dir, not the real one."""
    monkeypatch.setattr(settings, "upload_dir", tmp_path / "uploads")
    monkeypatch.setattr(settings, "output_dir", tmp_path / "outputs")
    monkeypatch.setattr(settings, "auto_process", False)  # API tests check upload handling only
    from main import app

    with TestClient(app) as c:
        yield c


def make_image_bytes(fmt: str = "PNG", size=(64, 48)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 220, 240)).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture()
def png_bytes() -> bytes:
    return make_image_bytes("PNG")

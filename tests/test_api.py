import time

import pytest

from config import settings
from conftest import make_image_bytes
from utils.cleanup import cleanup_old_jobs


def upload(client, data: bytes, name="design.png", ctype="image/png"):
    return client.post("/process", files={"file": (name, data, ctype)})


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.parametrize(
    "fmt,name,ctype",
    [
        ("PNG", "a.png", "image/png"),
        ("JPEG", "a.jpg", "image/jpeg"),
        ("JPEG", "a.jpeg", "image/jpeg"),
        ("WEBP", "a.webp", "image/webp"),
    ],
)
def test_upload_supported_formats(client, fmt, name, ctype):
    r = upload(client, make_image_bytes(fmt, (80, 60)), name, ctype)
    assert r.status_code == 202, r.text
    body = r.json()
    assert body["status"] == "uploaded"
    assert body["width"] == 80 and body["height"] == 60
    assert len(body["job_id"]) == 32


def test_upload_is_saved_with_safe_name(client):
    r = upload(client, make_image_bytes("PNG"), "../../evil name.png")
    assert r.status_code == 202
    job_dir = settings.upload_dir / r.json()["job_id"]
    assert [p.name for p in job_dir.iterdir()] == ["original.png"]


def test_job_ids_are_unique(client, png_bytes):
    ids = {upload(client, png_bytes).json()["job_id"] for _ in range(5)}
    assert len(ids) == 5


def test_rejects_bad_extension(client, png_bytes):
    r = upload(client, png_bytes, "design.gif", "image/gif")
    assert r.status_code == 415


def test_rejects_wrong_content_type(client, png_bytes):
    r = upload(client, png_bytes, "design.png", "application/pdf")
    assert r.status_code == 415


def test_rejects_fake_image(client):
    r = upload(client, b"this is not an image at all", "fake.png", "image/png")
    assert r.status_code == 400


def test_rejects_corrupted_image(client, png_bytes):
    r = upload(client, png_bytes[: len(png_bytes) // 2], "broken.png")
    assert r.status_code == 400


def test_rejects_empty_file(client):
    r = upload(client, b"", "empty.png")
    assert r.status_code == 400


def test_rejects_oversized_file(client, monkeypatch, png_bytes):
    monkeypatch.setattr(settings, "max_upload_mb", 1)
    r = upload(client, png_bytes + b"\0" * (2 * 1024 * 1024))
    assert r.status_code == 413


def test_rejects_too_many_pixels(client, monkeypatch, png_bytes):
    monkeypatch.setattr(settings, "max_image_pixels", 1000)
    r = upload(client, png_bytes)  # 64x48 = 3072 px
    assert r.status_code == 413


def test_rejects_tiny_image(client):
    r = upload(client, make_image_bytes("PNG", (8, 8)))
    assert r.status_code == 400


def test_status_found_and_not_found(client, png_bytes):
    job_id = upload(client, png_bytes).json()["job_id"]
    r = client.get(f"/status/{job_id}")
    assert r.status_code == 200 and r.json()["status"] == "uploaded"
    assert client.get("/status/doesnotexist").status_code == 404


def test_cors_allows_dev_origin(client):
    r = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cleanup_removes_only_old_folders(tmp_path):
    import os

    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    (tmp_path / ".gitkeep").touch()
    past = time.time() - 48 * 3600
    os.utime(old, (past, past))
    assert cleanup_old_jobs(tmp_path, 24) == 1
    assert not old.exists() and new.exists() and (tmp_path / ".gitkeep").exists()

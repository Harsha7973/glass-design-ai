
# GLASS DESIGN AI

Autonomous multi-agent app that extracts the outline of the main decorative
object in a photo (no text prompt needed) and exports a clean black-on-white
PNG and SVG.

**Status: Phase 4 of 10 (Groq integration).** Upload UI works; Groq image understanding is implemented and unit-tested with mocks. Segmentation, outline extraction and export are not built yet.

## Architecture

Upload -> Orchestrator -> Image Understanding (Groq vision) -> Segmentation
(SAM 2 / OpenCV fallback) -> Outline Extraction (OpenCV) -> Quality Evaluation
(retry loop, capped) -> Export (PNG + SVG).

- `backend/` FastAPI + agents + services. Groq key lives only in `backend/.env`.
- `frontend/` React + Vite. Dev server proxies `/api` to the backend.
- `tests/` pytest suite.

## Windows (PowerShell) setup

```powershell
cd glass-design-ai\backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env   # then edit .env and add GROQ_API_KEY

cd ..\frontend
npm install
```

Run (two terminals):

```powershell
# Terminal 1
cd glass-design-ai\backend; .\.venv\Scripts\Activate.ps1
uvicorn main:app --reload --port 8000

# Terminal 2
cd glass-design-ai\frontend; npm run dev
```

## Run tests

```powershell
cd glass-design-ai
.\backend\.venv\Scripts\Activate.ps1
python -m pytest -q
```

## API (Phase 2)

- `GET /health` health check
- `POST /process` upload an image (multipart field `file`); returns a job id (HTTP 202)
- `GET /status/{job_id}` job status (currently just `uploaded`)

## Groq setup (Phase 4)

Put your key only in `backend\.env` (copied from `.env.example`):

```
GROQ_API_KEY=your-key-here
GROQ_MODEL=qwen/qwen3.8-27b
```

`qwen/qwen3.8-27b` was verified in Groq's docs on 2026-11-14 (image input + JSON mode).
Groq lists it as a **preview** model, so it may change or disappear. Re-check Groq's
model list if requests fail with "model unavailable". Free-tier limits are not published
in the docs; see the limits page in your Groq console.

Live check with a real photo (uses 1 Groq call):

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python scripts\check_groq.py C:\path\to\photo.jpg
```

### Company networks (HTTPS inspection)

If you see `CERTIFICATE_VERIFY_FAILED`, your network re-signs HTTPS traffic. Run
`pip install truststore` (already in requirements.txt). The backend then trusts the
Windows certificate store. If that is not enough, set `GROQ_CA_BUNDLE` in `backend\.env`
to your company's root certificate file. Certificate checking is never turned off.
=======
# glass-design-ai


# GLASS DESIGN AI

Autonomous multi-agent app that extracts the outline of the main decorative
object in a photo (no text prompt needed) and exports a clean black-on-white
PNG and SVG.

**Status: Phase 5 of 10 (segmentation).** Upload UI, Groq image understanding and design isolation work. Outline extraction, quality evaluation, orchestration and export are not built yet.

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

## Segmentation (Phase 5)

`agents/segmentation_agent.py` isolates the design. Etched glass has almost the same colour
as its background, so the main method looks for a concentration of thin lines (top-hat /
black-hat filtering + line density) instead of colour. The AI's bounding box only chooses
*which* region; the region can extend beyond the box. Failures are reported with
`success=false`, never hidden.

Try it on your photo (reuses the Groq analysis saved earlier, so no new API call):

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python scripts\check_segmentation.py "C:\path\to\photo.jpg" --analysis outputs\check-XXXX\analysis.json
```

Open the printed folder and look at `debug_overlay.png` (red = AI box, yellow = search area,
green = isolated region), `isolated_crop.png` and `mask.png`.

### SAM 2 (optional, not required)

SAM 2 is off unless you install it and set `SAM2_CHECKPOINT`. The OpenCV method always runs
first; SAM 2 is used only if it agrees with it. SAM 2's own docs strongly recommend WSL on
Windows. This project has NOT tested the real SAM 2 model, only a stand-in predictor.

## Phase 6 – Outline extraction
`python scripts\check_outline.py "C:\path\photo.jpg" --analysis outputs\check-XXXX\analysis.json`
Writes outer_boundary.png, internal_design.png, combined_outline.png, centerlines.png and outline_preview.png (black on white).
Options: `--sensitivity low|medium|high`, `--style thin|medium|bold`, `--grow-reach`, `--density-scale`.
Tested on synthetic scenes only; real-photo quality depends on the photo's contrast.

## Phase 7 – Quality evaluation + capped retry
`python scripts\check_quality.py "C:\path\photo.jpg" --analysis outputs\check-XXXX\analysis.json [--groq] [--max-attempts 3]`
Scores each outline with local measurements (contrast vs frost, edge coverage, boundary fit, continuity, clutter),
optionally blends Groq's visual opinion (30%, 1 call per attempt, advisory), retries with adjusted settings up to the cap,
keeps the best attempt and reports accept / needs_review / failed. Writes refinement.json.
Limits: retries only change settings; on heavily frosted photos they may not improve the result.

## Phase 6c – cleaner outlines on frosted glass
New defaults: `bright_lines_only=True` (etched grooves are bright; dark frost speckle is ignored), `sensitivity="low"`,
short dead-end spur pruning (`spur_len`). If detail is missing, the Phase 7 loop turns bright-only off, then raises sensitivity.

## Phase 8 – Orchestrator, API wiring, live status
`POST /process` now validates, saves and starts the pipeline in the background; `GET /status/{job_id}` returns the real
step messages (`events`), `warnings`, and a `result` (score, attempts, decision, file URLs); `GET /result/{job_id}/{name}`
serves the generated PNGs (whitelisted names only). If Groq is unavailable the pipeline still runs, searching the whole photo,
and says so. Settings: `AUTO_PROCESS`, `GROQ_QUALITY_OPINION` (extra Groq call per attempt, default off),
`MAX_CONCURRENT_JOBS` (default 1), `PROCESSING_TIMEOUT_SECONDS` (stops further retries). PNG/SVG downloads: Phase 9.

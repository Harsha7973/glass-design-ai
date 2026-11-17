r"""Live check: isolate -> outline -> score -> retry (capped), on one photo.

Usage (from the backend folder, venv active):
    python scripts\check_quality.py "C:\path\to\photo.jpg" --analysis outputs\check-XXXX\analysis.json
Add --groq to also ask Groq for a visual opinion on each attempt (1 API call per attempt).
--max-attempts N caps the loop (default 3). Look at outline_preview.png and refinement.json.
"""
import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.image_understanding_agent import ImageUnderstandingAgent  # noqa: E402
from agents.quality_evaluation_agent import QualityEvaluationAgent  # noqa: E402
from config import settings  # noqa: E402
from schemas.analysis import ImageAnalysis  # noqa: E402
from services.groq_service import GroqError, get_groq_service  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--analysis")
    ap.add_argument("--groq", action="store_true")
    ap.add_argument("--max-attempts", type=int, default=3)
    args = ap.parse_args()
    image = Path(args.image)
    if not image.is_file():
        print(f"File not found: {image}")
        return 2
    job_id = "q-" + uuid.uuid4().hex[:8]
    if args.analysis:
        analysis = ImageAnalysis.model_validate_json(Path(args.analysis).read_text(encoding="utf-8"))
        print("Using saved analysis (no Groq call)")
    else:
        try:
            analysis = ImageUnderstandingAgent().run(job_id, image)
        except GroqError as exc:
            print(f"Groq analysis FAILED: {exc}")
            return 1
    groq = get_groq_service() if args.groq else None
    res = QualityEvaluationAgent(groq).refine(job_id, image, analysis, max_attempts=args.max_attempts, use_groq=args.groq)
    for a in res.attempts:
        r = a.report
        m = r.metrics
        extra = f" contrast {m.contrast:.2f} clutter {m.clutter:.2f} recall {m.recall:.2f} boundary {m.boundary_fit:.2f} continuity {m.continuity:.2f}" if m else ""
        print(f"Attempt {a.attempt}: score {r.score:.2f} ({r.decision}) sens={a.outline_params.sensitivity}{extra}")
        for p in r.problems:
            print(f"    - {p}")
    print(f"\nDecision: {res.decision}  (best attempt {res.best_attempt} of {res.max_attempts})")
    print(res.message)
    print(f"\nOpen outline_preview.png in:\n{settings.output_dir / job_id}")
    return 0 if res.decision in ("accept", "needs_review") else 1


if __name__ == "__main__":
    sys.exit(main())

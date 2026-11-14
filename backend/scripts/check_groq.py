r"""Live check: run Image Understanding on one photo using your real Groq key.

Usage (from the backend folder, venv active):
    python scripts\check_groq.py path\to\photo.jpg

Prints the validated analysis as JSON. The API key is never printed.
"""
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.image_understanding_agent import ImageUnderstandingAgent  # noqa: E402
from config import settings  # noqa: E402
from services.groq_service import GroqError  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts\\check_groq.py path\\to\\photo.jpg")
        return 2
    image = Path(sys.argv[1])
    if not image.is_file():
        print(f"File not found: {image}")
        return 2

    print(f"Model:          {settings.effective_groq_model}")
    print(f"API key set:    {'yes' if settings.groq_configured else 'NO'}")
    agent = ImageUnderstandingAgent()
    job_id = "check-" + uuid.uuid4().hex[:8]
    try:
        analysis = agent.run(job_id, image)
    except GroqError as exc:
        print(f"\nFAILED: {exc}")
        return 1

    print("\n--- analysis ---")
    print(json.dumps(analysis.model_dump(), indent=2))
    print("\n--- extraction instructions (derived locally) ---")
    print(json.dumps(agent.instructions(analysis).model_dump(), indent=2))
    print(f"\nSaved to: {settings.output_dir / job_id / 'analysis.json'}")
    print(f"Groq API calls made: {agent.groq.api_calls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

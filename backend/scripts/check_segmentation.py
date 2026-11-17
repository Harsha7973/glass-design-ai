r"""Live check: isolate the design in one photo and write debug images.

Usage (from the backend folder, venv active):
    python scripts\check_segmentation.py "C:\path\to\photo.jpg" --analysis outputs\check-XXXX\analysis.json

--analysis reuses a Groq result saved earlier by check_groq.py (no new API call).
Without --analysis the script asks Groq (1 API call).

Look at debug_overlay.png in the printed folder:
  red = the AI's box, yellow = search area, green = the region that was isolated.
"""
import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.image_understanding_agent import ImageUnderstandingAgent  # noqa: E402
from agents.segmentation_agent import SegmentationAgent  # noqa: E402
from config import settings  # noqa: E402
from schemas.analysis import ImageAnalysis  # noqa: E402
from schemas.segmentation import SegmentationParams  # noqa: E402
from services.groq_service import GroqError  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--analysis", help="analysis.json saved by check_groq.py")
    ap.add_argument("--grow-reach", type=float, default=None)
    ap.add_argument("--density-scale", type=float, default=None)
    ap.add_argument("--sensitivity", choices=["low", "medium", "high"], default=None)
    args = ap.parse_args()

    image = Path(args.image)
    if not image.is_file():
        print(f"File not found: {image}")
        return 2
    job_id = "seg-" + uuid.uuid4().hex[:8]

    if args.analysis:
        analysis = ImageAnalysis.model_validate_json(Path(args.analysis).read_text(encoding="utf-8"))
        print(f"Using saved analysis: {analysis.primary_object} (no Groq call)")
    else:
        try:
            analysis = ImageUnderstandingAgent().run(job_id, image)
        except GroqError as exc:
            print(f"Groq analysis FAILED: {exc}")
            return 1

    kw = {k: v for k, v in (('grow_reach', args.grow_reach), ('density_scale', args.density_scale), ('sensitivity', args.sensitivity)) if v is not None}
    result = SegmentationAgent().run(job_id, image, analysis, SegmentationParams(**kw))
    print(f"\nSegmentation success: {result.success}")
    print(f"Method:               {result.method}")
    print(f"SAM 2:                {result.sam2_status}")
    print(f"Confidence (heuristic, not a guarantee): {result.confidence}")
    print(f"Isolated area:        {result.area_fraction:.1%} of the image")
    print(f"AI box:               {analysis.bounding_box}")
    print(f"Isolated region box:  {tuple(round(v, 2) for v in result.mask_bbox) if result.mask_bbox else None}")
    for w in result.warnings:
        print(f"WARNING: {w}")
    print(f"\nOpen this folder and look at debug_overlay.png:\n{settings.output_dir / job_id}")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())

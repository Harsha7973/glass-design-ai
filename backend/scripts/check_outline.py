r"""Live check: isolate the design, then extract its outline (black lines on white).

Usage (from the backend folder, venv active):
    python scripts\check_outline.py "C:\path\to\photo.jpg" --analysis outputs\check-XXXX\analysis.json

Open outline_preview.png in the printed folder: photo crop on the left, outline on the right.
Extra files: outer_boundary.png, internal_design.png, combined_outline.png, centerlines.png.
Options: --sensitivity low|medium|high, --style thin|medium|bold, --grow-reach 0.07
"""
import argparse
import sys
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.image_understanding_agent import ImageUnderstandingAgent  # noqa: E402
from agents.outline_extraction_agent import OutlineExtractionAgent  # noqa: E402
from agents.segmentation_agent import SegmentationAgent  # noqa: E402
from config import settings  # noqa: E402
from schemas.analysis import ImageAnalysis  # noqa: E402
from schemas.outline import OutlineParams  # noqa: E402
from schemas.segmentation import SegmentationParams  # noqa: E402
from services.groq_service import GroqError  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--analysis")
    ap.add_argument("--sensitivity", choices=["low", "medium", "high"], default="medium")
    ap.add_argument("--style", choices=["thin", "medium", "bold"], default="medium")
    ap.add_argument("--grow-reach", type=float, default=None)
    ap.add_argument("--density-scale", type=float, default=None)
    args = ap.parse_args()

    image = Path(args.image)
    if not image.is_file():
        print(f"File not found: {image}")
        return 2
    job_id = "out-" + uuid.uuid4().hex[:8]
    if args.analysis:
        analysis = ImageAnalysis.model_validate_json(Path(args.analysis).read_text(encoding="utf-8"))
        print("Using saved analysis (no Groq call)")
    else:
        try:
            analysis = ImageUnderstandingAgent().run(job_id, image)
        except GroqError as exc:
            print(f"Groq analysis FAILED: {exc}")
            return 1

    sp = {k: v for k, v in (("grow_reach", args.grow_reach), ("density_scale", args.density_scale)) if v is not None}
    seg = SegmentationAgent().run(job_id, image, analysis, SegmentationParams(**sp))
    print(f"Segmentation success: {seg.success} (area {seg.area_fraction:.1%})")
    if not seg.success:
        print("Segmentation failed, so no outline is attempted.")
        for w in seg.warnings:
            print(f"WARNING: {w}")
        return 1
    mask = np.asarray(Image.open(settings.output_dir / job_id / "mask.png"))
    res = OutlineExtractionAgent().run(
        job_id, image, mask, analysis, OutlineParams(sensitivity=args.sensitivity, line_style=args.style)
    )
    print(f"Outline success: {res.success}")
    print(f"Line components: {res.components}, line length: {res.total_line_length_px}px, "
          f"noise removed: {res.noise_components_removed}, outer contours: {res.outer_contours}")
    for w in res.warnings:
        print(f"WARNING: {w}")
    print(f"\nOpen outline_preview.png in:\n{settings.output_dir / job_id}")
    return 0 if res.success else 1


if __name__ == "__main__":
    sys.exit(main())

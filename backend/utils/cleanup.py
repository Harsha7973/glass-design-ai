"""Temporary-file cleanup: removes job folders older than the configured age."""
import logging
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def cleanup_old_jobs(root: Path, max_age_hours: int) -> int:
    """Delete job sub-folders of `root` not modified within `max_age_hours`.

    Only direct sub-directories are touched; dotfiles such as .gitkeep stay.
    Returns the number of folders removed.
    """
    if not root.exists():
        return 0
    cutoff = time.time() - max_age_hours * 3600
    removed = 0
    for child in root.iterdir():
        if not child.is_dir():
            continue
        try:
            if child.stat().st_mtime < cutoff:
                shutil.rmtree(child)
                removed += 1
        except OSError as exc:
            logger.warning("Could not remove %s: %s", child.name, exc)
    if removed:
        logger.info("Cleanup removed %d old job folder(s) from %s", removed, root.name)
    return removed

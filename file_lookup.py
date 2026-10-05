"""Finding detections files and their videos on disk.

A detections file names its video only by file name (`videos[0].file_name`),
so the video is looked up under a root folder: the one with that name
closest to the detections file.
"""

from __future__ import annotations

import os
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
# Not searched for files.
SKIP_DIRS = {"node_modules", "__pycache__", "venv", "site-packages"}


def walk_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if not d.startswith(".") and d not in SKIP_DIRS
        )
        for name in filenames:
            if not name.startswith("."):
                yield Path(dirpath) / name


def find_videos(root: Path) -> list[Path]:
    return sorted(
        p for p in walk_files(root) if p.suffix.lower() in VIDEO_EXTENSIONS
    )


def best_video(json_path: Path, name: str, videos: list[Path]) -> Path | None:
    """The video called `name`, the closest one to the JSON if several."""
    candidates = [v for v in videos if v.name == name]
    if not candidates:
        return None

    def closeness(video: Path) -> tuple[int, int]:
        common = os.path.commonpath([video, json_path])
        return (-len(Path(common).parts), len(video.parts))

    return min(candidates, key=closeness)

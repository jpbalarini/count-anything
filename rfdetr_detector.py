"""RF-DETR detector for process_video.py.

Same interface as the cloud and locate-anything detectors:
`detect_frames` takes (key, BGR frame) pairs and returns
{key: sv.Detections | None}, in the pixel space of the frames it was
given.

The stock model knows the fixed COCO classes: it returns COCO class ids
and keeps every class above the threshold; filtering to the wanted
classes happens later. A fine-tuned model (`weights`, trained by
train_rfdetr.py) knows its own classes instead, `class_names`, and
returns their 0-based positions in that list as class ids.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
import supervision as sv

MODEL_SIZES = ["nano", "small", "medium", "large"]
DEFAULT_MODEL_SIZE = "medium"
# The checkpoint train_rfdetr.py keeps in its output folder, next to
# its description.
WEIGHTS_FILE = "checkpoint_best_total.pth"
MANIFEST_FILE = "model.json"


def weights_path(path: str | Path) -> Path:
    """The checkpoint for --weights: a .pth file, or a folder written by
    train_rfdetr.py."""
    path = Path(path)
    if path.is_dir():
        path = path / WEIGHTS_FILE
    if not path.is_file():
        raise ValueError(
            f"no RF-DETR weights at {path} (expected a .pth file or a "
            "folder made by train_rfdetr.py)"
        )
    return path


def recommended_threshold(weights: str | Path) -> float | None:
    """The confidence threshold train_rfdetr.py chose for a model (the
    best F1 on its validation frames), if it has one."""
    path = Path(weights)
    manifest = (path if path.is_dir() else path.parent) / MANIFEST_FILE
    try:
        threshold = json.loads(manifest.read_text()).get("threshold")
    except (OSError, ValueError, AttributeError):
        return None
    return float(threshold) if isinstance(threshold, (int, float)) else None


class RfDetrDetector:
    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE,
        threshold: float = 0.2,
        weights: str | Path | None = None,
    ):
        import rfdetr

        if weights is not None:
            path = weights_path(weights)
            print(f"Loading fine-tuned RF-DETR from {path}...")
            self.model = rfdetr.RFDETR.from_checkpoint(str(path))
            self.size = type(self.model).__name__.removeprefix("RFDETR").lower()
            # 0-based class ids into this list.
            self.class_names: list[str] | None = list(self.model.class_names)
            print(
                f"RF-DETR {self.size} loaded, classes: "
                + ", ".join(self.class_names)
            )
        else:
            name = model_size.capitalize()
            print(f"Loading RF-DETR {name}...")
            self.model = getattr(rfdetr, f"RFDETR{name}")()
            self.size = model_size
            self.class_names = None  # COCO ids
            print(f"RF-DETR {name} loaded")

        self.threshold = threshold
        self.calls = 0
        self.failures = 0  # RF-DETR errors are not recoverable: it raises

    def detect(self, frame_bgr: np.ndarray) -> sv.Detections:
        self.calls += 1
        # Frames come from OpenCV / supervision (BGR), RF-DETR wants RGB.
        detections = self.model.predict(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
            threshold=self.threshold,
        )
        if self.class_names is not None and len(detections):
            # A fine-tuned head also has a "no object" slot; drop it.
            detections = detections[
                detections.class_id < len(self.class_names)
            ]
        return detections

    def detect_frames(
        self,
        frames: Iterable[tuple[int, np.ndarray]],
        on_done: Callable[[int], None] | None = None,
    ) -> dict[int, sv.Detections | None]:
        """Detect on many (key, frame) pairs, one frame at a time."""
        results: dict[int, sv.Detections | None] = {}
        for key, frame in frames:
            results[key] = self.detect(frame)
            if on_done is not None:
                on_done(len(results))
        return results

"""RF-DETR detector for process_video.py.

Same interface as the cloud and locate-anything detectors:
`detect_frames` takes (key, BGR frame) pairs and returns
{key: sv.Detections | None}, in the pixel space of the frames it was
given. RF-DETR is a closed-vocabulary model (the fixed COCO classes), so
it returns COCO class ids and keeps every class above the threshold;
filtering to the wanted classes happens later.
"""

from __future__ import annotations

from typing import Callable, Iterable

import cv2
import numpy as np
import supervision as sv

MODEL_SIZES = ["nano", "small", "medium", "large"]
DEFAULT_MODEL_SIZE = "medium"


class RfDetrDetector:
    def __init__(
        self, model_size: str = DEFAULT_MODEL_SIZE, threshold: float = 0.2
    ):
        import rfdetr

        name = model_size.capitalize()
        print(f"Loading RF-DETR {name}...")
        self.model = getattr(rfdetr, f"RFDETR{name}")()
        print(f"RF-DETR {name} loaded")

        self.threshold = threshold
        self.calls = 0
        self.failures = 0  # RF-DETR errors are not recoverable: it raises

    def detect(self, frame_bgr: np.ndarray) -> sv.Detections:
        self.calls += 1
        # Frames come from OpenCV / supervision (BGR), RF-DETR wants RGB.
        return self.model.predict(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
            threshold=self.threshold,
        )

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

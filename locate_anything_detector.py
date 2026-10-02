"""locate-anything.cpp detector for process_video.py.

Wraps the `locate-anything-cli` binary
(https://github.com/mudler/locate-anything.cpp), an open-vocabulary
detector (NVIDIA LocateAnything-3B on ggml). Like the cloud detector it
is meant to run on sampled frames: each call writes the frame to a
temporary JPEG, runs

    locate-anything-cli detect --model M --input frame.jpg \\
        --prompt "Locate all the instances that matches the following \\
description: a</c>b." --output boxes.json

and reads back `{"detections": [{"label": ..., "box": [x1,y1,x2,y2]}]}`
(boxes are pixels of the image it was given). The CLI gives no
confidence score, so every detection gets confidence 1.0.

Note: the CLI loads the model on every call, so there is a fixed
per-frame cost on top of inference; use a low --sample-rate.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
import supervision as sv

CLI_NAME = "locate-anything-cli"
DEFAULT_LOCATE_MODE = "hybrid"
DEFAULT_LOCATE_SAMPLE_RATE = 1.0
PROMPT_PREFIX = (
    "Locate all the instances that matches the following description: "
)
CATEGORY_SEPARATOR = "</c>"


def _normalize(name: str) -> str:
    return (
        name.strip().strip(".").lower().replace("_", " ").replace("-", " ")
    )


class LocateAnythingDetector:
    def __init__(
        self,
        model_path: str,
        class_names: dict[int, str],
        mode: str = DEFAULT_LOCATE_MODE,
        threads: int = 0,
    ):
        self.cli = shutil.which(CLI_NAME)
        if self.cli is None:
            raise RuntimeError(f"`{CLI_NAME}` not found in PATH")
        if not Path(model_path).is_file():
            raise RuntimeError(f"locate-anything model not found: {model_path}")

        self.model_path = str(model_path)
        self.mode = mode
        self.threads = threads

        self.name_to_id = {
            _normalize(name): class_id
            for class_id, name in class_names.items()
        }
        self.prompt = (
            PROMPT_PREFIX
            + CATEGORY_SEPARATOR.join(class_names.values())
            + "."
        )

        self.calls = 0
        self.failures = 0
        self._warned_labels: set[str] = set()

    @staticmethod
    def _empty() -> sv.Detections:
        return sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.empty(0, dtype=np.float32),
            class_id=np.empty(0, dtype=int),
        )

    def _class_id(self, label: str) -> int | None:
        """Map the label the model wrote back to one of our classes."""
        key = _normalize(label)
        if key in self.name_to_id:
            return self.name_to_id[key]
        # Tolerate the model pluralizing / singularizing.
        for name, class_id in self.name_to_id.items():
            if name.rstrip("s") == key.rstrip("s"):
                return class_id
        if len(self.name_to_id) == 1:
            return next(iter(self.name_to_id.values()))
        if key not in self._warned_labels:
            self._warned_labels.add(key)
            print(
                f"warning: locate-anything returned unknown label "
                f"'{label}', ignoring those boxes"
            )
        return None

    def detect(self, frame_bgr: np.ndarray) -> sv.Detections | None:
        """Return detections, or None if the CLI call failed."""
        height, width = frame_bgr.shape[:2]
        self.calls += 1

        with tempfile.TemporaryDirectory(prefix="locate_") as tmp:
            image_path = Path(tmp) / "frame.jpg"
            out_path = Path(tmp) / "boxes.json"
            cv2.imwrite(
                str(image_path),
                frame_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

            cmd = [
                self.cli,
                "detect",
                "--model", self.model_path,
                "--input", str(image_path),
                "--prompt", self.prompt,
                "--output", str(out_path),
                "--mode", self.mode,
                "--threads", str(self.threads),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            try:
                if proc.returncode != 0:
                    raise RuntimeError(
                        proc.stderr.strip().splitlines()[-1]
                        if proc.stderr.strip()
                        else f"exit code {proc.returncode}"
                    )
                items = json.loads(out_path.read_text())["detections"]
            except (RuntimeError, OSError, ValueError, KeyError) as exc:
                self.failures += 1
                print(
                    f"warning: {CLI_NAME} failed, skipping sample: {exc}"
                )
                return None

        boxes, ids = [], []
        for item in items:
            try:
                class_id = self._class_id(item["label"])
                x1, y1, x2, y2 = (float(v) for v in item["box"])
            except (KeyError, TypeError, ValueError):
                continue
            if class_id is None:
                continue
            x1, x2 = sorted((min(max(x1, 0), width), min(max(x2, 0), width)))
            y1, y2 = sorted((min(max(y1, 0), height), min(max(y2, 0), height)))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            boxes.append([x1, y1, x2, y2])
            ids.append(class_id)

        if not boxes:
            return self._empty()
        return sv.Detections(
            xyxy=np.array(boxes, dtype=np.float32),
            confidence=np.ones(len(boxes), dtype=np.float32),
            class_id=np.array(ids, dtype=int),
        )

    def detect_frames(
        self,
        frames: Iterable[tuple[int, np.ndarray]],
        on_done: Callable[[int], None] | None = None,
    ) -> dict[int, sv.Detections | None]:
        """Detect on many (key, frame) pairs, one CLI call at a time."""
        results: dict[int, sv.Detections | None] = {}
        for key, frame in frames:
            results[key] = self.detect(frame)
            if on_done is not None:
                on_done(len(results))
        return results

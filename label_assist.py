"""Label Assist: run a detector on the frame being annotated.

annotate_server.py asks for boxes on one frame; the annotator shows them
as proposals to review before they are added. The detectors are the
ones process_video.py uses:

* cloud    a cloud vision model (cloud_detector.py). Any class name, each
           with an optional description added to the prompt.
* locate   locate-anything.cpp (locate_anything_detector.py). Any class
           name (descriptions are ignored).
* rfdetr   the stock RF-DETR, COCO classes only.
* trained  an RF-DETR fine-tuned by train_rfdetr.py, its own classes.

Local models stay loaded between calls (the last few RF-DETRs, and the
locate-anything worker, whatever the classes), so only the first call
pays for loading them.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from cli import DEFAULT_THRESHOLD
from cloud_detector import DEFAULT_CLOUD_MAX_SIDE, DEFAULT_CLOUD_MODEL, CloudDetector
from frames import inference_size
from locate_anything_detector import CLI_NAME, LocateAnythingDetector, class_prompt
from options import normalize_name
from rfdetr_detector import (
    DEFAULT_MODEL_SIZE,
    MODEL_SIZES,
    RfDetrDetector,
    recommended_threshold,
)

SOURCES = ("cloud", "locate", "rfdetr", "trained")
CLOUD_MODELS = {
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
    "claude-opus-5-5": "Claude Opus 5.5",
}
# Scored detectors return every box above this; the annotator hides the
# ones under the model's threshold with a slider, so it can be moved
# without detecting again.
MIN_SCORE = 0.05
# Shorter side of the frame given to locate-anything (see
# --infer-resolution in the README: 720p is ~5 s a frame).
LOCATE_RESOLUTION = 720
# RF-DETRs kept loaded (stock sizes and fine-tuned models).
MAX_LOADED_RFDETR = 2


class AssistError(Exception):
    """Detection ran but failed (API error, worker crash, ...)."""


def _box(xyxy, scale: float = 1.0) -> list[float]:
    return [round(float(v) / scale, 2) for v in xyxy]


class LabelAssist:
    def __init__(self):
        # Local models run one call at a time; cloud calls don't wait.
        self._lock = threading.Lock()
        self._rfdetr: OrderedDict[str, RfDetrDetector] = OrderedDict()
        self._locate: LocateAnythingDetector | None = None
        self._locate_key: tuple | None = None

    def close(self) -> None:
        with self._lock:
            if self._locate is not None:
                self._locate.close()
                self._locate = None

    # -- What can run ----------------------------------------------

    @staticmethod
    def locate_paths(info: dict) -> tuple[str | None, str | None]:
        """(model, library) for locate-anything: from the environment
        (.env) like process_video.py, else the model recorded in a
        locate-anything detections file."""
        model = os.environ.get("LOCATE_ANYTHING_MODEL")
        if not model and info.get("detector") == "locate-anything":
            model = info.get("model")
        return model or None, os.environ.get("LOCATE_ANYTHING_LIB") or None

    def sources(self, info: dict, trained: list[dict]) -> dict:
        """What the annotator offers, and why a source can't be used."""
        cloud_ready = bool(
            os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        )
        model, lib = self.locate_paths(info)
        if not model:
            locate_reason = "Set LOCATE_ANYTHING_MODEL in .env and restart the annotator."
        elif not Path(model).is_file():
            locate_reason = f"Model not found: {model}"
        elif lib and not Path(lib).is_file():
            locate_reason = f"Library not found: {lib}"
        elif not lib and shutil.which(CLI_NAME) is None:
            locate_reason = (
                f"`{CLI_NAME}` is not in PATH; set LOCATE_ANYTHING_LIB in .env "
                "and restart the annotator."
            )
        else:
            locate_reason = None
        return {
            "cloud": {
                "available": cloud_ready,
                "reason": None if cloud_ready else (
                    "Put ANTHROPIC_API_KEY=... in .env and restart the annotator."
                ),
                "models": [{"id": k, "name": v} for k, v in CLOUD_MODELS.items()],
                "default": DEFAULT_CLOUD_MODEL,
                "threshold": DEFAULT_THRESHOLD,
            },
            "locate": {
                "available": locate_reason is None,
                "reason": locate_reason,
                "model": Path(model).name if model else None,
                "resident": bool(lib),
            },
            "rfdetr": {
                "available": True,
                "sizes": MODEL_SIZES,
                "default": DEFAULT_MODEL_SIZE,
                "threshold": DEFAULT_THRESHOLD,
                "loaded": [k for k in self._rfdetr if k in MODEL_SIZES],
            },
            "trained": {
                "available": bool(trained),
                "reason": None if trained else "No trained models under the folder yet.",
                "models": trained,
            },
        }

    # -- Detection --------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
        source: str,
        model: str | None,
        classes: list[tuple[str, str]],
        info: dict,
        coco: dict[int, str],
    ) -> dict:
        """Boxes for `classes` ([(name, description)]) on a BGR frame,
        in its pixel space. ValueError: bad request; AssistError: the
        detector failed."""
        names = [n.strip() for n, _ in classes if n.strip()]
        if not names:
            raise ValueError("choose at least one class")
        descriptions = {n.strip(): d.strip() for n, d in classes if n.strip()}
        start = time.monotonic()
        if source == "cloud":
            result = self._cloud(frame, model or DEFAULT_CLOUD_MODEL, names, descriptions)
        elif source == "locate":
            with self._lock:
                result = self._locate_anything(frame, names, info)
        elif source == "rfdetr":
            with self._lock:
                result = self._stock_rfdetr(frame, model or DEFAULT_MODEL_SIZE, names, coco)
        elif source == "trained":
            if not model:
                raise ValueError("choose a trained model")
            with self._lock:
                result = self._trained(frame, model, names)
        else:
            raise ValueError(f"unknown source {source!r}")
        result["seconds"] = round(time.monotonic() - start, 2)
        return result

    def _cloud(self, frame, model, names, descriptions) -> dict:
        if model not in CLOUD_MODELS:
            raise ValueError(f"unknown cloud model {model}")
        detector = CloudDetector(
            model=model,
            class_names=dict(enumerate(names)),
            threshold=0.0,  # the annotator's slider filters
            max_side=DEFAULT_CLOUD_MAX_SIDE,
            descriptions=descriptions,
        )
        detections = detector.detect(frame)
        if detections is None:
            raise AssistError(f"{CLOUD_MODELS[model]} failed: {detector.last_error}")
        return {
            "boxes": [
                {"label": names[c], "confidence": round(float(s), 4), "box": _box(xyxy)}
                for xyxy, s, c in zip(
                    detections.xyxy, detections.confidence, detections.class_id
                )
            ],
            "threshold": DEFAULT_THRESHOLD,
            "model": CLOUD_MODELS[model],
            "usage": {
                "input_tokens": detector.input_tokens,
                "output_tokens": detector.output_tokens,
            },
        }

    def _locate_anything(self, frame, names, info) -> dict:
        model, lib = self.locate_paths(info)
        if not model:
            raise ValueError("no locate-anything model (LOCATE_ANYTHING_MODEL)")
        key = (model, lib)
        if self._locate is None or self._locate_key != key:
            if self._locate is not None:
                self._locate.close()
            self._locate = None
            try:
                # The worker (with `lib`) loads the model here and keeps
                # it for the next calls, whatever the classes.
                self._locate = LocateAnythingDetector(
                    model_path=model, class_names=dict(enumerate(names)), lib_path=lib
                )
            except RuntimeError as exc:
                raise AssistError(f"locate-anything: {exc}") from exc
            self._locate_key = key
        # One query per class (see locate_anything_detector.py).
        self._locate.prompts = {i: class_prompt(n) for i, n in enumerate(names)}
        height, width = frame.shape[:2]
        w, h = inference_size(width, height, LOCATE_RESOLUTION)
        scale = w / width
        small = frame if scale == 1 else cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        detections = self._locate.detect(small)
        if detections is None:
            raise AssistError(f"locate-anything failed: {self._locate.last_error}")
        return {
            "boxes": [
                {"label": names[c], "confidence": 1.0, "box": _box(xyxy, scale)}
                for xyxy, c in zip(detections.xyxy, detections.class_id)
            ],
            "threshold": None,
            "model": f"locate-anything ({Path(model).name})",
        }

    def _rfdetr_model(self, key: str, **kwargs) -> RfDetrDetector:
        detector = self._rfdetr.pop(key, None)
        if detector is None:
            detector = RfDetrDetector(threshold=MIN_SCORE, **kwargs)
            while len(self._rfdetr) >= MAX_LOADED_RFDETR:
                self._rfdetr.popitem(last=False)
        self._rfdetr[key] = detector  # most recently used last
        return detector

    def _stock_rfdetr(self, frame, size, names, coco) -> dict:
        if size not in MODEL_SIZES:
            raise ValueError(f"unknown RF-DETR size {size}")
        ids = {normalize_name(n): i for i, n in coco.items()}
        unknown = [n for n in names if normalize_name(n) not in ids]
        if unknown:
            raise ValueError(f"not COCO classes: {', '.join(unknown)}")
        wanted = {ids[normalize_name(n)]: n for n in names}
        detections = self._rfdetr_model(size, model_size=size).detect(frame)
        return {
            "boxes": [
                {"label": wanted[c], "confidence": round(float(s), 4), "box": _box(xyxy)}
                for xyxy, s, c in zip(
                    detections.xyxy, detections.confidence, detections.class_id
                )
                if int(c) in wanted
            ],
            "threshold": DEFAULT_THRESHOLD,
            "model": f"RF-DETR {size} (COCO)",
        }

    def _trained(self, frame, folder: str, names) -> dict:
        detector = self._rfdetr_model(folder, weights=folder)
        model_names = detector.class_names or []
        wanted = {normalize_name(n) for n in names}
        detections = detector.detect(frame)
        return {
            "boxes": [
                {
                    "label": model_names[c],
                    "confidence": round(float(s), 4),
                    "box": _box(xyxy),
                }
                for xyxy, s, c in zip(
                    detections.xyxy, detections.confidence, detections.class_id
                )
                if normalize_name(model_names[c]) in wanted
            ],
            "threshold": recommended_threshold(folder) or DEFAULT_THRESHOLD,
            "model": Path(folder).name,
        }

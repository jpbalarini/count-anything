"""Cloud vision model detector (Anthropic) for process_video.py.

Alternative to the RF-DETR detector: instead of running on every frame,
`CloudDetector.detect` is called on sampled frames only and returns
`sv.Detections` in original-frame pixel coordinates.
"""

from __future__ import annotations

import base64
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable

import cv2
import numpy as np
import supervision as sv

DEFAULT_CLOUD_MODEL = "claude-sonnet-5-5"
DEFAULT_SAMPLE_RATE = 2.0  # cloud samples per second of video
DEFAULT_CLOUD_MAX_SIDE = 1568  # longest image side sent to the model
DEFAULT_CLOUD_CONCURRENCY = 8  # simultaneous API calls

_DETECTIONS_TOOL = "report_detections"


def _normalize(name: str) -> str:
    return name.strip().lower().replace("_", " ").replace("-", " ")


class CloudDetector:
    """Detects objects by asking a cloud vision model (Anthropic).

    The frame is downscaled, sent as a JPEG, and the model answers
    through a forced tool call, which gives us schema-checked JSON
    (label / confidence / box) instead of free text to parse.
    Boxes come back in the pixel space of the image the model saw and
    are scaled back to the original frame.
    """

    def __init__(
        self,
        model: str,
        class_names: dict[int, str],
        threshold: float,
        max_side: int,
        descriptions: dict[str, str] | None = None,
    ):
        """`descriptions`: optional {class name: what it means}, added
        to the prompt (e.g. "only people wearing a helmet")."""
        import anthropic

        self.anthropic = anthropic
        # The SDK retries 429/5xx with backoff; be generous since we
        # may hit rate limits when calls run in parallel.
        self.client = anthropic.Anthropic(max_retries=5)
        self.model = model
        self.threshold = threshold
        self.max_side = max_side

        self.name_to_id = {
            _normalize(name): class_id
            for class_id, name in class_names.items()
        }
        self.labels = list(class_names.values())
        self.descriptions = {
            name: text.strip()
            for name, text in (descriptions or {}).items()
            if text and text.strip()
        }

        self.tool = {
            "name": _DETECTIONS_TOOL,
            "description": (
                "Report every detected object with a tight "
                "bounding box."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "detections": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {
                                    "type": "string",
                                    "enum": self.labels,
                                },
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "box": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 4,
                                    "maxItems": 4,
                                    "description": (
                                        "[x1, y1, x2, y2] in pixels "
                                        "of the image, origin at the "
                                        "top-left corner."
                                    ),
                                },
                            },
                            "required": ["label", "confidence", "box"],
                        },
                    }
                },
                "required": ["detections"],
            },
        }

        self._lock = threading.Lock()  # guards the counters below
        self.calls = 0
        self.failures = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.last_error: str | None = None  # why the last failed call failed

    @staticmethod
    def _empty() -> sv.Detections:
        return sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.empty(0, dtype=np.float32),
            class_id=np.empty(0, dtype=int),
        )

    def _prepare(
        self, frame_bgr: np.ndarray
    ) -> tuple[str, float, int, int]:
        """Downscale + JPEG-encode a frame (cheap, done in the caller).

        Returns (base64 data, scale, image width, image height).
        Only this small payload is handed to worker threads, so full
        resolution frames are not kept alive while calls are pending.
        """
        height, width = frame_bgr.shape[:2]
        scale = min(1.0, self.max_side / max(height, width))
        image = (
            cv2.resize(
                frame_bgr,
                (round(width * scale), round(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
            if scale < 1.0
            else frame_bgr
        )
        img_h, img_w = image.shape[:2]

        ok, buf = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
        if not ok:
            raise RuntimeError("could not JPEG-encode frame")
        data = base64.standard_b64encode(buf.tobytes()).decode()
        return data, scale, img_w, img_h

    def detect(self, frame_bgr: np.ndarray) -> sv.Detections | None:
        """Return detections, or None if the API call failed."""
        return self._detect_prepared(self._prepare(frame_bgr))

    def detect_frames(
        self,
        frames: Iterable[tuple[int, np.ndarray]],
        concurrency: int,
        on_done: Callable[[int, int], None] | None = None,
    ) -> dict[int, sv.Detections | None]:
        """Detect on many (key, frame) pairs with parallel API calls.

        At most `concurrency` calls are in flight at once (a small
        queue on top keeps the workers busy). Returns {key: detections
        or None if that call failed}. `on_done(finished, submitted)`
        is called from worker threads as calls complete.
        """
        futures = {}
        slots = threading.BoundedSemaphore(concurrency * 2)
        finished = 0

        def run(payload):
            try:
                return self._detect_prepared(payload)
            finally:
                slots.release()

        def done(_future):
            nonlocal finished
            with self._lock:
                finished += 1
                n, total = finished, len(futures)
            if on_done is not None:
                on_done(n, total)

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for key, frame in frames:
                slots.acquire()  # blocks while the queue is full
                future = pool.submit(run, self._prepare(frame))
                futures[key] = future
                future.add_done_callback(done)
        return {key: f.result() for key, f in futures.items()}

    def _failed(self, reason: str) -> None:
        with self._lock:
            self.failures += 1
            self.last_error = reason
        print(f"warning: cloud call failed, skipping sample: {reason}")
        return None

    @staticmethod
    def _parse_text_answer(response) -> list:
        text = "".join(
            b.text for b in response.content if b.type == "text"
        )
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            return []
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return []
        items = parsed.get("detections", []) if isinstance(parsed, dict) else []
        return items if isinstance(items, list) else []

    def _detect_prepared(
        self, prepared: tuple[str, float, int, int]
    ) -> sv.Detections | None:
        data, scale, img_w, img_h = prepared

        prompt = (
            f"This image is {img_w}x{img_h} pixels. Detect every "
            "visible instance of these object types: "
            f"{', '.join(self.labels)}.\n"
            + "".join(
                f"Only count as {name}: {text}\n"
                for name, text in self.descriptions.items()
            )
            + "Give a tight bounding box for each one as "
            "[x1, y1, x2, y2] in pixel coordinates of this image "
            "(origin top-left). Include partially visible objects, "
            "report each physical object exactly once, and ignore "
            "everything that is not in the list. If there are none, "
            "return an empty list. "
            f"You MUST answer by calling the {_DETECTIONS_TOOL} tool "
            "exactly once, and reply with nothing else."
        )

        with self._lock:
            self.calls += 1
        try:
            # Streamed: the SDK refuses a non-streaming request with this
            # many max_tokens (it could outlast its 10 minute timeout).
            # Only the final message is used.
            with self.client.messages.stream(
                model=self.model,
                # Room for thinking plus a long list of boxes.
                max_tokens=32000,
                tools=[self.tool],
                # Not {"type": "tool", ...}: forcing a tool is rejected
                # by some models (400), so ask for it in the prompt.
                tool_choice={"type": "auto"},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": data,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            ) as stream:
                response = stream.get_final_message()
        except self.anthropic.APIError as exc:
            return self._failed(str(exc))

        with self._lock:
            self.input_tokens += response.usage.input_tokens
            self.output_tokens += response.usage.output_tokens

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None)
            return self._failed(
                "the model declined the request"
                + (f" ({category})" if category else "")
            )

        items = []
        for block in response.content:
            if (
                block.type == "tool_use"
                and block.name == _DETECTIONS_TOOL
            ):
                items = block.input.get("detections", [])
                break
        else:
            # No tool call: accept a JSON answer in plain text.
            items = self._parse_text_answer(response)
            if not items and response.stop_reason == "max_tokens":
                return self._failed("the answer was cut off (max_tokens)")

        boxes, confs, ids = [], [], []
        for item in items:
            try:
                class_id = self.name_to_id[_normalize(item["label"])]
                conf = float(item.get("confidence", 1.0))
                x1, y1, x2, y2 = (float(v) for v in item["box"])
            except (KeyError, TypeError, ValueError):
                continue  # malformed entry
            if conf < self.threshold:
                continue
            # Clip to the image, restore original frame scale.
            x1, x2 = sorted((min(max(x1, 0), img_w), min(max(x2, 0), img_w)))
            y1, y2 = sorted((min(max(y1, 0), img_h), min(max(y2, 0), img_h)))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            boxes.append([x1 / scale, y1 / scale, x2 / scale, y2 / scale])
            confs.append(conf)
            ids.append(class_id)

        if not boxes:
            return self._empty()
        return sv.Detections(
            xyxy=np.array(boxes, dtype=np.float32),
            confidence=np.array(confs, dtype=np.float32),
            class_id=np.array(ids, dtype=int),
        )

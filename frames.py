"""Frame-level helpers: which frames get detected, the detection
resolution, and box interpolation between detected frames.
"""

from __future__ import annotations

import copy

import numpy as np
import supervision as sv


def sampling_due(
    index: int, next_at: float, step: float
) -> tuple[bool, float]:
    """Is `index` a sampled frame? Returns (due, updated next_at).

    Shared by the detection pre-pass and the render pass so both pick
    exactly the same frames.
    """
    if index + 1e-9 < next_at:
        return False, next_at
    while next_at <= index + 1e-9:
        next_at += step
    return True, next_at


def interpolate_detections(
    before: sv.Detections, after: sv.Detections, t: float
) -> sv.Detections:
    """Boxes between two tracked samples, `t` in [0, 1] of the way.

    Every object in `before` is kept (with its class, confidence and
    track id). Objects that are also in `after` (same track id) move
    linearly towards their box there; objects that are gone by `after`
    stay where they were. Objects that only appear in `after` are not
    drawn yet.
    """
    if (
        len(before) == 0
        or len(after) == 0
        or before.tracker_id is None
        or after.tracker_id is None
    ):
        return before

    after_row = {int(tid): i for i, tid in enumerate(after.tracker_id)}
    xyxy = before.xyxy.astype(np.float64)
    for i, tid in enumerate(before.tracker_id):
        j = after_row.get(int(tid))
        if j is not None:
            xyxy[i] = (1 - t) * xyxy[i] + t * after.xyxy[j]

    out = copy.copy(before)
    out.xyxy = xyxy.astype(np.float32)
    return out


class CountedLine:
    """A LineZone that reports given in/out counts instead of its own.

    The line is tracked ahead of the frame being drawn when boxes are
    interpolated, so the annotator needs the counts as they were at
    the sample being shown.
    """

    def __init__(self, line: sv.LineZone, in_count: int, out_count: int):
        self._line = line
        self.in_count = in_count
        self.out_count = out_count

    def __getattr__(self, name: str):
        return getattr(self._line, name)


def inference_size(
    width: int, height: int, resolution: int | None
) -> tuple[int, int]:
    """Frame size used for detection: shorter side = `resolution`.

    Keeps the aspect ratio and never upscales (1080p with 720 gives
    1280x720; a 640x480 video is left alone).
    """
    if not resolution or min(width, height) <= resolution:
        return width, height
    scale = resolution / min(width, height)
    return round(width * scale), round(height * scale)

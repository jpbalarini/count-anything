"""Frame-level helpers: which frames get detected, the detection
resolution, and box interpolation between detected frames.
"""

from __future__ import annotations

import bisect

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


class TrackTimeline:
    """Where every tracked box is on each frame between the samples.

    A track's detections are keyframes. Its box on any frame is linearly
    interpolated between the keyframes on either side of it, so a track
    that a sample missed is placed using the samples before and after.

    A stretch of missed samples is only bridged when it is at most
    `max_missed` samples long. Otherwise the object is taken to be gone:
    its box stays where it was until the next sample, and the track is
    drawn again from its next detection on. The last sample's boxes stay
    until the video ends. A track is never drawn before its first
    detection.

    `indices` are the frame indices of the samples (ascending), `samples`
    their tracked detections.
    """

    def __init__(
        self,
        indices: list[int],
        samples: list[sv.Detections],
        max_missed: int,
    ):
        self.indices = indices

        # {track id: [(sample position, row in that sample), ...]}
        keyframes: dict[int, list[tuple[int, int]]] = {}
        for pos, detections in enumerate(samples):
            if detections.tracker_id is None:
                continue
            for row, tracker_id in enumerate(detections.tracker_id):
                keyframes.setdefault(int(tracker_id), []).append((pos, row))

        # Per interval between samples, the boxes drawn in it, each as
        # (first frame, last frame, first box, last box, class id,
        # confidence, track id): linear motion from the first box on the
        # first frame to the last box on the last frame (none if equal).
        drawn: list[list[tuple]] = [[] for _ in indices]
        for tracker_id, keys in keyframes.items():
            for k, (pos, row) in enumerate(keys):
                first = samples[pos]
                start_box = first.xyxy[row]
                confidence = (
                    first.confidence[row]
                    if first.confidence is not None
                    else 1.0
                )

                next_pos, next_row = keys[k + 1] if k + 1 < len(keys) else (
                    None, None
                )
                bridged = (
                    next_pos is not None
                    and next_pos - pos - 1 <= max_missed
                )
                if bridged:
                    end_frame = indices[next_pos]
                    end_box = samples[next_pos].xyxy[next_row]
                    last_interval = next_pos - 1
                else:  # stays put until the next sample
                    end_frame, end_box, last_interval = (
                        indices[pos], start_box, pos
                    )

                entry = (
                    indices[pos], end_frame, start_box, end_box,
                    first.class_id[row], confidence, tracker_id,
                )
                for interval in range(pos, last_interval + 1):
                    drawn[interval].append(entry)

        self._intervals = [self._pack(entries) for entries in drawn]

    @staticmethod
    def _pack(entries: list[tuple]):
        if not entries:
            return None
        first, last, start_box, end_box, class_id, confidence, tracker_id = (
            zip(*entries)
        )
        return (
            np.array(first, dtype=np.float64),
            np.array(last, dtype=np.float64),
            np.array(start_box, dtype=np.float64),
            np.array(end_box, dtype=np.float64),
            np.array(class_id, dtype=int),
            np.array(confidence, dtype=np.float32),
            np.array(tracker_id, dtype=int),
        )

    @staticmethod
    def _empty() -> sv.Detections:
        empty = sv.Detections.empty()
        empty.tracker_id = np.array([], dtype=int)
        return empty

    def at(self, index: int) -> sv.Detections:
        """The boxes to draw on frame `index`."""
        pos = bisect.bisect_right(self.indices, index) - 1
        if pos < 0 or self._intervals[pos] is None:
            return self._empty()

        first, last, start_box, end_box, class_id, confidence, tracker_id = (
            self._intervals[pos]
        )
        span = last - first
        t = np.divide(
            index - first, span, out=np.zeros_like(first), where=span > 0
        )
        t = np.clip(t, 0.0, 1.0)[:, None]
        return sv.Detections(
            xyxy=((1 - t) * start_box + t * end_box).astype(np.float32),
            confidence=confidence,
            class_id=class_id,
            tracker_id=tracker_id,
        )


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

"""Save / load raw detections as JSON.

Detections are stored before tracking, counting and drawing, so the
rendering options (colors, HUD, line, size filter, tracker settings...)
can be changed and the video re-rendered without running the detector
again (`--save-detections` / `--load-detections`).

File format (boxes are in original-video pixel coordinates):

    {
      "version": 1,
      "meta": {"detector": "rfdetr", "sample_step": 1.0,
               "fps": 30.0, "width": 1920, ...},
      "frames": {
        "0":  [{"label": "car", "confidence": 0.93,
                "box": [x1, y1, x2, y2]}, ...],
        "5":  [],
        "10": null              # detection call failed on this frame
      }
    }

Only the frames the detector actually ran on are present. Classes are
stored by name, so a file can be re-rendered with a different set of
--classes (rfdetr files contain every COCO class above the threshold).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import supervision as sv

FORMAT_VERSION = 1


def save_detections(
    path: Path,
    meta: dict,
    frames: dict[int, sv.Detections | None],
    label_of: Callable[[int], str],
) -> None:
    out: dict[str, list | None] = {}
    for index in sorted(frames):
        det = frames[index]
        if det is None:
            out[str(index)] = None
            continue
        confidences = (
            det.confidence
            if det.confidence is not None
            else np.ones(len(det), dtype=np.float32)
        )
        out[str(index)] = [
            {
                "label": label_of(int(class_id)),
                "confidence": round(float(conf), 4),
                "box": [round(float(v), 2) for v in box],
            }
            for box, conf, class_id in zip(
                det.xyxy, confidences, det.class_id
            )
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"version": FORMAT_VERSION, "meta": meta, "frames": out},
            separators=(",", ":"),
        )
    )


def load_detections(path: Path) -> tuple[dict, dict[int, list | None]]:
    """Return (meta, {frame index: list of raw detection dicts | None})."""
    try:
        data = json.loads(path.read_text())
        version = data["version"]
        meta = data["meta"]
        frames = {int(k): v for k, v in data["frames"].items()}
    except (OSError, ValueError, KeyError, AttributeError) as exc:
        raise ValueError(f"cannot read detections file {path}: {exc}")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"{path}: unsupported detections format version {version}"
        )
    return meta, frames


def to_detections(
    items: list[dict],
    name_to_id: dict[str, int],
    normalize: Callable[[str], str],
    threshold: float,
) -> sv.Detections:
    """Raw detection dicts -> sv.Detections for the classes we know.

    Labels that don't map to a known class are dropped, and so are
    detections below `threshold`.
    """
    boxes, confs, ids = [], [], []
    for item in items:
        class_id = name_to_id.get(normalize(item["label"]))
        conf = float(item.get("confidence", 1.0))
        if class_id is None or conf < threshold:
            continue
        boxes.append([float(v) for v in item["box"]])
        confs.append(conf)
        ids.append(class_id)

    if not boxes:
        return sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.empty(0, dtype=np.float32),
            class_id=np.empty(0, dtype=int),
        )
    return sv.Detections(
        xyxy=np.array(boxes, dtype=np.float32),
        confidence=np.array(confs, dtype=np.float32),
        class_id=np.array(ids, dtype=int),
    )

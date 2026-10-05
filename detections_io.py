"""Save / load raw detections as a COCO-style video JSON file.

Detections are stored before tracking, counting and drawing, so the
rendering options (colors, HUD, line, size filter, tracker settings...)
can be changed and the video re-rendered without running the detector
again (`--save-detections` / `--load-detections`).

The file follows COCO, with a video treated as a sequence of images: an
annotation belongs to an image, and an image is a frame of a video. The
images are virtual (not on disk); `file_name` only names the frame.

    {
      "info": {"description": "...", "url": "", "version": "1.0",
               "year": 2026, "contributor": "", "date_created": "...",
               "format": "coco-video", "format_version": 2,
               "detector": "rfdetr", "sample_step": 1.0, ...},
      "categories": [{"id": 3, "name": "car"}, ...],
      "videos": [{"id": 1, "file_name": "input.mp4", "fps": 30.0,
                  "width": 1920, "height": 1080, "total_frames": 300}],
      "images": [{"id": 1, "video_id": 1, "frame_id": 0,
                  "file_name": "input/000000.jpg",
                  "width": 1920, "height": 1080}, ...],
      "annotations": [{"id": 1, "image_id": 1, "category_id": 3,
                       "bbox": [x, y, width, height], "area": 1234.5,
                       "iscrowd": 0, "segmentation": [], "score": 0.93},
                      ...]
    }

- `info` holds the detection settings (detector, sample_step,
  infer_resolution, threshold, model...); the video's own properties
  are only in `videos`.
- There is one image per frame the detector ran on. An image without
  annotations is a frame where nothing was found; `"detection_failed":
  true` marks a frame whose detection call failed. Images changed by
  hand in the annotator have `"edited": true`.
- `bbox` is COCO's [x, y, width, height] in video pixels. In memory the
  boxes are [x1, y1, x2, y2] (see `DetectionsFile.frames`).
- Category ids: COCO's own ids for rfdetr files (1-90, with gaps),
  1..N for free-form classes. Categories are matched by name when
  loading, so a file can be re-rendered with a different set of
  --classes (rfdetr files contain every COCO class above the threshold).
- Annotations have no `track_id`: tracking runs when rendering.
- Only files with one video are supported.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import supervision as sv

FORMAT = "coco-video"
FORMAT_VERSION = 2


@dataclass
class DetectionsFile:
    """A detections file in memory.

    `frames` maps each frame the detector ran on to its detections,
    `{"label": name, "confidence": score, "box": [x1, y1, x2, y2]}`, or
    to None if the detection call failed on it.
    """

    info: dict
    video: dict
    categories: list[dict]
    frames: dict[int, list[dict] | None]
    edited: set[int] = field(default_factory=set)


def new_info(
    settings: dict,
    *,
    description: str = "",
    contributor: str = "",
    url: str = "",
) -> dict:
    """`info` for a new file: COCO's fields, the format, then the
    detection settings."""
    now = datetime.now().astimezone()
    return {
        "description": description,
        "url": url,
        "version": "1.0",
        "year": now.year,
        "contributor": contributor,
        "date_created": now.isoformat(timespec="seconds"),
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        **settings,
    }


def frame_file_name(video_name: str, index: int) -> str:
    return f"{Path(video_name).stem}/{index:06d}.jpg"


def _round(value: float) -> float:
    return round(float(value), 2)


def to_document(data: DetectionsFile) -> dict:
    """The JSON document for `data`. Image and annotation ids are
    renumbered; labels without a category get a new one."""
    categories = [dict(c) for c in data.categories]
    category_of = {c["name"]: c["id"] for c in categories}
    next_id = max(category_of.values(), default=0) + 1

    video = {"id": 1, **{k: v for k, v in data.video.items() if k != "id"}}
    width, height = video.get("width"), video.get("height")
    images: list[dict] = []
    annotations: list[dict] = []
    for index in sorted(data.frames):
        items = data.frames[index]
        image = {
            "id": len(images) + 1,
            "video_id": 1,
            "frame_id": index,
            "file_name": frame_file_name(video.get("file_name", "video"), index),
            "width": width,
            "height": height,
        }
        if items is None:
            image["detection_failed"] = True
        if index in data.edited:
            image["edited"] = True
        images.append(image)
        for item in items or []:
            label = item["label"]
            if label not in category_of:
                category_of[label] = next_id
                categories.append({"id": next_id, "name": label})
                next_id += 1
            x1, y1, x2, y2 = (float(v) for v in item["box"])
            w, h = _round(x2 - x1), _round(y2 - y1)
            annotations.append({
                "id": len(annotations) + 1,
                "image_id": image["id"],
                "category_id": category_of[label],
                "bbox": [_round(x1), _round(y1), w, h],
                "area": _round(w * h),
                "iscrowd": 0,
                "segmentation": [],
                "score": round(float(item.get("confidence", 1.0)), 4),
            })

    categories.sort(key=lambda c: c["id"])
    return {
        "info": data.info,
        "categories": categories,
        "videos": [video],
        "images": images,
        "annotations": annotations,
    }


def write_detections(path: Path, data: DetectionsFile) -> None:
    """Write `data` to `path` (atomically: a crash never leaves half a
    file behind)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(to_document(data), separators=(",", ":")))
    os.replace(tmp, path)


def save_detections(
    path: Path,
    info: dict,
    video: dict,
    frames: dict[int, sv.Detections | None],
    label_of: Callable[[int], str],
    category_id_of: Callable[[int], int],
    class_ids: list[int] = (),
) -> None:
    """Save detector output. `class_ids` are listed as categories even
    when nothing of that class was found."""
    out: dict[int, list | None] = {}
    seen = dict.fromkeys(class_ids)
    for index in sorted(frames):
        det = frames[index]
        if det is None:
            out[index] = None
            continue
        confidences = (
            det.confidence
            if det.confidence is not None
            else np.ones(len(det), dtype=np.float32)
        )
        out[index] = []
        for box, conf, class_id in zip(det.xyxy, confidences, det.class_id):
            seen[int(class_id)] = None
            out[index].append({
                "label": label_of(int(class_id)),
                "confidence": float(conf),
                "box": [float(v) for v in box],
            })
    categories = [
        {"id": category_id_of(c), "name": label_of(c)} for c in seen
    ]
    write_detections(path, DetectionsFile(info, video, categories, out))


def _format_error(path: Path, data: object) -> str | None:
    if not isinstance(data, dict):
        return f"{path} is not a detections file"
    info = data.get("info")
    if not isinstance(info, dict) or info.get("format") != FORMAT:
        return f"{path} is not a {FORMAT} detections file"
    if info.get("format_version") != FORMAT_VERSION:
        return (
            f"{path}: unsupported format_version "
            f"{info.get('format_version')!r} (expected {FORMAT_VERSION})"
        )
    return None


def load_detections(path: Path) -> DetectionsFile:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read detections file {path}: {exc}")
    error = _format_error(path, data)
    if error:
        raise ValueError(error)

    videos = data.get("videos")
    if not isinstance(videos, list) or len(videos) != 1:
        count = len(videos) if isinstance(videos, list) else "no"
        raise ValueError(
            f"{path}: has {count} videos, only files with one video are "
            "supported"
        )
    try:
        names = {c["id"]: c["name"] for c in data["categories"]}
        frame_of: dict[int, int] = {}
        frames: dict[int, list[dict] | None] = {}
        edited: set[int] = set()
        for image in data["images"]:
            index = int(image["frame_id"])
            frame_of[image["id"]] = index
            frames[index] = None if image.get("detection_failed") else []
            if image.get("edited"):
                edited.add(index)
        for ann in data["annotations"]:
            index = frame_of[ann["image_id"]]
            if frames[index] is None:
                frames[index] = []
            x, y, w, h = (float(v) for v in ann["bbox"])
            frames[index].append({
                "label": names[ann["category_id"]],
                "confidence": float(ann.get("score", 1.0)),
                "box": [x, y, x + w, y + h],
            })
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: malformed detections file ({exc!r})")

    return DetectionsFile(
        info=data["info"],
        video=videos[0],
        categories=list(data["categories"]),
        frames=frames,
        edited=edited,
    )


def is_detections_file(path: Path) -> bool:
    """Cheap check from the start of a JSON file: is it a detections
    file in this format?"""
    try:
        with path.open("rb") as f:
            head = f.read(4096).decode("utf-8", "ignore")
    except OSError:
        return False
    return re.search(rf'"format"\s*:\s*"{FORMAT}"', head) is not None


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

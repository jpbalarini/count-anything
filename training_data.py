"""Turn detections files into an RF-DETR training dataset.

The detections of a big / slow model (cloud, locate-anything, RF-DETR
large, or boxes fixed by hand in the annotator) become the labels for
fine-tuning a small, fast RF-DETR. Every frame a detections file has
(except the ones whose detection failed) is cut out of its video and
written as a JPEG, with its boxes, in the layout RF-DETR trains on
(Roboflow's COCO export):

    <out>/train/_annotations.coco.json   + the train images
    <out>/valid/_annotations.coco.json   + the validation images
    <out>/dataset.json                   what was built, from what

Frames a detector found nothing on are kept: they teach the model what
is not an object.

Consecutive frames of a video are nearly identical, so frames are split
into train / valid in contiguous stretches of each video rather than at
random; otherwise the validation score would mostly measure how well
the model remembers the frame next to it.

build_project_dataset does the same from a project of the annotator's
workspace (workspace.py): its labeled (and pre-labeled) images, split
in stretches of each batch.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from detections_io import DetectionsFile, load_detections
from frames import inference_size
from options import normalize_name

# The shorter side of the stored images. RF-DETR trains at 384-704 px,
# so bigger images only make loading slower.
DEFAULT_IMAGE_SIZE = 720
DEFAULT_VAL_FRACTION = 0.2
DEFAULT_MIN_SCORE = 0.5
# Each video is cut into this many stretches, which are then assigned to
# train or valid (fewer if it has fewer frames).
SPLIT_BLOCKS = 10
JPEG_QUALITY = 95
MARKER = "dataset.json"
SPLITS = ("train", "valid")


@dataclass
class Source:
    """A detections file and the video it was made from."""

    detections: Path
    video: Path
    data: DetectionsFile | None = None

    def load(self) -> DetectionsFile:
        if self.data is None:
            self.data = load_detections(self.detections)
        return self.data


def labels_in(sources: list[Source], min_score: float = 0.0) -> list[str]:
    """Every label used on a box (of at least `min_score`), in category
    order (first file first). Names that only differ in case or `_`/`-`
    are the same class."""
    names: dict[str, str] = {}
    used: set[str] = set()
    for source in sources:
        data = source.load()
        for items in data.frames.values():
            for item in items or []:
                if item.get("confidence", 1.0) >= min_score:
                    used.add(normalize_name(item["label"]))
        for category in data.categories:
            names.setdefault(normalize_name(category["name"]), category["name"])
        for items in data.frames.values():
            for item in items or []:
                names.setdefault(normalize_name(item["label"]), item["label"])
    return [name for key, name in names.items() if key in used]


def resolve_classes(
    sources: list[Source], wanted: list[str] | None, min_score: float = 0.0
) -> tuple[list[str], list[str]]:
    """(classes to train, requested classes no box has).

    Without `wanted`, every class that has a box. Boxes below
    `min_score` don't count."""
    available = labels_in(sources, min_score)
    if not wanted:
        return available, []
    by_key = {normalize_name(n): n for n in available}
    classes: dict[str, str] = {}
    missing = []
    for name in wanted:
        key = normalize_name(name)
        if key in by_key:
            classes.setdefault(key, by_key[key])
        elif name.strip():
            missing.append(name.strip())
    return list(classes.values()), missing


def split_of(position: int, count: int, val_fraction: float) -> str:
    """train / valid for the `position`-th of `count` frames of a video.

    The frames are cut into SPLIT_BLOCKS contiguous stretches (fewer for
    short videos) and `val_fraction` of them, spread over the video, go
    to valid. A video with a single frame is all train."""
    blocks = min(count, SPLIT_BLOCKS)
    if blocks < 2:
        return "train"
    block = position * blocks // count
    valid_blocks = max(1, round(blocks * val_fraction))
    # The block where the running share of validation blocks crosses a
    # whole number is a validation block.
    if (block + 1) * valid_blocks // blocks > block * valid_blocks // blocks:
        return "valid"
    return "train"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "video"


def prepare_output(out: Path) -> None:
    """Make `out` an empty dataset folder. A folder this module built
    before is replaced; anything else that isn't empty is left alone."""
    if out.exists():
        if not out.is_dir():
            raise ValueError(f"{out} exists and is not a folder")
        if any(out.iterdir()):
            if not (out / MARKER).exists():
                raise ValueError(
                    f"{out} is not empty and is not a dataset built by "
                    "this tool; choose another folder"
                )
            for split in SPLITS:
                shutil.rmtree(out / split, ignore_errors=True)
            (out / MARKER).unlink()
    out.mkdir(parents=True, exist_ok=True)


def build_dataset(
    sources: list[Source],
    out: Path,
    *,
    classes: list[str] | None = None,
    min_score: float = DEFAULT_MIN_SCORE,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    every: int = 1,
    only_edited: bool = False,
    image_size: int | None = DEFAULT_IMAGE_SIZE,
    on_progress: Callable[[int, int], None] | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Write the dataset for `sources` to `out` and return its summary
    (also saved as <out>/dataset.json).

    - `classes`: the classes to learn (default: every class with a box).
      Boxes of other classes are left out.
    - `min_score`: boxes below this confidence are left out (detectors
      without scores, and boxes drawn by hand, have 1.0).
    - `every`: use every N-th frame of each file.
    - `only_edited`: use only the frames changed in the annotator.
    - `image_size`: shorter side of the stored images (never upscaled);
      None or 0 keeps the video's size.
    """
    if not sources:
        raise ValueError("no detections files given")
    if not 0 < val_fraction <= 0.5:
        raise ValueError("the validation fraction must be in (0, 0.5]")
    if every < 1:
        raise ValueError("every must be >= 1")

    class_names, missing = resolve_classes(sources, classes, min_score)
    scored = f" with a confidence of {min_score:g} or more" if min_score > 0 else ""
    if missing:
        log(f"warning: no boxes of {', '.join(missing)}{scored} in the detections")
    if not class_names:
        raise ValueError(
            f"no boxes of the requested classes{scored} in the detections"
        )
    category_of = {normalize_name(n): i + 1 for i, n in enumerate(class_names)}

    # Which frames of which source go where.
    plan: list[tuple[Source, list[tuple[int, str]]]] = []
    skipped_failed = 0
    for source in sources:
        data = source.load()
        indices = []
        for index in sorted(data.frames):
            if data.frames[index] is None:
                skipped_failed += 1
            elif not only_edited or index in data.edited:
                indices.append(index)
        indices = indices[::every]
        plan.append((
            source,
            [
                (index, split_of(pos, len(indices), val_fraction))
                for pos, index in enumerate(indices)
            ],
        ))

    total = sum(len(frames) for _, frames in plan)
    if total == 0:
        raise ValueError(
            "no frames to train on"
            + (" (no edited frames)" if only_edited else "")
        )
    if not any(split == "valid" for _, frames in plan for _, split in frames):
        # Too few frames for a validation stretch: validate on the last one.
        source, frames = next((s, f) for s, f in reversed(plan) if f)
        frames[-1] = (frames[-1][0], "valid")
    duplicate_valid = not any(
        split == "train" for _, frames in plan for _, split in frames
    )
    if duplicate_valid:
        log("warning: a single frame: it is used to train and to validate")

    prepare_output(out)
    for split in SPLITS:
        (out / split).mkdir()

    coco = {
        split: {"images": [], "annotations": []} for split in SPLITS
    }
    per_class = {name: 0 for name in class_names}
    done = 0
    for number, (source, frames) in enumerate(plan, start=1):
        if not frames:
            continue
        data = source.load()
        wanted = dict(frames)
        cap = cv2.VideoCapture(str(source.video))
        if not cap.isOpened():
            raise ValueError(f"cannot open video {source.video}")
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            mismatch = [
                f"{key} {data.video[key]} (video: {actual})"
                for key, actual in (("width", width), ("height", height))
                if data.video.get(key) not in (None, actual)
            ]
            if mismatch:
                raise ValueError(
                    f"{source.detections} doesn't match {source.video}: "
                    + ", ".join(mismatch)
                )
            out_w, out_h = inference_size(width, height, image_size or None)
            scale = np.array([out_w / width, out_h / height] * 2)
            prefix = f"{number:02d}_{_slug(source.video.stem)}"

            last = max(wanted)
            for index in range(last + 1):
                if index not in wanted:
                    if not cap.grab():
                        break
                    continue
                ok, frame = cap.read()
                if not ok:
                    break
                if (out_w, out_h) != (width, height):
                    frame = cv2.resize(
                        frame, (out_w, out_h), interpolation=cv2.INTER_AREA
                    )
                file_name = f"{prefix}_{index:06d}.jpg"
                splits = (
                    SPLITS if duplicate_valid else (wanted[index],)
                )
                for split in splits:
                    cv2.imwrite(
                        str(out / split / file_name),
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                    )
                    _add_image(
                        coco[split], file_name, out_w, out_h,
                        data.frames[index], category_of, scale, min_score,
                        per_class if split == splits[0] else None,
                        class_names,
                    )
                done += 1
                if on_progress is not None:
                    on_progress(done, total)
        finally:
            cap.release()
        if done < sum(len(f) for _, f in plan[:number]):
            log(
                f"warning: {source.video} ended before frame {max(wanted)}; "
                "the frames after its end were skipped"
            )

    if not any(per_class.values()):
        raise ValueError(
            "none of the chosen frames has a box to learn from"
            + (" (only edited frames were used)" if only_edited else "")
        )
    empty = [name for name, n in per_class.items() if n == 0]
    if empty:
        log(f"warning: no boxes of {', '.join(empty)} in the chosen frames")

    now = _write_splits(out, coco, class_names, "detections")
    summary = {
        "created": now,
        "classes": class_names,
        "sources": [
            {"detections": str(s.detections), "video": str(s.video),
             "detector": s.load().info.get("detector"),
             "frames": len(frames)}
            for s, frames in plan
        ],
        "images": {s: len(coco[s]["images"]) for s in SPLITS},
        "boxes": {s: len(coco[s]["annotations"]) for s in SPLITS},
        "boxes_per_class": per_class,
        "skipped_failed_frames": skipped_failed,
        "options": {
            "min_score": min_score,
            "val_fraction": val_fraction,
            "every": every,
            "only_edited": only_edited,
            "image_size": image_size or None,
        },
    }
    (out / MARKER).write_text(json.dumps(summary, indent=2))
    return summary


def _write_splits(
    out: Path, coco: dict, class_names: list[str], built_from: str
) -> str:
    """Write each split's _annotations.coco.json; returns the time."""
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    categories = [
        {"id": i + 1, "name": name, "supercategory": "none"}
        for i, name in enumerate(class_names)
    ]
    for split in SPLITS:
        document = {
            "info": {
                "description": f"{split} split, built from {built_from}",
                "date_created": now,
            },
            "licenses": [],
            "categories": categories,
            **coco[split],
        }
        (out / split / "_annotations.coco.json").write_text(
            json.dumps(document, separators=(",", ":"))
        )
    return now


def build_project_dataset(
    project_dir: Path,
    out: Path,
    *,
    classes: list[str] | None = None,
    min_score: float = DEFAULT_MIN_SCORE,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    every: int = 1,
    only_labeled: bool = False,
    image_size: int | None = DEFAULT_IMAGE_SIZE,
    on_progress: Callable[[int, int], None] | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Like build_dataset, from a workspace project (workspace.py).

    - Unlabeled images are left out (nothing says what is on them).
      Labeled images are used with all their boxes; predicted ones
      (boxes from a model, not reviewed) only with the boxes of at
      least `min_score`, and not at all with `only_labeled`.
    - Each batch is split into train / valid in contiguous stretches,
      as the frames of a video are by build_dataset.
    """
    from workspace import load_dataset_images

    if not 0 < val_fraction <= 0.5:
        raise ValueError("the validation fraction must be in (0, 0.5]")
    if every < 1:
        raise ValueError("every must be >= 1")
    project_classes, batches, images = load_dataset_images(project_dir)
    wanted_status = {"labeled"} if only_labeled else {"labeled", "predicted"}
    usable = [im for im in images if im.get("status") in wanted_status]

    def kept(image: dict) -> list[dict]:
        if image["status"] == "labeled":
            return image["boxes"]
        return [b for b in image["boxes"] if b.get("confidence", 1.0) >= min_score]

    # Classes: the requested ones, or every class with a box to learn.
    used: dict[str, str] = {}
    for name in project_classes:
        used.setdefault(normalize_name(name), name)
    counts: dict[str, int] = {}
    for image in usable:
        for item in kept(image):
            key = normalize_name(item["label"])
            used.setdefault(key, item["label"])
            counts[key] = counts.get(key, 0) + 1
    if classes:
        class_names, missing = [], []
        for name in classes:
            key = normalize_name(name)
            if counts.get(key):
                if used[key] not in class_names:
                    class_names.append(used[key])
            elif name.strip():
                missing.append(name.strip())
        if missing:
            log(f"warning: no boxes of {', '.join(missing)} on the images to train on")
    else:
        class_names = [name for key, name in used.items() if counts.get(key)]
    if not class_names:
        raise ValueError(
            "no boxes to learn from"
            + (" on the labeled images" if only_labeled else "")
            + "; label some images first"
        )
    category_of = {normalize_name(n): i + 1 for i, n in enumerate(class_names)}

    # Which image goes where: each batch split in stretches.
    plan: list[tuple[dict, str]] = []
    by_batch: dict[str, list[dict]] = {}
    for image in usable:
        by_batch.setdefault(image.get("batch_id"), []).append(image)
    for batch_images in by_batch.values():
        batch_images = batch_images[::every]
        plan += [
            (image, split_of(pos, len(batch_images), val_fraction))
            for pos, image in enumerate(batch_images)
        ]
    if not plan:
        raise ValueError("no images to train on")
    if not any(split == "valid" for _, split in plan):
        plan[-1] = (plan[-1][0], "valid")
    duplicate_valid = not any(split == "train" for _, split in plan)
    if duplicate_valid:
        log("warning: a single image: it is used to train and to validate")

    prepare_output(out)
    for split in SPLITS:
        (out / split).mkdir()
    coco = {split: {"images": [], "annotations": []} for split in SPLITS}
    per_class = {name: 0 for name in class_names}
    names_by_batch = {b["id"]: b.get("name") for b in batches}
    for done, (image, split) in enumerate(plan, start=1):
        frame = cv2.imread(str(image["path"]))
        if frame is None:
            log(f"warning: cannot read {image['path']}, skipped")
            continue
        height, width = frame.shape[:2]
        out_w, out_h = inference_size(width, height, image_size or None)
        if (out_w, out_h) != (width, height):
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        scale = np.array([out_w / width, out_h / height] * 2)
        file_name = f"{image['id']:06d}_{_slug(Path(image['file_name']).stem)}.jpg"
        splits = SPLITS if duplicate_valid else (split,)
        for target in splits:
            cv2.imwrite(
                str(out / target / file_name), frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )
            _add_image(
                coco[target], file_name, out_w, out_h,
                [dict(b, confidence=1.0) for b in kept(image)],
                category_of, scale, 0.0,
                per_class if target == splits[0] else None,
                class_names,
            )
        if on_progress is not None:
            on_progress(done, len(plan))

    empty = [name for name, n in per_class.items() if n == 0]
    if empty:
        log(f"warning: no boxes of {', '.join(empty)} in the chosen images")
    now = _write_splits(out, coco, class_names, "a project")
    status_counts: dict[str, int] = {}
    for image, _ in plan:
        status_counts[image["status"]] = status_counts.get(image["status"], 0) + 1
    summary = {
        "created": now,
        "classes": class_names,
        "project": str(project_dir),
        "sources": [
            {"batch": batch_id, "name": names_by_batch.get(batch_id),
             "frames": len(items[::every])}
            for batch_id, items in by_batch.items()
        ],
        "status": status_counts,
        "images": {s: len(coco[s]["images"]) for s in SPLITS},
        "boxes": {s: len(coco[s]["annotations"]) for s in SPLITS},
        "boxes_per_class": per_class,
        "options": {
            "min_score": min_score,
            "val_fraction": val_fraction,
            "every": every,
            "only_labeled": only_labeled,
            "image_size": image_size or None,
        },
    }
    (out / MARKER).write_text(json.dumps(summary, indent=2))
    return summary


def _add_image(
    split: dict,
    file_name: str,
    width: int,
    height: int,
    items: list[dict],
    category_of: dict[str, int],
    scale: np.ndarray,
    min_score: float,
    per_class: dict[str, int] | None,
    class_names: list[str],
) -> None:
    image_id = len(split["images"]) + 1
    split["images"].append({
        "id": image_id, "file_name": file_name,
        "width": width, "height": height,
    })
    for item in items:
        category = category_of.get(normalize_name(item["label"]))
        if category is None or item.get("confidence", 1.0) < min_score:
            continue
        x1, y1, x2, y2 = np.array(item["box"], dtype=float) * scale
        x1, x2 = np.clip([x1, x2], 0, width)
        y1, y2 = np.clip([y1, y2], 0, height)
        w, h = x2 - x1, y2 - y1
        if w < 1 or h < 1:
            continue
        split["annotations"].append({
            "id": len(split["annotations"]) + 1,
            "image_id": image_id,
            "category_id": category,
            "bbox": [round(float(v), 2) for v in (x1, y1, w, h)],
            "area": round(float(w * h), 2),
            "iscrowd": 0,
        })
        if per_class is not None:
            per_class[class_names[category - 1]] += 1

"""Projects and folders: the annotator's workspace on disk.

A folder holds folders and projects; a project holds a dataset of
images (uploaded pictures, and frames sampled from uploaded videos),
their boxes, and the models trained on them. Everything is plain files
under one directory, the projects location (see workspace_location):

    workspace/
      hand-tracking/                 a folder
        folder.json                  {"id", "name", "created", ...}
        eagerhands/                  a project
          project.json               {"id", "name", "type", "stats", ...}
          annotations.json           the dataset (COCO, see below)
          images/<batch id>/...      the images
          thumbs/<batch id>/...      their thumbnails (made on demand)
          models/<name>/             models trained on it (train_rfdetr.py)

Folders and projects are found by their folder.json / project.json, so
directories can be renamed or moved by hand; their `id` is what the UI
uses.

annotations.json is a COCO detection file, plus:

- `batches`: one per upload (a video, or a set of pictures):
  {"id", "name", "kind": "video" | "images", "created", "state",
   "source": {...}}. `state` is "ready", or "processing" while frames
  are being sampled from a video ("interrupted" if the server stopped
  in the middle; the frames sampled so far are kept).
- per image: `batch_id`, `status`, `tags` (free text, e.g. "one hand",
  "blurry"; optional) and, for video frames, `frame_id`.
  `status` is "unlabeled" (no boxes yet), "predicted" (boxes from a
  model, not reviewed) or "labeled" (saved or approved by hand).
- per annotation: `score` (1.0 for boxes drawn by hand).

`file_name` is relative to the project folder.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from options import normalize_name

# Where projects used to be kept: <root>/workspace, next to the code.
WORKSPACE_DIR = "workspace"
# The projects location picked in the UI is saved here.
CONFIG_FILE = (
    Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    / "ai-studio" / "config.json"
)
# The projects location until one is picked.
DEFAULT_LOCATION = Path.home() / ".count-anything"
FOLDER_FILE = "folder.json"
PROJECT_FILE = "project.json"
DATASET_FILE = "annotations.json"
DATASET_FORMAT = "studio-dataset"
IMAGES_DIR = "images"
THUMBS_DIR = "thumbs"
MODELS_DIR = "models"

STATUSES = ("unlabeled", "predicted", "labeled")
MAX_TAG_LENGTH = 60
PROJECT_TYPES = {"object-detection": "Object Detection"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

JPEG_QUALITY = 92
THUMB_WIDTH = 240
THUMB_QUALITY = 75
# A video batch saves the dataset every this many sampled frames, so
# the project page shows them coming in.
SAVE_EVERY = 25
# Edits are written at most this often (seconds): a big dataset takes a
# moment to write, and the annotator saves on every change.
SAVE_DELAY = 0.5


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex[:10]


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "untitled"


def unique_child(parent: Path, name: str) -> Path:
    """parent/name, or parent/name-2, -3... if that exists."""
    path = parent / name
    n = 2
    while path.exists():
        path = parent / f"{name}-{n}"
        n += 1
    return path


def read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_json(path: Path, data: dict, indent: int | None = 2) -> None:
    """Write atomically: a crash never leaves half a file behind."""
    tmp = path.with_name(f".{path.name}.tmp")
    separators = None if indent else (",", ":")
    tmp.write_text(json.dumps(data, indent=indent, separators=separators))
    os.replace(tmp, path)


def saved_location() -> Path | None:
    """The projects location picked in the UI, if any."""
    path = (read_json(CONFIG_FILE) or {}).get("workspace")
    return Path(path).expanduser() if isinstance(path, str) and path else None


def save_location(path: Path) -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    config = read_json(CONFIG_FILE) or {}
    config["workspace"] = str(path)
    write_json(CONFIG_FILE, config)


def workspace_location(root: Path) -> tuple[Path, str]:
    """Where the projects are (without --workspace), and why:
    "settings" (picked in the UI), "legacy" (<root>/workspace, which
    already has folders or projects, until another location is picked)
    or "default" (DEFAULT_LOCATION)."""
    saved = saved_location()
    if saved is not None:
        return saved, "settings"
    legacy = root / WORKSPACE_DIR
    try:
        if any(c.is_dir() and not c.name.startswith(".") for c in legacy.iterdir()):
            return legacy, "legacy"
    except OSError:
        pass
    return DEFAULT_LOCATION, "default"


def encode_jpeg(frame: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def make_thumbnail(frame: np.ndarray) -> bytes:
    height, width = frame.shape[:2]
    size = (THUMB_WIDTH, max(1, round(height * THUMB_WIDTH / width)))
    small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return encode_jpeg(small, THUMB_QUALITY)


def decode_image(data: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def media_type(path: Path) -> str:
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".bmp": "image/bmp", ".webp": "image/webp", ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }.get(path.suffix.lower(), "application/octet-stream")


# -- Project ------------------------------------------------------------


class Project:
    """A project's metadata and dataset, held in memory and written on
    every change. `lock` guards both."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.RLock()
        meta = read_json(path / PROJECT_FILE)
        if meta is None or not meta.get("id"):
            raise ValueError(f"{path} is not a project")
        self.meta = meta
        self._timer: threading.Timer | None = None
        self._load()

    # -- Loading / saving

    def _load(self) -> None:
        doc = read_json(self.path / DATASET_FILE) or {}
        self.info = doc.get("info") or {
            "description": f"{self.meta.get('name', '')} dataset",
            "date_created": now_iso(),
            "format": DATASET_FORMAT,
            "format_version": 1,
        }
        self.categories: list[dict] = list(doc.get("categories") or [])
        self.batches: list[dict] = list(doc.get("batches") or [])
        names = {c["id"]: c["name"] for c in self.categories}
        self.images: dict[int, dict] = {}
        for image in doc.get("images") or []:
            record = {k: v for k, v in image.items()}
            record["boxes"] = []
            record.setdefault("status", "unlabeled")
            self.images[int(image["id"])] = record
        for ann in doc.get("annotations") or []:
            record = self.images.get(int(ann["image_id"]))
            label = names.get(ann.get("category_id"))
            if record is None or label is None:
                continue
            x, y, w, h = (float(v) for v in ann["bbox"])
            record["boxes"].append({
                "label": label,
                "confidence": float(ann.get("score", 1.0)),
                "box": [x, y, x + w, y + h],
            })
        # A sampling run the server didn't finish (it was stopped).
        for batch in self.batches:
            if batch.get("state") == "processing":
                batch["state"] = "interrupted"

    def save_soon(self) -> None:
        """Save within SAVE_DELAY seconds (one write for many edits)."""
        with self.lock:
            if self._timer is None:
                self._timer = threading.Timer(SAVE_DELAY, self.flush)
                self._timer.daemon = True
                self._timer.start()

    def discard(self) -> None:
        """Drop a pending save_soon() (the project is being deleted)."""
        with self.lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

    def flush(self) -> None:
        """Write a pending save_soon() now."""
        with self.lock:
            if self._timer is not None:
                self.save()

    def save(self) -> None:
        with self.lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            categories = [dict(c) for c in self.categories]
            category_of = {c["name"]: c["id"] for c in categories}
            images, annotations = [], []
            for image_id in sorted(self.images):
                record = self.images[image_id]
                images.append({k: v for k, v in record.items() if k != "boxes"})
                for item in record["boxes"]:
                    x1, y1, x2, y2 = (float(v) for v in item["box"])
                    w, h = round(x2 - x1, 2), round(y2 - y1, 2)
                    annotations.append({
                        "id": len(annotations) + 1,
                        "image_id": image_id,
                        "category_id": category_of[item["label"]],
                        "bbox": [round(x1, 2), round(y1, 2), w, h],
                        "area": round(w * h, 2),
                        "iscrowd": 0,
                        "segmentation": [],
                        "score": round(float(item.get("confidence", 1.0)), 4),
                    })
            write_json(self.path / DATASET_FILE, {
                "info": self.info,
                "categories": categories,
                "batches": self.batches,
                "images": images,
                "annotations": annotations,
            }, indent=None)
            self.meta["updated"] = now_iso()
            self.meta["stats"] = self.stats()
            write_json(self.path / PROJECT_FILE, self.meta)

    def save_meta(self) -> None:
        with self.lock:
            write_json(self.path / PROJECT_FILE, self.meta)

    # -- Summaries

    @property
    def id(self) -> str:
        return self.meta["id"]

    @property
    def name(self) -> str:
        return self.meta.get("name", self.path.name)

    def stats(self) -> dict:
        """What the project cards show (also kept in project.json, so the
        home page doesn't load every dataset)."""
        with self.lock:
            counts = {s: 0 for s in STATUSES}
            for record in self.images.values():
                counts[record.get("status", "unlabeled")] += 1
            cover = next(
                (i for i in sorted(self.images) if self.images[i]["boxes"]),
                min(self.images, default=None),
            )
            return {
                "images": len(self.images),
                **counts,
                "batches": len(self.batches),
                "classes": len(self.categories),
                "cover": cover,
            }

    def classes(self) -> list[str]:
        return [c["name"] for c in self.categories]

    def batch(self, batch_id: str) -> dict:
        for batch in self.batches:
            if batch["id"] == batch_id:
                return batch
        raise KeyError(batch_id)

    def batch_images(self, batch_id: str | None) -> list[int]:
        """Image ids of a batch (all images if None), in upload / frame
        order."""
        with self.lock:
            order = {b["id"]: n for n, b in enumerate(self.batches)}
            ids = [
                i for i, r in self.images.items()
                if batch_id is None or r.get("batch_id") == batch_id
            ]
            return sorted(
                ids,
                key=lambda i: (
                    order.get(self.images[i].get("batch_id"), len(order)),
                    self.images[i].get("frame_id", 0),
                    i,
                ),
            )

    def batch_summary(self, batch: dict) -> dict:
        with self.lock:
            counts = {s: 0 for s in STATUSES}
            cover = None
            boxes = 0
            for image_id in self.batch_images(batch["id"]):
                record = self.images[image_id]
                counts[record.get("status", "unlabeled")] += 1
                boxes += len(record["boxes"])
                if cover is None:
                    cover = image_id
            return {
                **batch,
                "images": sum(counts.values()),
                **counts,
                "boxes": boxes,
                "cover": cover,
            }

    def class_counts(self) -> dict[str, int]:
        with self.lock:
            counts = {name: 0 for name in self.classes()}
            for record in self.images.values():
                for item in record["boxes"]:
                    counts[item["label"]] = counts.get(item["label"], 0) + 1
            return counts

    # -- Tags (free-text notes on images, to filter the dataset by)

    def tag_counts(self) -> dict[str, int]:
        """Every tag with its number of images, most used first."""
        with self.lock:
            counts: dict[str, int] = {}
            for record in self.images.values():
                for tag in record.get("tags") or []:
                    counts[tag] = counts.get(tag, 0) + 1
            return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].lower())))

    def _canonical_tag(self, tag: str, known: dict[str, str]) -> str | None:
        """`tag` cleaned up and spelled like the same tag already used
        (case and `_`/`-` don't matter), or None if empty."""
        tag = " ".join(tag.split())[:MAX_TAG_LENGTH]
        if not tag:
            return None
        return known.setdefault(normalize_name(tag), tag)

    def _known_tags(self) -> dict[str, str]:
        return {normalize_name(t): t for t in self.tag_counts()}

    def set_tags(self, image_id: int, tags: list[str]) -> list[str]:
        """Replace an image's tags (its status doesn't change)."""
        with self.lock:
            record = self.images[image_id]
            known = self._known_tags()
            cleaned: list[str] = []
            for tag in tags:
                tag = self._canonical_tag(tag, known)
                if tag and tag not in cleaned:
                    cleaned.append(tag)
            if cleaned:
                record["tags"] = cleaned
            else:
                record.pop("tags", None)
            self.save_soon()
            return cleaned

    def edit_tags(self, image_ids: list[int], add: list[str], remove: list[str]) -> int:
        """Add / remove tags on many images; returns how many changed."""
        with self.lock:
            known = self._known_tags()
            add = [t for t in (self._canonical_tag(t, known) for t in add) if t]
            gone = {normalize_name(t) for t in remove}
            changed = 0
            for image_id in image_ids:
                record = self.images.get(image_id)
                if record is None:
                    continue
                before = list(record.get("tags") or [])
                after = [t for t in before if normalize_name(t) not in gone]
                after += [t for t in add if t not in after]
                if after != before:
                    changed += 1
                    if after:
                        record["tags"] = after
                    else:
                        record.pop("tags", None)
            if changed:
                self.save_soon()
            return changed

    # -- Paths

    def image_path(self, image_id: int) -> Path:
        return self.path / self.images[image_id]["file_name"]

    def thumb_path(self, image_id: int) -> Path:
        name = Path(self.images[image_id]["file_name"])
        return self.path / THUMBS_DIR / name.parent.name / f"{name.stem}.jpg"

    def thumbnail(self, image_id: int) -> bytes:
        """The image's thumbnail (made and kept the first time)."""
        path = self.thumb_path(image_id)
        try:
            return path.read_bytes()
        except OSError:
            pass
        frame = cv2.imread(str(self.image_path(image_id)))
        if frame is None:
            raise KeyError(image_id)
        data = make_thumbnail(frame)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return data

    @property
    def models_dir(self) -> Path:
        return self.path / MODELS_DIR

    # -- Classes and boxes

    def _canonical(self, label: str) -> str:
        """`label` as spelled by its category, adding the category if
        it's new."""
        label = label.strip()
        key = normalize_name(label)
        for category in self.categories:
            if normalize_name(category["name"]) == key:
                return category["name"]
        new_id = max((c["id"] for c in self.categories), default=0) + 1
        self.categories.append({"id": new_id, "name": label})
        return label

    def _clean(self, item: dict, width: int, height: int) -> dict:
        x1, y1, x2, y2 = (float(v) for v in item["box"])
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        x1, x2 = np.clip([x1, x2], 0, width)
        y1, y2 = np.clip([y1, y2], 0, height)
        return {
            "label": self._canonical(item["label"]),
            "confidence": round(float(item.get("confidence", 1.0)), 4),
            "box": [round(float(v), 2) for v in (x1, y1, x2, y2)],
        }

    def set_boxes(
        self, image_id: int, boxes: list[dict], status: str = "labeled"
    ) -> list[dict]:
        with self.lock:
            record = self.images[image_id]
            cleaned = [
                self._clean(b, record["width"], record["height"]) for b in boxes
            ]
            record["boxes"] = cleaned
            record["status"] = status
            self.save_soon()
            return cleaned

    def set_status(self, image_ids: list[int], status: str) -> int:
        if status not in STATUSES:
            raise ValueError(f"unknown status {status}")
        with self.lock:
            changed = 0
            for image_id in image_ids:
                record = self.images.get(image_id)
                if record is not None and record.get("status") != status:
                    record["status"] = status
                    changed += 1
            if changed:
                self.save_soon()
            return changed

    def set_classes(self, classes: list[str]) -> None:
        """Make the categories `classes`. Categories still used on a box
        are kept (with their ids)."""
        with self.lock:
            wanted = {normalize_name(c) for c in classes if c.strip()}
            used = {
                normalize_name(item["label"])
                for record in self.images.values()
                for item in record["boxes"]
            }
            self.categories = [
                c for c in self.categories
                if normalize_name(c["name"]) in wanted | used
            ]
            for name in classes:
                if name.strip():
                    self._canonical(name)
            self.save()

    def rename_class(self, old: str, new: str) -> None:
        """Rename a class; into an existing one, merge them."""
        new = new.strip()
        if not new:
            raise ValueError("the new name is empty")
        with self.lock:
            source = next(
                (c for c in self.categories
                 if normalize_name(c["name"]) == normalize_name(old)),
                None,
            )
            if source is None:
                raise KeyError(old)
            target = next(
                (c for c in self.categories
                 if c is not source
                 and normalize_name(c["name"]) == normalize_name(new)),
                None,
            )
            if target is None:
                renamed = new
                source["name"] = new
            else:
                renamed = target["name"]
                self.categories.remove(source)
            for record in self.images.values():
                for item in record["boxes"]:
                    if normalize_name(item["label"]) == normalize_name(old):
                        item["label"] = renamed
            self.save()

    def delete_class(self, name: str) -> int:
        """Remove a class and its boxes; returns how many boxes."""
        key = normalize_name(name)
        with self.lock:
            removed = 0
            for record in self.images.values():
                kept = [b for b in record["boxes"] if normalize_name(b["label"]) != key]
                removed += len(record["boxes"]) - len(kept)
                record["boxes"] = kept
            self.categories = [
                c for c in self.categories if normalize_name(c["name"]) != key
            ]
            self.save()
            return removed

    # -- Batches and images

    def add_batch(self, name: str, kind: str, source: dict | None = None,
                  state: str = "ready") -> dict:
        batch = {
            "id": new_id(),
            "name": name.strip() or "Upload",
            "kind": kind,
            "created": now_iso(),
            "state": state,
            "source": source or {},
        }
        with self.lock:
            self.batches.append(batch)
            (self.path / IMAGES_DIR / batch["id"]).mkdir(parents=True, exist_ok=True)
            self.save()
        return batch

    def add_image(
        self,
        batch_id: str,
        name: str,
        frame: np.ndarray,
        data: bytes | None = None,
        *,
        frame_id: int | None = None,
        boxes: list[dict] | None = None,
        status: str | None = None,
        save: bool = True,
    ) -> int:
        """Add an image to a batch. `data` is the file as uploaded (kept
        as is); without it `frame` is stored as JPEG. Returns its id.
        `save`: write the dataset soon (else the caller saves)."""
        folder = self.path / IMAGES_DIR / batch_id
        folder.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).stem).strip("_") or "image"
        suffix = Path(name).suffix.lower() if data is not None else ".jpg"
        with self.lock:
            path = folder / f"{stem}{suffix}"
            n = 2
            while path.exists():
                path = folder / f"{stem}-{n}{suffix}"
                n += 1
            path.write_bytes(data if data is not None else encode_jpeg(frame, JPEG_QUALITY))
            thumb = self.path / THUMBS_DIR / batch_id / f"{path.stem}.jpg"
            thumb.parent.mkdir(parents=True, exist_ok=True)
            thumb.write_bytes(make_thumbnail(frame))

            height, width = frame.shape[:2]
            # Ids are never reused, so a cached thumbnail can't show
            # another image.
            image_id = max(max(self.images, default=0), self.info.get("last_image_id", 0)) + 1
            self.info["last_image_id"] = image_id
            record = {
                "id": image_id,
                "file_name": str(path.relative_to(self.path)),
                "width": int(width),
                "height": int(height),
                "batch_id": batch_id,
                "status": "unlabeled",
                "added": now_iso(),
                "boxes": [],
            }
            if frame_id is not None:
                record["frame_id"] = int(frame_id)
            self.images[image_id] = record
            if boxes:
                record["boxes"] = [self._clean(b, width, height) for b in boxes]
            record["status"] = status or ("predicted" if boxes else "unlabeled")
            if save:
                self.save_soon()
            return image_id

    def delete_images(self, image_ids: list[int]) -> int:
        with self.lock:
            removed = 0
            for image_id in image_ids:
                if image_id not in self.images:
                    continue
                self.image_path(image_id).unlink(missing_ok=True)
                self.thumb_path(image_id).unlink(missing_ok=True)
                del self.images[image_id]
                removed += 1
            if removed:
                self.save()
            return removed

    def delete_batch(self, batch_id: str) -> None:
        with self.lock:
            batch = self.batch(batch_id)
            for image_id in self.batch_images(batch_id):
                del self.images[image_id]
            self.batches.remove(batch)
            shutil.rmtree(self.path / IMAGES_DIR / batch_id, ignore_errors=True)
            shutil.rmtree(self.path / THUMBS_DIR / batch_id, ignore_errors=True)
            self.save()

    def update_batch(self, batch_id: str, **fields) -> dict:
        with self.lock:
            batch = self.batch(batch_id)
            batch.update(fields)
            self.save()
            return batch


def sample_indices(total: int, fps: float, rate: float | None) -> list[int]:
    """The frames to keep from a video: `rate` per second, or every
    frame if None (or the video is slower than that)."""
    if total <= 0:
        return []
    step = 1.0 if not rate or rate >= fps else fps / rate
    indices, at = [], 0.0
    while round(at) < total:
        if not indices or round(at) != indices[-1]:
            indices.append(round(at))
        at += step
    return indices


def video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot open video {path.name}")
        return {
            "fps": float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        cap.release()


def sample_video(
    project: Project,
    batch_id: str,
    video: Path,
    indices: list[int],
    *,
    name: str | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    stop: threading.Event | None = None,
) -> int:
    """Add frames `indices` of `video` to a batch, in decode order, as
    <name>_<frame>.jpg (`name`: the video's, by default). Returns how
    many frames were added."""
    stem = Path(name or video.name).stem
    wanted = set(indices)
    if not wanted:
        return 0
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video.name}")
    added = 0
    try:
        for index in range(max(wanted) + 1):
            if stop is not None and stop.is_set():
                break
            if index not in wanted:
                if not cap.grab():
                    break
                continue
            ok, frame = cap.read()
            if not ok:
                break
            project.add_image(
                batch_id,
                f"{stem}_{index:06d}.jpg",
                frame,
                frame_id=index,
                save=False,
            )
            added += 1
            if added % SAVE_EVERY == 0:
                project.save()
            if on_progress is not None:
                on_progress(added, len(wanted))
    finally:
        cap.release()
        project.save()
    return added


# -- Workspace ------------------------------------------------------------


@dataclass
class Node:
    kind: str  # "folder" | "project"
    id: str
    path: Path
    parent: str | None
    meta: dict


class Workspace:
    def __init__(self, path: Path):
        self.path = path
        path.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self._projects: dict[str, Project] = {}
        # Edits waiting for their save_soon() are written on the way out.
        atexit.register(self.flush)

    def flush(self) -> None:
        for project in self.loaded():
            project.flush()

    def nodes(self) -> dict[str, Node]:
        """Every folder and project, by id. Projects aren't searched
        (they hold no folders, and many images)."""
        found: dict[str, Node] = {}

        def walk(directory: Path, parent: str | None) -> None:
            try:
                children = sorted(directory.iterdir())
            except OSError:
                return
            for child in children:
                if not child.is_dir() or child.name.startswith("."):
                    continue
                for kind, file_name in (("folder", FOLDER_FILE), ("project", PROJECT_FILE)):
                    meta = read_json(child / file_name)
                    if meta is None or not meta.get("id") or meta["id"] in found:
                        continue
                    found[meta["id"]] = Node(kind, meta["id"], child, parent, meta)
                    if kind == "folder":
                        walk(child, meta["id"])
                    break

        with self.lock:
            walk(self.path, None)
        return found

    def node(self, node_id: str) -> Node:
        node = self.nodes().get(node_id)
        if node is None:
            raise KeyError(node_id)
        return node

    def folder_path(self, folder_id: str | None) -> Path:
        if not folder_id:
            return self.path
        node = self.node(folder_id)
        if node.kind != "folder":
            raise ValueError("not a folder")
        return node.path

    def project(self, project_id: str) -> Project:
        with self.lock:
            project = self._projects.get(project_id)
            node = self.nodes().get(project_id)
            if node is None or node.kind != "project":
                self._projects.pop(project_id, None)
                raise KeyError(project_id)
            if project is None:
                project = Project(node.path)
                self._projects[project_id] = project
            else:
                # Moved / renamed by hand since it was loaded.
                project.path = node.path
            return project

    def loaded(self) -> list[Project]:
        with self.lock:
            return list(self._projects.values())

    def create_folder(self, name: str, parent: str | None) -> Node:
        return self._create("folder", name, parent, {})

    def create_project(self, name: str, parent: str | None,
                       type_: str = "object-detection") -> Node:
        if type_ not in PROJECT_TYPES:
            raise ValueError(f"unknown project type {type_}")
        node = self._create("project", name, parent, {"type": type_})
        Project(node.path).save()
        return node

    def _create(self, kind: str, name: str, parent: str | None, extra: dict) -> Node:
        name = name.strip()
        if not name:
            raise ValueError("the name is empty")
        with self.lock:
            directory = unique_child(self.folder_path(parent), slug(name))
            directory.mkdir(parents=True)
            stamp = now_iso()
            meta = {"id": new_id(), "name": name, **extra,
                    "created": stamp, "updated": stamp}
            write_json(directory / (FOLDER_FILE if kind == "folder" else PROJECT_FILE), meta)
            return Node(kind, meta["id"], directory, parent, meta)

    def rename(self, node_id: str, name: str) -> None:
        name = name.strip()
        if not name:
            raise ValueError("the name is empty")
        with self.lock:
            node = self.node(node_id)
            if node.kind == "project":
                project = self.project(node_id)
                with project.lock:
                    project.meta["name"] = name
                    project.save_meta()
            else:
                node.meta["name"] = name
                write_json(node.path / FOLDER_FILE, node.meta)
            # The directory follows the name when it can.
            target = node.path.parent / slug(name)
            if target != node.path and not target.exists():
                self._move_dir(node, target)

    def move(self, node_id: str, parent: str | None) -> None:
        with self.lock:
            nodes = self.nodes()
            node = nodes.get(node_id)
            if node is None:
                raise KeyError(node_id)
            if parent:
                # Not into itself or one of its own folders.
                at = parent
                while at:
                    if at == node_id:
                        raise ValueError("a folder can't go inside itself")
                    at = nodes[at].parent if at in nodes else None
            if (node.parent or None) == (parent or None):
                return
            destination = self.folder_path(parent)
            self._move_dir(node, unique_child(destination, node.path.name))

    def relocate(self, path: Path, move: bool) -> int:
        """Use the folder `path` (made if missing) from now on. With
        `move`, the folders and projects of the current one go there
        too (renamed if the name is taken), and the current folder is
        removed if that empties it; without, projects loaded from the
        current folder are forgotten. Returns how many were moved."""
        path = Path(os.path.abspath(path.expanduser()))
        with self.lock:
            if path == self.path:
                return 0
            if path.is_relative_to(self.path):
                raise ValueError("the new location can't be inside the current one")
            if path.exists() and not path.is_dir():
                raise ValueError(f"{path} is a file")
            self.flush()
            path.mkdir(parents=True, exist_ok=True)
            moved = 0
            if move:
                top = [n for n in self.nodes().values() if n.parent is None]
                for node in top:
                    try:
                        self._move_dir(node, unique_child(path, node.path.name), shutil.move)
                    except OSError as exc:
                        raise ValueError(
                            f"moved {moved} of {len(top)}, then {node.path.name} "
                            f"failed ({exc}); the rest are still in {self.path}"
                        )
                    moved += 1
                try:
                    self.path.rmdir()
                except OSError:
                    pass  # Not empty: other files were kept there.
            else:
                self._projects.clear()
            self.path = path
            return moved

    def _move_dir(self, node: Node, target: Path, mover=os.replace) -> None:
        mover(node.path, target)
        # Loaded projects (open in the annotator) follow.
        for project in self._projects.values():
            try:
                project.path = target / project.path.relative_to(node.path)
            except ValueError:
                pass

    def delete(self, node_id: str) -> None:
        with self.lock:
            node = self.node(node_id)
            for project in self._projects.values():
                if project.path.is_relative_to(node.path):
                    project.discard()
            shutil.rmtree(node.path)
            for project_id, project in list(self._projects.items()):
                if not project.path.exists():
                    del self._projects[project_id]

    def project_ids_under(self, folder_id: str) -> list[str]:
        nodes = self.nodes()
        found = []
        for node in nodes.values():
            if node.kind != "project":
                continue
            at = node.parent
            while at:
                if at == folder_id:
                    found.append(node.id)
                    break
                at = nodes[at].parent if at in nodes else None
        return found

    def breadcrumbs(self, folder_id: str | None) -> list[dict]:
        nodes = self.nodes()
        trail = []
        at = folder_id
        while at and at in nodes:
            trail.append({"id": at, "name": nodes[at].meta.get("name")})
            at = nodes[at].parent
        return trail[::-1]


def load_dataset_images(project_dir: Path) -> tuple[list[str], list[dict], list[dict]]:
    """(classes, batches, images) of a project folder, for training. Each
    image: its record plus `path` and `boxes` ([x1, y1, x2, y2] items)."""
    project = Project(project_dir)
    images = []
    for image_id in project.batch_images(None):
        record = dict(project.images[image_id])
        record["path"] = project.image_path(image_id)
        images.append(record)
    return project.classes(), project.batches, images

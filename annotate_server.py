"""Web UI to review and fix saved detections by hand.

Serves the frames of a video next to a detections file written by
`process_video.py --save-detections`, so boxes can be moved, resized,
relabeled, added and deleted. Every change is written back to the same
file; re-render the video with `--load-detections`.

    python annotate_server.py                      # pick the files in the UI
    python annotate_server.py DETECTIONS.json      # video found from videos[0].file_name
    python annotate_server.py VIDEO DETECTIONS.json [--port 8000]

The UI lists the detections files found under --root (default: the
current directory), each matched to the video named in its videos[0].file_name.

Only the frames the detector ran on (the ones in the file) can be
edited. The first change keeps a copy of the file as it was
(<name>.orig.json, never overwritten).

The UI can also fine-tune an RF-DETR on detections files
(train_rfdetr.py) and run a fine-tuned model on a video
(process_video.py --no-render), as background jobs, and propose boxes
for the frame being edited with a cloud, local or trained model
(label_assist.py).

The home page (/) is the workspace (workspace.py): folders and projects,
each project a dataset of uploaded pictures and video frames, labeled
in the same annotator (/annotate?project=ID) and trained on.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import shlex
import shutil
import signal
import sys
import threading
import time
import uuid
import webbrowser
from collections import OrderedDict
from functools import cache
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import train_rfdetr
import training_data
from detections_io import (
    DetectionsFile,
    is_detections_file,
    load_detections,
    write_detections,
)
from file_lookup import VIDEO_EXTENSIONS, best_video, walk_files
from jobs import JobManager
from label_assist import SOURCES, AssistError, LabelAssist
from options import normalize_name
from rfdetr_detector import MODEL_SIZES, WEIGHTS_FILE
from workspace import (
    IMAGE_EXTENSIONS,
    PROJECT_TYPES,
    STATUSES,
    WORKSPACE_DIR,
    Project,
    Workspace,
    decode_image,
    encode_jpeg,
    make_thumbnail,
    media_type,
    now_iso,
    sample_indices,
    sample_video,
    video_info,
)

# ANTHROPIC_API_KEY / LOCATE_ANYTHING_* for Label Assist, as in
# process_video.py.
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv()

UI_DIR = Path(__file__).parent / "annotator_ui"
STUDIO_DIR = Path(__file__).parent / "studio_ui"

JPEG_QUALITY = 90
# Full-size frames kept in memory (encoded JPEG).
FRAME_CACHE_SIZE = 48
# Short forward jumps are read frame by frame instead of seeking, so
# stepping through samples never depends on the codec seeking exactly.
MAX_READ_AHEAD = 120

# Annotator backups, not listed.
BACKUP_SUFFIXES = (".orig.json",)

HERE = Path(__file__).resolve().parent
TRAIN_SCRIPT = HERE / "train_rfdetr.py"
DETECT_SCRIPT = HERE / "process_video.py"
# Where the UI puts new models and uploaded videos (under --root).
MODELS_DIR = "models"
UPLOADS_DIR = "videos/uploads"
# Seconds between checks of whether the server's code changed on disk.
CODE_CHECK_INTERVAL = 5.0
# Annotator sessions kept open at once (one per tab, roughly); the
# oldest is closed when another opens.
MAX_SESSIONS = 6


class CodeWatch:
    """Notices when the Python files this server runs change on disk.

    The UI files are re-read on every page load, but the server keeps
    running the code it started with, so after an update the page can
    call endpoints the running server doesn't have yet. API responses
    then carry `X-Server-Outdated: 1` and the page asks for a restart.
    (process_video.py / train_rfdetr.py run as new processes, so their
    changes apply without a restart.)
    """

    def __init__(self):
        self._files = {
            Path(m.__file__): Path(m.__file__).stat().st_mtime_ns
            for m in list(sys.modules.values())
            if getattr(m, "__file__", None)
            and Path(m.__file__).resolve().parent == HERE
        }
        self._checked = 0.0
        self._outdated = False

    def outdated(self) -> bool:
        now = time.monotonic()
        if not self._outdated and now - self._checked > CODE_CHECK_INTERVAL:
            self._checked = now
            for path, mtime in self._files.items():
                try:
                    changed = path.stat().st_mtime_ns != mtime
                except OSError:
                    changed = True
                if changed:
                    self._outdated = True
                    print(
                        f"note: {path.name} changed since the server started; "
                        "restart it to use the new version"
                    )
                    break
        return self._outdated


class FrameReader:
    """Random access to the video frames, as JPEG.

    Frame indices match `sv.get_video_frames_generator` (decode order
    from 0), which is what the detections file is indexed by.
    """

    def __init__(self, path: Path):
        self.path = path
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise ValueError(f"cannot open video {path}")
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._next = 0  # index of the frame the next read returns
        self._lock = threading.Lock()
        self._frames: OrderedDict[int, bytes] = OrderedDict()
        self._thumbs: dict[int, bytes] = {}
        self._stop = threading.Event()

    def close(self) -> None:
        """Stop the thumbnail pre-pass and release the video."""
        self._stop.set()
        with self._lock:
            self._cap.release()

    def _read(self, index: int) -> np.ndarray:
        if not 0 <= index - self._next <= MAX_READ_AHEAD:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self._next = index
        while self._next < index:
            if not self._cap.grab():
                raise KeyError(index)
            self._next += 1
        ok, frame = self._cap.read()
        if not ok:
            # Leave the position unknown so the next read seeks.
            self._next = -MAX_READ_AHEAD - 1
            raise KeyError(index)
        self._next += 1
        return frame

    def frame(self, index: int) -> bytes:
        with self._lock:
            data = self._frames.get(index)
            if data is not None:
                self._frames.move_to_end(index)
                return data
            frame = self._read(index)
            data = encode_jpeg(frame, JPEG_QUALITY)
            self._frames[index] = data
            while len(self._frames) > FRAME_CACHE_SIZE:
                self._frames.popitem(last=False)
            if index not in self._thumbs:
                self._thumbs[index] = make_thumbnail(frame)
            return data

    def thumbnail(self, index: int) -> bytes:
        with self._lock:
            data = self._thumbs.get(index)
        if data is None:
            self.frame(index)
            data = self._thumbs[index]
        return data

    def warm_thumbnails(self, indices: list[int]) -> None:
        """Decode the video once, in order, and keep the thumbnails of
        `indices` (run in a background thread, on its own capture)."""
        wanted = set(indices)
        if not wanted:
            return
        cap = cv2.VideoCapture(str(self.path))
        try:
            for index in range(max(wanted) + 1):
                if self._stop.is_set():
                    return
                if index not in wanted:
                    if not cap.grab():
                        return
                    continue
                ok, frame = cap.read()
                if not ok:
                    return
                with self._lock:
                    known = index in self._thumbs
                if not known:
                    thumb = make_thumbnail(frame)
                    with self._lock:
                        self._thumbs.setdefault(index, thumb)
        finally:
            cap.release()


class Box(BaseModel):
    label: str = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    box: list[float] = Field(min_length=4, max_length=4)


class FrameUpdate(BaseModel):
    boxes: list[Box]


class ClassesUpdate(BaseModel):
    classes: list[str]


class DetectionsStore:
    """The detections file, held in memory and written on every change.

    Classes are the file's categories, which may include classes no box
    uses yet (added in the UI). Frames changed here are marked edited.
    """

    def __init__(self, path: Path, width: int, height: int):
        self.path = path
        self.data: DetectionsFile = load_detections(path)
        self.frames = self.data.frames
        self.width = width
        self.height = height
        self._lock = threading.Lock()
        # rfdetr files use COCO's category ids, so new COCO classes get
        # theirs.
        self._coco_ids = (
            {normalize_name(n): i for i, n in coco_classes().items()}
            if uses_coco(self.data.info)
            else {}
        )

    @property
    def info(self) -> dict:
        return self.data.info

    @property
    def video(self) -> dict:
        return self.data.video

    @property
    def detector(self) -> str | None:
        return self.data.info.get("detector")

    @property
    def edited(self) -> list[int]:
        return sorted(self.data.edited)

    def classes(self) -> list[str]:
        """Every class name: the categories, then labels used on boxes
        without one (most frequent first). Names that only differ in
        case or `_`/`-` (the same class for process_video.py) appear
        once."""
        counts: dict[str, int] = {}
        for items in self.frames.values():
            for item in items or []:
                counts[item["label"]] = counts.get(item["label"], 0) + 1
        names: dict[str, str] = {}
        for category in self.data.categories:
            names.setdefault(normalize_name(category["name"]), category["name"])
        for name in sorted(counts, key=lambda n: -counts[n]):
            names.setdefault(normalize_name(name), name)
        return list(names.values())

    def _canonical(self, label: str) -> str:
        """`label` as spelled by its category, adding the category if
        it's new (with its COCO id for rfdetr files)."""
        label = label.strip()
        key = normalize_name(label)
        for category in self.data.categories:
            if normalize_name(category["name"]) == key:
                return category["name"]
        taken = {c["id"] for c in self.data.categories}
        new_id = self._coco_ids.get(key)
        if new_id is None or new_id in taken:
            new_id = max(taken | set(self._coco_ids.values()), default=0) + 1
        self.data.categories.append({"id": new_id, "name": label})
        return label

    def _clean(self, box: Box) -> dict:
        x1, y1, x2, y2 = box.box
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        x1, x2 = np.clip([x1, x2], 0, self.width)
        y1, y2 = np.clip([y1, y2], 0, self.height)
        return {
            "label": self._canonical(box.label),
            "confidence": round(box.confidence, 4),
            "box": [round(float(v), 2) for v in (x1, y1, x2, y2)],
        }

    def set_frame(self, index: int, boxes: list[Box]) -> list[dict]:
        with self._lock:
            if index not in self.frames:
                raise KeyError(index)
            cleaned = [self._clean(b) for b in boxes]
            self.frames[index] = cleaned
            self.data.edited.add(index)
            self._write()
            return cleaned

    def set_classes(self, classes: list[str]) -> None:
        """Make the categories `classes`. Categories still used on a
        frame are kept (with their ids)."""
        with self._lock:
            wanted = {normalize_name(c) for c in classes if c.strip()}
            used = {
                normalize_name(item["label"])
                for items in self.frames.values()
                for item in items or []
            }
            self.data.categories = [
                c
                for c in self.data.categories
                if normalize_name(c["name"]) in wanted | used
            ]
            for name in classes:
                if name.strip():
                    self._canonical(name)
            self._write()

    def _write(self) -> None:
        backup = self.path.with_name(f"{self.path.stem}.orig{self.path.suffix}")
        if not backup.exists():
            shutil.copy2(self.path, backup)
        write_detections(self.path, self.data)


def uses_coco(info: dict) -> bool:
    """Files of the stock RF-DETR have COCO classes; a fine-tuned one
    (`weights`) has its own."""
    return info.get("detector") == "rfdetr" and not info.get("weights")


@cache
def coco_classes() -> dict[int, str]:
    """COCO id -> name. Offered as suggestions for rfdetr files (whose
    labels must be COCO classes to load). Empty if rfdetr isn't
    importable."""
    try:
        from options import load_coco_classes

        return dict(sorted(load_coco_classes().items()))
    except Exception:
        return {}


class Session:
    """A video and its detections file, open for editing."""

    def __init__(self, video: Path, detections: Path):
        self.id = uuid.uuid4().hex[:10]
        self.video = Path(os.path.abspath(video))
        self.detections = Path(os.path.abspath(detections))
        self.reader = FrameReader(self.video)
        try:
            self.store = DetectionsStore(
                self.detections, self.reader.width, self.reader.height
            )
        except ValueError:
            self.reader.close()
            raise
        video = self.store.video
        self.suggestions = (
            list(coco_classes().values())
            if uses_coco(self.store.info)
            else []
        )
        self.warnings = [
            f"{key} {video[key]} (video: {actual})"
            for key, actual in (
                ("width", self.reader.width),
                ("height", self.reader.height),
                ("total_frames", self.reader.total_frames),
            )
            if video.get(key) not in (None, actual)
        ]
        threading.Thread(
            target=self.reader.warm_thumbnails,
            args=(sorted(self.store.frames),),
            daemon=True,
        ).start()

    # What the endpoints use; ProjectSession has the same.

    kind = "video"

    @property
    def info(self) -> dict:
        return self.store.info

    def has(self, index: int) -> bool:
        return index in self.store.frames

    def boxes(self, index: int) -> list[dict] | None:
        return self.store.frames[index]

    def save(self, index: int, boxes: list[Box]) -> dict:
        return {"boxes": self.store.set_frame(index, boxes)}

    def image(self, index: int) -> tuple[bytes, str]:
        return self.reader.frame(index), "image/jpeg"

    def thumbnail(self, index: int) -> bytes:
        return self.reader.thumbnail(index)

    def array(self, index: int) -> np.ndarray:
        return decode_image(self.reader.frame(index))

    def classes(self) -> list[str]:
        return self.store.classes()

    def set_classes(self, classes: list[str]) -> None:
        self.store.set_classes(classes)

    def describe(self, rel) -> dict:
        store, reader = self.store, self.reader
        return {
            "id": self.id,
            "mode": "video",
            "video": self.video.name,
            "video_path": rel(self.video),
            "detections": self.detections.name,
            "detections_path": rel(self.detections),
            "warnings": self.warnings,
            "meta": store.info,
            "width": reader.width,
            "height": reader.height,
            "fps": reader.fps,
            "total_frames": reader.total_frames,
            "classes": store.classes(),
            "suggestions": self.suggestions,
            "edited": store.edited,
            # [index, box count (-1 = detection failed)] per frame.
            "frames": [
                [i, -1 if store.frames[i] is None else len(store.frames[i])]
                for i in sorted(store.frames)
            ],
        }

    def close(self) -> None:
        self.reader.close()


class ProjectSession:
    """The images of a project (or of one of its batches), open in the
    annotator. Frame indices are image ids. Saving an image's boxes
    marks it labeled."""

    kind = "project"

    def __init__(self, project: Project, batch_id: str | None = None):
        self.id = uuid.uuid4().hex[:10]
        self.project = project
        self.batch = project.batch(batch_id) if batch_id else None
        self.indices = project.batch_images(batch_id)
        self._open = set(self.indices)
        self.info: dict = {}
        self.warnings: list[str] = []

    def has(self, index: int) -> bool:
        return index in self._open and index in self.project.images

    def boxes(self, index: int) -> list[dict]:
        return self.project.images[index]["boxes"]

    def save(self, index: int, boxes: list[Box]) -> dict:
        cleaned = self.project.set_boxes(index, [b.model_dump() for b in boxes])
        return {"boxes": cleaned, "status": "labeled"}

    def image(self, index: int) -> tuple[bytes, str]:
        path = self.project.image_path(index)
        try:
            return path.read_bytes(), media_type(path)
        except OSError:
            raise KeyError(index)

    def thumbnail(self, index: int) -> bytes:
        return self.project.thumbnail(index)

    def array(self, index: int) -> np.ndarray:
        frame = cv2.imread(str(self.project.image_path(index)))
        if frame is None:
            raise KeyError(index)
        return frame

    def classes(self) -> list[str]:
        return self.project.classes()

    def set_classes(self, classes: list[str]) -> None:
        self.project.set_classes(classes)

    def describe(self, rel) -> dict:
        project = self.project
        with project.lock:
            images = [
                (i, project.images[i]) for i in self.indices if i in project.images
            ]
            first = images[0][1] if images else {"width": 0, "height": 0}
            return {
                "id": self.id,
                "mode": "project",
                "project": {"id": project.id, "name": project.name},
                "batch": (
                    {"id": self.batch["id"], "name": self.batch["name"],
                     "kind": self.batch.get("kind")}
                    if self.batch else None
                ),
                # The key the page remembers its position by.
                "detections_path": f"project:{project.id}:{self.batch['id'] if self.batch else 'all'}",
                "warnings": [],
                "meta": {"detector": None},
                "width": first["width"],
                "height": first["height"],
                "fps": None,
                "total_frames": len(images),
                "classes": project.classes(),
                "suggestions": [],
                "edited": [i for i, r in images if r.get("status") == "labeled"],
                "frames": [[i, len(r["boxes"])] for i, r in images],
                # [id, width, height, file name, status] per image.
                "images": [
                    [i, r["width"], r["height"], Path(r["file_name"]).name,
                     r.get("status", "unlabeled")]
                    for i, r in images
                ],
            }

    def close(self) -> None:
        pass


# -- Finding files --------------------------------------------------


# path -> ((mtime, size), summary): files are only parsed again when
# they change.
_summaries: dict[Path, tuple[tuple[int, int], dict | None]] = {}


def summarize(path: Path) -> dict | None:
    """What the file picker shows about a detections file, or None if
    it isn't one."""
    try:
        stat = path.stat()
    except OSError:
        return None
    key = (stat.st_mtime_ns, stat.st_size)
    cached = _summaries.get(path)
    if cached and cached[0] == key:
        return cached[1]

    summary: dict | None = None
    if is_detections_file(path):
        try:
            data = load_detections(path)
        except ValueError:
            data = None
        if data is not None:
            used: dict[str, int] = {}
            for items in data.frames.values():
                for item in items or []:
                    used[item["label"]] = used.get(item["label"], 0) + 1
            summary = {
                "detector": data.info.get("detector"),
                "weights": data.info.get("weights"),
                "source": data.video.get("file_name"),
                "samples": len(data.frames),
                "edited": len(data.edited),
                "failed": sum(v is None for v in data.frames.values()),
                # {class: box count} of the classes that have boxes.
                "classes": used,
                "scored": any(
                    item.get("confidence", 1.0) < 1.0
                    for items in data.frames.values()
                    for item in items or []
                ),
            }
    _summaries[path] = (key, summary)
    return summary


def video_of(detections: Path, root: Path) -> tuple[str | None, Path | None]:
    """(video name in the file, that video under `root` or None)."""
    summary = summarize(detections) or {}
    source = summary.get("source")
    if not source:
        return None, None
    _, videos = find_files(root)
    return source, best_video(Path(os.path.abspath(detections)), source, videos)


def find_files(
    root: Path, models: list[Path] | None = None, skip: Path | None = None
) -> tuple[list[dict], list[Path]]:
    """Detections files under `root` (newest first) and the videos.
    The model manifests found are added to `models`. Nothing under
    `skip` (the workspace, which has its own lists) is looked at."""
    videos: list[Path] = []
    jsons: list[Path] = []
    for path in walk_files(root):
        if skip is not None and path.is_relative_to(skip):
            continue
        suffix = path.suffix.lower()
        if suffix in VIDEO_EXTENSIONS:
            videos.append(path)
        elif path.name == train_rfdetr.MANIFEST:
            if models is not None:
                models.append(path)
        elif suffix == ".json" and not path.name.endswith(BACKUP_SUFFIXES):
            jsons.append(path)

    found = []
    for path in jsons:
        summary = summarize(path)
        if summary is None:
            continue
        source = summary.get("source")
        found.append({
            "path": path,
            "detector": None,
            "source": None,
            "samples": None,
            "edited": 0,
            **summary,
            "video": best_video(path, source, videos) if source else None,
            "modified": path.stat().st_mtime,
        })
    found.sort(key=lambda f: -f["modified"])
    return found, sorted(videos)


# -- Models and jobs ------------------------------------------------


def read_model(manifest: Path) -> dict | None:
    """What the UI shows about a model trained by train_rfdetr.py, or
    None if `manifest` isn't one (or its weights are gone)."""
    try:
        data = json.loads(manifest.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("format") != train_rfdetr.MANIFEST_FORMAT:
        return None
    folder = manifest.parent
    if not (folder / str(data.get("weights", ""))).is_file():
        return None
    dataset = data.get("dataset") or {}
    return {
        "path": folder,
        # The name given in the UI (kept in model.json), else the folder's.
        "name": str(data.get("name") or folder.name),
        "folder": folder.name,
        "classes": data.get("classes", []),
        "threshold": data.get("threshold"),
        "base_model": data.get("base_model"),
        "created": data.get("created"),
        "best": data.get("best"),
        "history": data.get("history", []),
        "stopped_early": data.get("stopped_early", False),
        "epochs_run": (data.get("training") or {}).get("epochs_run"),
        "images": dataset.get("images"),
        "boxes_per_class": dataset.get("boxes_per_class"),
        "sources": [
            s.get("detections") or s.get("name") for s in dataset.get("sources", [])
        ],
        "project": dataset.get("project"),
        "options": dataset.get("options"),
        "minutes": (data.get("training") or {}).get("minutes"),
        "calibration": data.get("calibration"),
        "modified": manifest.stat().st_mtime,
    }


# What train_rfdetr.py writes in a model folder (besides dataset/).
MODEL_FILES = (
    train_rfdetr.MANIFEST,
    WEIGHTS_FILE,
    *train_rfdetr.BEST_CHECKPOINTS,
    "metrics.csv",
    "training_config.json",
)


def delete_model(folder: Path) -> list[str]:
    """Delete what train_rfdetr.py wrote in `folder`, and the folder if
    nothing else is left in it. Returns the names left behind."""
    # The weights and manifest go first: without them the folder is no
    # longer listed as a model, even if the rest fails.
    for path in [folder / n for n in MODEL_FILES] + train_rfdetr.extra_checkpoints(folder):
        path.unlink(missing_ok=True)
    for name in ("dataset", "dataset_grids"):
        sub = folder / name
        if name == "dataset" and not (sub / training_data.MARKER).exists():
            continue
        shutil.rmtree(sub, ignore_errors=True)
    left = sorted(p.name for p in folder.iterdir())
    if not left:
        folder.rmdir()
    return left


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-") or "model"


def unique_path(path: Path) -> Path:
    """`path`, or `path` with -2, -3... added if it exists."""
    if not path.exists():
        return path
    for n in range(2, 1000):
        candidate = path.with_name(f"{path.stem}-{n}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise HTTPException(400, f"{path} exists")


def train_defaults() -> dict:
    """The training options' defaults, for the UI (from the CLI)."""
    t = train_rfdetr
    return {
        "models": MODEL_SIZES,
        "model": t.DEFAULT_TRAIN_MODEL,
        "epochs": t.DEFAULT_EPOCHS,
        "batch_size": t.DEFAULT_BATCH_SIZE,
        "min_score": t.DEFAULT_MIN_SCORE,
        "val_fraction": t.DEFAULT_VAL_FRACTION,
        "image_size": t.DEFAULT_IMAGE_SIZE,
    }


class TrainRequest(BaseModel):
    detections: list[str] = []
    # A workspace project id, instead of detections files.
    project: str | None = None
    classes: list[str] | None = None
    model: str = "small"
    epochs: int = Field(20, ge=1, le=2000)
    batch_size: int = Field(4, ge=1, le=128)
    every: int = Field(1, ge=1)
    only_edited: bool = False
    min_score: float = Field(0.5, ge=0, le=1)
    val_fraction: float = Field(0.2, gt=0, le=0.5)
    image_size: int = Field(720, ge=0)
    patience: int = Field(0, ge=0)
    output: str | None = None


class ModelUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class DetectRequest(BaseModel):
    model: str = Field(min_length=1)
    video: str = Field(min_length=1)
    output: str | None = None
    # None: the one chosen for the model when it was trained.
    threshold: float | None = Field(None, ge=0, le=1)
    sample_rate: float | None = Field(None, gt=0)
    # The workspace project the job is run from (shown on its page).
    project: str | None = None


# -- App ------------------------------------------------------------


class AssistClass(BaseModel):
    name: str = Field(min_length=1)
    description: str = ""


class AssistRequest(BaseModel):
    source: str
    # Cloud model id, RF-DETR size, or trained model folder.
    model: str | None = None
    classes: list[AssistClass] = Field(min_length=1)


class AutoLabelRequest(BaseModel):
    source: str
    # Cloud model id, RF-DETR size, or trained model folder.
    model: str | None = None
    classes: list[AssistClass] = Field(min_length=1)
    # "unlabeled": only images without labels; "todo": also the
    # pre-labeled ones (their boxes are replaced). Labeled images are
    # never touched.
    images: str = Field("unlabeled", pattern="^(unlabeled|todo)$")
    # Boxes below it are left out; None: the detector's own threshold.
    min_confidence: float | None = Field(None, ge=0, le=1)


class OpenRequest(BaseModel):
    detections: str = Field(min_length=1)
    video: str | None = None


class NodeCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    # Folder id; None / "" for the top level.
    parent: str | None = None
    type: str = "object-detection"


class NodeUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=200)
    # Move to this folder id ("" = the top level); None leaves it.
    parent: str | None = None


class BatchCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    kind: str = Field(pattern="^(images|video)$")


class BatchUpdate(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class ImagesStatus(BaseModel):
    # The images: these ids, or those of a batch (with `from_status`).
    ids: list[int] = []
    batch: str | None = None
    from_status: str | None = None
    status: str


class ImageIds(BaseModel):
    ids: list[int] = Field(min_length=1)


class ClassRename(BaseModel):
    old: str = Field(min_length=1)
    new: str = Field(min_length=1)


class AnnotateRequest(BaseModel):
    batch: str | None = None


def create_app(
    root: Path,
    initial: tuple[Path, Path] | None = None,
    workspace: Workspace | None = None,
) -> FastAPI:
    root = root.resolve()
    ws = workspace or Workspace(root / WORKSPACE_DIR)
    # Open annotator sessions by id, oldest first.
    sessions: OrderedDict[str, Session | ProjectSession] = OrderedDict()
    lock = threading.Lock()
    jobs = JobManager()
    # A training run outliving the server would hold the GPU unseen.
    atexit.register(jobs.shutdown)
    assist = LabelAssist()
    atexit.register(assist.close)
    # The COCO names import rfdetr (slow); load them before Label Assist
    # asks.
    threading.Thread(target=coco_classes, daemon=True).start()

    def rel(path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            return str(path.relative_to(root))
        except ValueError:
            return str(path)

    def resolve(path: str) -> Path:
        p = Path(path).expanduser()
        return Path(os.path.abspath(p if p.is_absolute() else root / p))

    def add_session(session):
        with lock:
            sessions[session.id] = session
            closed = []
            while len(sessions) > MAX_SESSIONS:
                closed.append(sessions.popitem(last=False)[1])
        for old in closed:
            old.close()
        return session

    def open_session(video: Path, detections: Path) -> Session:
        session = add_session(Session(video, detections))
        if session.warnings:
            print(
                "warning: detections file doesn't match this video: "
                + ", ".join(session.warnings)
            )
        print(f"Opened {rel(session.detections)} on {rel(session.video)}")
        return session

    if initial is not None:
        open_session(*initial)

    def current(s: str | None = None) -> Session | ProjectSession:
        """The session `s` (the id the page was loaded with), or the
        newest one. A page never edits files after its session was
        closed (too many others were opened since)."""
        with lock:
            if s:
                session = sessions.get(s)
                if session is None:
                    raise HTTPException(
                        409, "other files were opened since this page was loaded"
                    )
                return session
            if not sessions:
                raise HTTPException(409, "no files are open")
            return next(reversed(sessions.values()))

    def frame_or_404(session, index: int) -> None:
        if not session.has(index):
            raise HTTPException(404, f"frame {index} is not open")

    app = FastAPI(title="Detections annotator")
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")
    app.mount("/studio", StaticFiles(directory=STUDIO_DIR), name="studio")

    code = CodeWatch()

    @app.middleware("http")
    async def revalidate_ui(request, call_next):
        # Always revalidate the UI files, so an updated UI is picked up.
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache"
        elif code.outdated():
            response.headers["X-Server-Outdated"] = "1"
        return response

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STUDIO_DIR / "index.html")

    @app.get("/annotate")
    def annotate_page() -> FileResponse:
        return FileResponse(UI_DIR / "index.html")

    @app.get("/api/files")
    def files() -> dict:
        found, videos = find_files(root, skip=ws.path)
        return {
            "root": str(root),
            "detections": [
                {**f, "path": rel(f["path"]), "video": rel(f["video"])}
                for f in found
            ],
            "videos": [rel(v) for v in videos],
        }

    @app.post("/api/open")
    def open_files(req: OpenRequest) -> dict:
        detections = resolve(req.detections)
        if not detections.is_file():
            raise HTTPException(400, f"no such file: {req.detections}")
        if req.video:
            video = resolve(req.video)
            if not video.is_file():
                raise HTTPException(400, f"no such file: {req.video}")
        else:
            try:
                source, video = video_of(detections, root)
            except ValueError as exc:
                raise HTTPException(400, str(exc))
            if video is None:
                raise HTTPException(
                    400,
                    f"can't find the video {source or '(not recorded)'} "
                    f"under {root}, choose it",
                )
        try:
            session = open_session(video, detections)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return {"id": session.id}

    @app.get("/api/project")
    def project(s: str | None = Query(None)) -> dict:
        session = current(s)
        return {"root": str(root), **session.describe(rel)}

    @app.get("/api/frames/{index}")
    def get_frame(index: int, s: str | None = Query(None)) -> dict:
        session = current(s)
        frame_or_404(session, index)
        return {"index": index, "boxes": session.boxes(index)}

    @app.put("/api/frames/{index}")
    def put_frame(
        index: int, update: FrameUpdate, s: str | None = Query(None)
    ) -> dict:
        session = current(s)
        frame_or_404(session, index)
        return {"index": index, **session.save(index, update.boxes)}

    @app.put("/api/classes")
    def put_classes(update: ClassesUpdate, s: str | None = Query(None)) -> dict:
        session = current(s)
        session.set_classes(update.classes)
        return {"classes": session.classes()}

    def image_response(data: bytes, kind: str = "image/jpeg") -> Response:
        # Image URLs carry the session (or project) id, so caching can't
        # mix videos.
        return Response(
            data,
            media_type=kind,
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @app.get("/api/frames/{index}/image")
    def frame_image(index: int, s: str | None = Query(None)) -> Response:
        session = current(s)
        frame_or_404(session, index)
        try:
            return image_response(*session.image(index))
        except KeyError:
            raise HTTPException(404, f"cannot read frame {index}")

    @app.get("/api/frames/{index}/thumbnail")
    def frame_thumbnail(index: int, s: str | None = Query(None)) -> Response:
        session = current(s)
        frame_or_404(session, index)
        try:
            return image_response(session.thumbnail(index))
        except KeyError:
            raise HTTPException(404, f"cannot read frame {index}")

    # -- Training / detection jobs ----------------------------------

    def cli(command: list[str]) -> str:
        """`command` as the user would type it (relative paths, no UI
        flags)."""
        shown = []
        for i, arg in enumerate(command):
            if arg == "--progress-json":
                continue
            if i == 0:
                arg = "python"
            elif os.path.isabs(arg):
                arg = rel(Path(arg)) or arg
            shown.append(arg)
        return shlex.join(shown)

    def video_for(detections: Path) -> Path:
        with lock:
            for session in sessions.values():
                if getattr(session, "detections", None) == detections:
                    return session.video
        source, video = video_of(detections, root)
        if video is None:
            raise HTTPException(
                400,
                f"can't find the video {source or '(not recorded)'} of "
                f"{rel(detections)} under {root}",
            )
        return video

    def job_response(job, command: list[str], dry: bool, result: dict) -> dict:
        if dry:
            return {"cli": cli(command), "result": result}
        return {**job.to_dict(), "cli": cli(command)}

    @app.post("/api/videos")
    async def upload_video(request: Request, name: str = Query(min_length=1)) -> dict:
        """Save a video sent as the request body (the page's Upload…) in
        <root>/videos/uploads/."""
        original = Path(name)
        suffix = original.suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            raise HTTPException(
                400,
                f"{original.name} is not a video "
                f"({', '.join(sorted(VIDEO_EXTENSIONS))})",
            )
        folder = root / UPLOADS_DIR
        folder.mkdir(parents=True, exist_ok=True)
        target = unique_path(folder / f"{slug(original.stem)}{suffix}")
        # Hidden while it uploads (not listed), with the real extension so
        # OpenCV can check it.
        partial = target.with_name(f".{target.stem}.uploading{suffix}")
        try:
            with partial.open("wb") as f:
                async for chunk in request.stream():
                    f.write(chunk)
            cap = cv2.VideoCapture(str(partial))
            readable = cap.isOpened() and cap.read()[0]
            cap.release()
            if not readable:
                raise HTTPException(400, f"can't read {original.name} as a video")
            os.replace(partial, target)
        finally:
            partial.unlink(missing_ok=True)
        print(f"Uploaded {rel(target)}")
        return {"video": rel(target)}

    def project_models(project: Project) -> list[dict]:
        """A project's models, newest first."""
        folder = project.models_dir
        found = []
        if folder.is_dir():
            for child in folder.iterdir():
                model = read_model(child / train_rfdetr.MANIFEST)
                if model is not None:
                    found.append(model)
        found.sort(key=lambda m: -m["modified"])
        return [{**m, "path": rel(m["path"])} for m in found]

    def trained_models() -> list[dict]:
        """Models under the root and in the workspace's projects, newest
        first, with relative paths."""
        manifests: list[Path] = []
        find_files(root, manifests, skip=ws.path)
        for node in ws.nodes().values():
            if node.kind == "project" and (node.path / "models").is_dir():
                manifests += [
                    child / train_rfdetr.MANIFEST
                    for child in (node.path / "models").iterdir()
                ]
        found = [m for m in map(read_model, manifests) if m is not None]
        found.sort(key=lambda m: -m["modified"])
        return [{**m, "path": rel(m["path"])} for m in found]

    @app.get("/api/models")
    def models() -> dict:
        return {"models": trained_models(), "defaults": train_defaults()}

    # -- Label Assist -----------------------------------------------

    def assist_choices(info: dict, project: Project | None) -> dict:
        """What Label Assist can run (with a project: its own models
        first)."""
        trained = [
            {k: m[k] for k in ("path", "name", "classes", "threshold")}
            for m in trained_models()
        ]
        if project is not None:
            mine = str(project.models_dir)
            trained.sort(key=lambda m: not str(resolve(m["path"])).startswith(mine))
        return {
            "sources": assist.sources(info, trained),
            "coco": list(coco_classes().values()),
        }

    @app.get("/api/assist")
    def assist_sources(s: str | None = Query(None)) -> dict:
        session = current(s)
        return assist_choices(
            session.info, session.project if session.kind == "project" else None
        )

    @app.post("/api/assist/{index}")
    def assist_frame(
        index: int, req: AssistRequest, s: str | None = Query(None)
    ) -> dict:
        """Proposed boxes for a frame; nothing is saved."""
        session = current(s)
        frame_or_404(session, index)
        if req.source not in SOURCES:
            raise HTTPException(400, f"unknown source {req.source}")
        model = req.model
        if req.source == "trained":
            folder = resolve(model or "")
            if read_model(folder / train_rfdetr.MANIFEST) is None:
                raise HTTPException(400, f"{model} is not a trained model")
            model = str(folder)
        try:
            frame = session.array(index)
        except KeyError:
            raise HTTPException(404, f"cannot read frame {index}")
        try:
            result = assist.detect(
                frame,
                req.source,
                model,
                [(c.name, c.description) for c in req.classes],
                session.info,
                coco_classes(),
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        except AssistError as exc:
            raise HTTPException(502, str(exc))
        print(
            f"Label Assist on frame {index}: {len(result['boxes'])} boxes "
            f"from {result['model']} in {result['seconds']:g} s"
        )
        return {"index": index, **result}

    @app.get("/api/jobs")
    def list_jobs() -> dict:
        return {"jobs": jobs.list()}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        return job.to_dict(log=True)

    @app.delete("/api/jobs/{job_id}")
    def remove_job(job_id: str) -> dict:
        try:
            jobs.remove(job_id)
        except KeyError:
            raise HTTPException(404, "no such job")
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return {"removed": job_id}

    def model_folder(path: str) -> Path:
        """The folder of a trained model under the root or the workspace."""
        folder = resolve(path)
        if folder in (root, ws.path) or not (
            folder.is_relative_to(root) or folder.is_relative_to(ws.path)
        ):
            raise HTTPException(400, f"{path} is not a model under {root}")
        if read_model(folder / train_rfdetr.MANIFEST) is None:
            raise HTTPException(400, f"{path} is not a trained model")
        return folder

    @app.patch("/api/models")
    def rename_model(update: ModelUpdate, path: str = Query(min_length=1)) -> dict:
        """Give a model another name. Only model.json changes: the folder
        keeps its name, so the paths that point at it (detections files,
        --weights commands) still work."""
        folder = model_folder(path)
        busy = jobs.using(str(folder))
        if busy is not None:
            raise HTTPException(409, f"'{busy.title}' is using it; stop it first")
        manifest = folder / train_rfdetr.MANIFEST
        data = json.loads(manifest.read_text())
        data["name"] = update.name.strip()
        tmp = manifest.with_name(f".{manifest.name}.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, manifest)
        print(f"Renamed the model {rel(folder)} to {data['name']}")
        return {**read_model(manifest), "path": rel(folder)}

    @app.delete("/api/models")
    def remove_model(path: str = Query(min_length=1)) -> dict:
        folder = model_folder(path)
        busy = jobs.using(str(folder))
        if busy is not None:
            raise HTTPException(409, f"'{busy.title}' is using it; stop it first")
        left = delete_model(folder)
        if left:
            print(f"Deleted the model in {rel(folder)}; kept {', '.join(left)}")
        else:
            print(f"Deleted the model {rel(folder)}")
        return {"deleted": rel(folder), "kept": left}

    @app.post("/api/jobs/{job_id}/stop")
    def stop_job(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        job.stop()
        return job.to_dict()

    def start(kind: str, title: str, command: list[str], result: dict):
        # train_rfdetr.py stops cleanly (keeping the best weights so far)
        # on SIGUSR1; Ctrl-C would skip that.
        stop_signal = (
            getattr(signal, "SIGUSR1", signal.SIGINT)
            if kind == "train"
            else signal.SIGINT
        )
        try:
            return jobs.start(
                kind, title, command, str(root), result, stop_signal
            )
        except RuntimeError as exc:
            raise HTTPException(409, f"{exc}; wait for it or stop it first")

    @app.post("/api/jobs/train")
    def train(req: TrainRequest, dry: bool = Query(False)) -> dict:
        t = train_rfdetr

        if req.model not in MODEL_SIZES:
            raise HTTPException(400, f"unknown model size {req.model}")
        project = None
        if req.project:
            project = project_or_404(req.project)
            # train_rfdetr.py reads the dataset from disk.
            project.flush()
            inputs = ["--project", str(project.path)]
        elif req.detections:
            detections = []
            for name in req.detections:
                path = resolve(name)
                if not path.is_file():
                    raise HTTPException(400, f"no such file: {name}")
                detections.append(path)
            videos = [video_for(d) for d in detections]
            inputs = [*map(str, detections), "--videos", *map(str, videos)]
        else:
            raise HTTPException(400, "choose detections files or a project")
        if req.output and req.output.strip():
            out = resolve(req.output.strip())
            if out.exists() and any(out.iterdir()):
                raise HTTPException(400, f"{req.output} already exists")
        elif project is not None:
            out = unique_path(project.models_dir / f"{slug(project.path.name)}-{req.model}")
        else:
            name = t.default_output(str(detections[0]), req.model).name
            out = unique_path(root / MODELS_DIR / slug(name))

        command = [
            sys.executable, str(TRAIN_SCRIPT),
            *inputs,
            "-o", str(out),
            "--progress-json",
        ]
        # Only what differs from the CLI's defaults, so the command shown
        # to the user stays short.
        for flag, value, default in (
            ("--model", req.model, t.DEFAULT_TRAIN_MODEL),
            ("--epochs", req.epochs, t.DEFAULT_EPOCHS),
            ("--batch-size", req.batch_size, t.DEFAULT_BATCH_SIZE),
            ("--every", req.every, 1),
            ("--min-score", req.min_score, t.DEFAULT_MIN_SCORE),
            ("--val-fraction", req.val_fraction, t.DEFAULT_VAL_FRACTION),
            ("--image-size", req.image_size, t.DEFAULT_IMAGE_SIZE),
        ):
            if value != default:
                command += [flag, f"{value:g}" if isinstance(value, float) else str(value)]
        if req.only_edited:
            command.append("--only-edited")
        if req.patience:
            command += ["--patience", str(req.patience)]
        classes = [c.strip() for c in req.classes or [] if c.strip()]
        if classes:
            command += ["--classes", *classes]
        result = {"model": rel(out)}
        if project is not None:
            result["project"] = project.id
        if dry:
            return job_response(None, command, True, result)
        job = start("train", f"Train {out.name}", command, result)
        return job_response(job, command, False, result)

    @app.post("/api/jobs/detect")
    def detect(req: DetectRequest, dry: bool = Query(False)) -> dict:
        model = resolve(req.model)
        info = read_model(model / train_rfdetr.MANIFEST)
        if info is None:
            raise HTTPException(400, f"{req.model} is not a trained model")
        video = resolve(req.video)
        if not video.is_file():
            raise HTTPException(400, f"no such video: {req.video}")
        if req.output and req.output.strip():
            out = resolve(req.output.strip())
            if out.exists():
                raise HTTPException(400, f"{req.output} already exists")
        else:
            out = unique_path(
                video.parent / f"{video.stem}_{slug(model.name)}_dets.json"
            )
        command = [
            sys.executable, str(DETECT_SCRIPT), str(video),
            "--weights", str(model),
            "--save-detections", str(out),
            "--no-render",
        ]
        if req.threshold is not None:
            command += ["--threshold", f"{req.threshold:g}"]
        if req.sample_rate:
            command += ["--sample-rate", f"{req.sample_rate:g}"]
        result = {"detections": rel(out), "video": rel(video)}
        if req.project:
            result["project"] = project_or_404(req.project).id
        if dry:
            return job_response(None, command, True, result)
        job = start(
            "detect", f"Detect {video.name} with {info['name']}", command, result
        )
        return job_response(job, command, False, result)

    # -- Workspace: folders and projects ----------------------------

    # Videos being sampled into a batch: batch id -> progress.
    ingest: dict[str, dict] = {}

    # Auto-label runs (Label Assist on every image of a batch): batch
    # id -> progress.
    autolabel: dict[str, dict] = {}

    def progress_of(batch_id: str) -> dict | None:
        progress = ingest.get(batch_id)
        if progress is None:
            return None
        return {"done": progress["done"], "total": progress["total"]}

    def autolabel_of(batch_id: str) -> dict | None:
        run = autolabel.get(batch_id)
        if run is None:
            return None
        return {k: v for k, v in run.items() if k not in ("stop", "thread", "project")}

    def project_or_404(project_id: str) -> Project:
        try:
            return ws.project(project_id)
        except KeyError:
            raise HTTPException(404, "no such project")
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    def model_count(path: Path) -> int:
        folder = path / "models"
        if not folder.is_dir():
            return 0
        return sum((c / train_rfdetr.MANIFEST).is_file() for c in folder.iterdir())

    def node_summary(node, nodes: dict) -> dict:
        base = {
            "kind": node.kind,
            "id": node.id,
            "name": node.meta.get("name", node.path.name),
            "parent": node.parent,
            "created": node.meta.get("created"),
            "updated": node.meta.get("updated"),
        }
        if node.kind == "folder":
            children = [n for n in nodes.values() if n.parent == node.id]
            return {
                **base,
                "folders": sum(n.kind == "folder" for n in children),
                "projects": sum(n.kind == "project" for n in children),
            }
        meta = node.meta
        # A loaded project's stats are newer than its project.json.
        stats = meta.get("stats") or {}
        for project in ws.loaded():
            if project.id == node.id:
                stats = project.stats()
                base["updated"] = project.meta.get("updated", base["updated"])
        project_type = meta.get("type", "object-detection")
        return {
            **base,
            "type": project_type,
            "type_name": PROJECT_TYPES.get(project_type, project_type),
            "stats": stats,
            "models": model_count(node.path),
        }

    def folder_line(nodes: dict, folder_id: str | None) -> str:
        names = []
        while folder_id and folder_id in nodes:
            names.append(nodes[folder_id].meta.get("name", ""))
            folder_id = nodes[folder_id].parent
        return " / ".join(reversed(names))

    @app.get("/api/ws")
    def ws_list(folder: str | None = Query(None), q: str = Query("")) -> dict:
        nodes = ws.nodes()
        if folder and (folder not in nodes or nodes[folder].kind != "folder"):
            raise HTTPException(404, "no such folder")
        query = q.strip().lower()
        if query:
            # Search the whole workspace.
            items = [n for n in nodes.values() if query in n.meta.get("name", "").lower()]
        else:
            items = [n for n in nodes.values() if n.parent == (folder or None)]
        summaries = []
        for node in items:
            summary = node_summary(node, nodes)
            if query:
                summary["where"] = folder_line(nodes, node.parent)
            summaries.append(summary)
        return {
            "root": str(ws.path),
            "folder": (
                {"id": folder, "name": nodes[folder].meta.get("name")} if folder else None
            ),
            "trail": ws.breadcrumbs(folder),
            "folders": [x for x in summaries if x["kind"] == "folder"],
            "projects": [x for x in summaries if x["kind"] == "project"],
            "types": PROJECT_TYPES,
        }

    @app.get("/api/ws/folders")
    def ws_folders() -> dict:
        nodes = ws.nodes()
        folders = [
            {"id": n.id, "name": n.meta.get("name"), "parent": n.parent,
             "path": folder_line(nodes, n.id)}
            for n in nodes.values() if n.kind == "folder"
        ]
        folders.sort(key=lambda f: f["path"].lower())
        return {"folders": folders}

    def ws_error(exc: Exception):
        if isinstance(exc, KeyError):
            return HTTPException(404, "not found")
        return HTTPException(400, str(exc))

    @app.post("/api/ws/folders")
    def ws_create_folder(req: NodeCreate) -> dict:
        try:
            node = ws.create_folder(req.name, req.parent or None)
        except (KeyError, ValueError, OSError) as exc:
            raise ws_error(exc)
        return node_summary(node, ws.nodes())

    @app.post("/api/ws/projects")
    def ws_create_project(req: NodeCreate) -> dict:
        try:
            node = ws.create_project(req.name, req.parent or None, req.type)
        except (KeyError, ValueError, OSError) as exc:
            raise ws_error(exc)
        print(f"Created the project {req.name}")
        return node_summary(ws.nodes()[node.id], ws.nodes())

    def busy_projects(node_id: str) -> None:
        """409 if a job (training) is using a project under the node."""
        nodes = ws.nodes()
        ids = [node_id] if nodes[node_id].kind == "project" else ws.project_ids_under(node_id)
        for project_id in ids:
            job = jobs.using(str(nodes[project_id].path))
            if job is not None:
                raise HTTPException(409, f"'{job.title}' is using {nodes[project_id].meta.get('name')}; stop it first")
            if any(r["project"] == project_id for r in autolabel.values()):
                raise HTTPException(409, "it is being auto-labeled; stop that first")
            if any(i.get("project") == project_id for i in ingest.values()):
                raise HTTPException(409, "a video is still being added to it; wait for it")

    @app.patch("/api/ws/nodes/{node_id}")
    def ws_update(node_id: str, req: NodeUpdate) -> dict:
        try:
            if req.parent is not None:
                busy_projects(node_id)
                ws.move(node_id, req.parent or None)
            if req.name is not None:
                ws.rename(node_id, req.name)
        except (KeyError, ValueError, OSError) as exc:
            raise ws_error(exc)
        nodes = ws.nodes()
        return node_summary(nodes[node_id], nodes)

    @app.delete("/api/ws/nodes/{node_id}")
    def ws_delete(node_id: str) -> dict:
        nodes = ws.nodes()
        if node_id not in nodes:
            raise HTTPException(404, "not found")
        busy_projects(node_id)
        gone = set(
            [node_id] if nodes[node_id].kind == "project" else ws.project_ids_under(node_id)
        )
        with lock:
            for sid in [sid for sid, x in sessions.items()
                        if x.kind == "project" and x.project.id in gone]:
                sessions.pop(sid).close()
        ws.delete(node_id)
        print(f"Deleted {nodes[node_id].meta.get('name')} ({rel(nodes[node_id].path)})")
        return {"deleted": node_id}

    # -- A project's dataset

    @app.get("/api/ws/projects/{project_id}")
    def ws_project(project_id: str) -> dict:
        project = project_or_404(project_id)
        nodes = ws.nodes()
        node = nodes[project_id]
        counts = project.class_counts()
        return {
            **node_summary(node, nodes),
            "path": rel(project.path),
            "trail": ws.breadcrumbs(node.parent),
            "batches": [
                {
                    **project.batch_summary(b),
                    "progress": progress_of(b["id"]),
                    "autolabeling": autolabel_of(b["id"]),
                }
                for b in reversed(project.batches)
            ],
            "classes": [{"name": n, "boxes": c} for n, c in counts.items()],
            "models": project_models(project),
        }

    @app.get("/api/ws/projects/{project_id}/detections")
    def ws_detections(project_id: str) -> dict:
        """The detections files (--save-detections) that belong to a
        project: made with one of its models, imported into one of its
        batches, or by a job run from its page. Also the videos under
        the root, to run a model on."""
        project = project_or_404(project_id)
        models_dir = project.models_dir.resolve()
        imported = {
            resolve(b["source"]["detections"]): b["name"]
            for b in project.batches
            if (b.get("source") or {}).get("detections")
        }
        from_jobs = {
            resolve(j["result"]["detections"])
            for j in jobs.list()
            if j["result"].get("project") == project.id and j["result"].get("detections")
        }
        found, videos = find_files(root, skip=ws.path)
        files = []
        for f in found:
            path = Path(os.path.abspath(f["path"]))
            model = None
            if f.get("weights"):
                weights = Path(f["weights"]).expanduser().resolve()
                folder = weights if weights.is_dir() else weights.parent
                if folder.is_relative_to(models_dir):
                    info = read_model(folder / train_rfdetr.MANIFEST)
                    model = info["name"] if info else folder.name
            batch = imported.get(path)
            if model is None and batch is None and path not in from_jobs:
                continue
            files.append({
                **{k: f.get(k) for k in ("detector", "source", "samples", "edited", "failed", "classes", "modified")},
                "path": rel(path),
                "name": path.name,
                "video": rel(f["video"]),
                "model": model,
                "batch": batch,
            })
        return {"files": files, "videos": [rel(v) for v in videos]}

    @app.get("/api/ws/projects/{project_id}/images")
    def ws_images(
        project_id: str,
        batch: str | None = Query(None),
        status: str | None = Query(None),
        cls: str | None = Query(None),
        q: str = Query(""),
        sort: str = Query("newest"),
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1, le=500),
    ) -> dict:
        project = project_or_404(project_id)
        with project.lock:
            ids = project.batch_images(batch or None)
            if sort == "newest":
                ids = ids[::-1]
            elif sort == "name":
                ids.sort(key=lambda i: project.images[i]["file_name"].lower())
            records = [(i, project.images[i]) for i in ids]
            if status == "empty":
                records = [(i, r) for i, r in records
                           if r.get("status") == "labeled" and not r["boxes"]]
            elif status:
                records = [(i, r) for i, r in records if r.get("status") == status]
            if cls:
                key = normalize_name(cls)
                records = [(i, r) for i, r in records
                           if any(normalize_name(b["label"]) == key for b in r["boxes"])]
            if q.strip():
                needle = q.strip().lower()
                records = [(i, r) for i, r in records
                           if needle in Path(r["file_name"]).name.lower()]
            page = records[offset:offset + limit]
            batch_names = {b["id"]: b["name"] for b in project.batches}
            return {
                "total": len(records),
                "offset": offset,
                "images": [
                    {
                        "id": i,
                        "name": Path(r["file_name"]).name,
                        "width": r["width"],
                        "height": r["height"],
                        "status": r.get("status", "unlabeled"),
                        "batch": r.get("batch_id"),
                        "batch_name": batch_names.get(r.get("batch_id")),
                        "boxes": [
                            [b["label"], *b["box"], b.get("confidence", 1.0)]
                            for b in r["boxes"]
                        ],
                    }
                    for i, r in page
                ],
            }

    def image_or_404(project: Project, image_id: int) -> None:
        if image_id not in project.images:
            raise HTTPException(404, "no such image")

    @app.get("/api/ws/projects/{project_id}/images/{image_id}/thumbnail")
    def ws_thumbnail(project_id: str, image_id: int) -> Response:
        project = project_or_404(project_id)
        image_or_404(project, image_id)
        try:
            return image_response(project.thumbnail(image_id))
        except KeyError:
            raise HTTPException(404, "cannot read the image")

    @app.get("/api/ws/projects/{project_id}/images/{image_id}/image")
    def ws_image(project_id: str, image_id: int) -> Response:
        project = project_or_404(project_id)
        image_or_404(project, image_id)
        path = project.image_path(image_id)
        if not path.is_file():
            raise HTTPException(404, "the image file is gone")
        return FileResponse(path, media_type=media_type(path))

    @app.post("/api/ws/projects/{project_id}/images/status")
    def ws_images_status(project_id: str, req: ImagesStatus) -> dict:
        project = project_or_404(project_id)
        if req.status not in STATUSES:
            raise HTTPException(400, f"unknown status {req.status}")
        ids = list(req.ids)
        if req.batch:
            with project.lock:
                ids += [
                    i for i in project.batch_images(req.batch)
                    if req.from_status in (None, project.images[i].get("status"))
                ]
        if not ids:
            raise HTTPException(400, "no images given")
        return {"changed": project.set_status(ids, req.status)}

    @app.post("/api/ws/projects/{project_id}/images/delete")
    def ws_images_delete(project_id: str, req: ImageIds) -> dict:
        project = project_or_404(project_id)
        return {"deleted": project.delete_images(req.ids)}

    @app.put("/api/ws/projects/{project_id}/classes")
    def ws_set_classes(project_id: str, update: ClassesUpdate) -> dict:
        project = project_or_404(project_id)
        project.set_classes(update.classes)
        return {"classes": project.classes()}

    @app.post("/api/ws/projects/{project_id}/classes/rename")
    def ws_class_rename(project_id: str, req: ClassRename) -> dict:
        project = project_or_404(project_id)
        try:
            project.rename_class(req.old, req.new)
        except KeyError:
            raise HTTPException(404, f"no class {req.old}")
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return {"classes": project.classes()}

    @app.delete("/api/ws/projects/{project_id}/classes")
    def ws_class_delete(project_id: str, name: str = Query(min_length=1)) -> dict:
        project = project_or_404(project_id)
        return {"boxes": project.delete_class(name), "classes": project.classes()}

    @app.post("/api/ws/projects/{project_id}/annotate")
    def ws_annotate(project_id: str, req: AnnotateRequest) -> dict:
        """Open the project (or one batch) in the annotator."""
        project = project_or_404(project_id)
        try:
            session = add_session(ProjectSession(project, req.batch or None))
        except KeyError:
            raise HTTPException(404, "no such batch")
        return {"id": session.id}

    # -- Uploads

    @app.post("/api/ws/projects/{project_id}/batches")
    def ws_create_batch(project_id: str, req: BatchCreate) -> dict:
        project = project_or_404(project_id)
        return project.add_batch(req.name, req.kind)

    @app.patch("/api/ws/projects/{project_id}/batches/{batch_id}")
    def ws_update_batch(project_id: str, batch_id: str, req: BatchUpdate) -> dict:
        project = project_or_404(project_id)
        try:
            return project.update_batch(batch_id, name=req.name.strip())
        except KeyError:
            raise HTTPException(404, "no such batch")

    @app.delete("/api/ws/projects/{project_id}/batches/{batch_id}")
    def ws_delete_batch(project_id: str, batch_id: str) -> dict:
        project = project_or_404(project_id)
        # Stop sampling / auto-labeling first; they delete nothing.
        for runs in (ingest, autolabel):
            progress = runs.get(batch_id)
            if progress is not None:
                progress["stop"].set()
                progress["thread"].join(timeout=30)
        try:
            project.delete_batch(batch_id)
        except KeyError:
            raise HTTPException(404, "no such batch")
        return {"deleted": batch_id}

    # -- Auto label: Label Assist on a whole batch

    @app.get("/api/ws/projects/{project_id}/assist")
    def ws_assist_sources(project_id: str) -> dict:
        return assist_choices({}, project_or_404(project_id))

    def run_autolabel(
        project: Project, batch_id: str, ids: list[int], req: AutoLabelRequest,
        model: str | None,
    ) -> None:
        run = autolabel[batch_id]
        classes = [(c.name, c.description) for c in req.classes]
        error = None
        try:
            for image_id in ids:
                if run["stop"].is_set():
                    break
                with project.lock:
                    record = project.images.get(image_id)
                    labeled = record is None or record.get("status") == "labeled"
                    path = None if record is None else project.image_path(image_id)
                if labeled:
                    # Deleted, or labeled by hand since the run started.
                    run["skipped"] += 1
                    continue
                frame = cv2.imread(str(path))
                if frame is None:
                    run["failed"] += 1
                    continue
                try:
                    result = assist.detect(
                        frame, req.source, model, classes, {}, coco_classes()
                    )
                except ValueError as exc:
                    # A bad choice (class, model): the same on every image.
                    error = str(exc)
                    break
                except AssistError as exc:
                    run["failed"] += 1
                    run["last_error"] = str(exc)
                    if run["failed"] >= 3 and not run["done"]:
                        # It isn't working at all: don't try every image.
                        error = str(exc)
                        break
                    continue
                threshold = req.min_confidence
                if threshold is None:
                    threshold = result.get("threshold") or 0.0
                boxes = [b for b in result["boxes"] if b["confidence"] >= threshold]
                with project.lock:
                    record = project.images.get(image_id)
                    if record is None or record.get("status") == "labeled":
                        run["skipped"] += 1
                        continue
                    project.set_boxes(image_id, boxes, status="predicted")
                run["done"] += 1
                run["boxes"] += len(boxes)
                if req.source != "trained":
                    run["model"] = result.get("model", run["model"])
        except Exception as exc:  # noqa: BLE001 - reported on the batch
            error = str(exc)
        finally:
            autolabel.pop(batch_id, None)
            summary = {
                "source": req.source,
                "model": run["model"],
                "classes": [c.name for c in req.classes],
                "images": run["done"],
                "boxes": run["boxes"],
                "failed": run["failed"],
                "stopped": run["stop"].is_set(),
                "error": error,
                "finished": now_iso(),
            }
            try:
                project.update_batch(batch_id, autolabel=summary)
            except KeyError:
                pass  # the batch was deleted
            print(
                f"Auto-labeled {run['done']} images of {project.name} with "
                f"{run['model']}: {run['boxes']} boxes"
                + (f", {run['failed']} failed" if run["failed"] else "")
                + (f" ({error})" if error else "")
            )

    @app.post("/api/ws/projects/{project_id}/batches/{batch_id}/autolabel")
    def ws_autolabel(project_id: str, batch_id: str, req: AutoLabelRequest) -> dict:
        """Run Label Assist on a batch's images in the background; their
        boxes are saved as pre-labels (status "predicted") to review."""
        project = project_or_404(project_id)
        try:
            batch = project.batch(batch_id)
        except KeyError:
            raise HTTPException(404, "no such batch")
        if batch_id in autolabel:
            raise HTTPException(409, "this batch is already being auto-labeled")
        if batch_id in ingest:
            raise HTTPException(409, "frames are still being extracted; wait for them")
        if req.source not in SOURCES:
            raise HTTPException(400, f"unknown source {req.source}")
        if not any(c.name.strip() for c in req.classes):
            raise HTTPException(400, "choose at least one class")
        model = req.model
        names = [c.name.strip() for c in req.classes if c.name.strip()]
        if req.source == "trained":
            folder = resolve(model or "")
            info = read_model(folder / train_rfdetr.MANIFEST)
            if info is None:
                raise HTTPException(400, f"{model} is not a trained model")
            known = {normalize_name(c) for c in info["classes"]}
            if not any(normalize_name(n) in known for n in names):
                raise HTTPException(
                    400, f"{info['name']} finds {', '.join(info['classes'])}; choose some of them"
                )
            model = str(folder)
        elif req.source == "rfdetr":
            coco = {normalize_name(n) for n in coco_classes().values()}
            unknown = [n for n in names if normalize_name(n) not in coco]
            if unknown:
                raise HTTPException(400, f"not COCO classes: {', '.join(unknown)}")
        wanted = {"unlabeled"} if req.images == "unlabeled" else {"unlabeled", "predicted"}
        with project.lock:
            ids = [
                i for i in project.batch_images(batch_id)
                if project.images[i].get("status", "unlabeled") in wanted
            ]
        if not ids:
            raise HTTPException(400, "no images to label in this batch")
        sources = assist.sources({}, [])
        if req.source != "trained" and not sources[req.source]["available"]:
            raise HTTPException(400, sources[req.source]["reason"])
        autolabel[batch_id] = {
            "project": project.id,
            "source": req.source,
            "model": info["name"] if req.source == "trained" else (model or req.source),
            "total": len(ids),
            "done": 0,
            "boxes": 0,
            "failed": 0,
            "skipped": 0,
            "last_error": None,
            "started": time.time(),
            "stop": threading.Event(),
        }
        thread = threading.Thread(
            target=run_autolabel,
            args=(project, batch_id, ids, req, model),
            daemon=True,
        )
        autolabel[batch_id]["thread"] = thread
        thread.start()
        print(f"Auto-labeling {len(ids)} images of {batch['name']} ({req.source})")
        return autolabel_of(batch_id)

    @app.post("/api/ws/projects/{project_id}/batches/{batch_id}/autolabel/stop")
    def ws_autolabel_stop(project_id: str, batch_id: str) -> dict:
        project_or_404(project_id)
        run = autolabel.get(batch_id)
        if run is None:
            raise HTTPException(404, "not being auto-labeled")
        # The image being detected is finished first.
        run["stop"].set()
        return {"stopping": batch_id}

    def run_ingest(
        project: Project, batch_id: str, video: Path, indices: list[int], name: str
    ) -> None:
        progress = ingest[batch_id]

        def on_progress(done: int, total: int) -> None:
            progress["done"] = done

        try:
            added = sample_video(
                project, batch_id, video, indices, name=name,
                on_progress=on_progress, stop=progress["stop"],
            )
            if not progress["stop"].is_set():
                project.update_batch(batch_id, state="ready")
            print(f"Added {added} frames of {name} to {project.name}")
        except Exception as exc:  # noqa: BLE001 - reported on the batch
            print(f"error: sampling {name}: {exc}")
            try:
                project.update_batch(batch_id, state="failed", error=str(exc))
            except KeyError:
                pass
        finally:
            video.unlink(missing_ok=True)
            ingest.pop(batch_id, None)

    @app.post("/api/ws/projects/{project_id}/batches/{batch_id}/files")
    async def ws_upload(
        project_id: str,
        batch_id: str,
        request: Request,
        name: str = Query(min_length=1),
        rate: float | None = Query(None, gt=0, le=240),
    ) -> dict:
        """Add a file sent as the request body to a batch: a picture, or
        a video whose frames are sampled (`rate` per second, default all)
        in the background."""
        project = project_or_404(project_id)
        try:
            batch = project.batch(batch_id)
        except KeyError:
            raise HTTPException(404, "no such batch")
        original = Path(name)
        suffix = original.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            data = await request.body()
            frame = decode_image(data)
            if frame is None:
                raise HTTPException(400, f"can't read {original.name} as an image")
            image_id = project.add_image(batch_id, original.name, frame, data)
            return {"image": image_id}
        if suffix not in VIDEO_EXTENSIONS:
            raise HTTPException(
                400,
                f"{original.name} is not a picture or a video "
                f"({', '.join(sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS))})",
            )
        if batch.get("kind") != "video" or batch.get("source", {}).get("video"):
            raise HTTPException(400, "a video goes in a new video batch of its own")
        # Kept (hidden) until its frames are sampled.
        uploads = project.path / ".uploads"
        uploads.mkdir(exist_ok=True)
        video = uploads / f"{batch_id}{suffix}"
        try:
            with video.open("wb") as f:
                async for chunk in request.stream():
                    f.write(chunk)
            info = video_info(video)
        except (ValueError, OSError):
            video.unlink(missing_ok=True)
            raise HTTPException(400, f"can't read {original.name} as a video")
        indices = sample_indices(info["total_frames"], info["fps"], rate)
        if not indices:
            video.unlink(missing_ok=True)
            raise HTTPException(400, f"{original.name} has no frames")
        project.update_batch(
            batch_id,
            state="processing",
            source={
                "video": original.name,
                "fps": info["fps"],
                "total_frames": info["total_frames"],
                "width": info["width"],
                "height": info["height"],
                "rate": rate,
                "frames": len(indices),
            },
        )
        ingest[batch_id] = {
            "project": project.id,
            "done": 0,
            "total": len(indices),
            "stop": threading.Event(),
        }
        thread = threading.Thread(
            target=run_ingest,
            args=(project, batch_id, video, indices, original.name),
            daemon=True,
        )
        ingest[batch_id]["thread"] = thread
        thread.start()
        print(f"Sampling {len(indices)} frames of {original.name} into {project.name}")
        return {"frames": len(indices)}

    return app


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Review and fix the boxes of a detections file "
            "(--save-detections) in the browser. Without files, pick "
            "them in the UI."
        )
    )
    p.add_argument(
        "files",
        nargs="*",
        metavar="FILE",
        help=(
            "Optional: the detections JSON to edit (in place), and the "
            "source video. Without the video, the one named in the "
            "file's videos[0].file_name is looked up under --root."
        ),
    )
    p.add_argument(
        "--root",
        default=".",
        help=(
            "Folder searched for detections files and videos, and that "
            "relative paths are resolved against (default: current "
            "directory)."
        ),
    )
    p.add_argument(
        "--workspace",
        help="Folder of the projects (default: <root>/workspace).",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the UI in the browser.",
    )
    args = p.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        p.error(f"--root {args.root} is not a folder")
    jsons = [Path(f) for f in args.files if f.lower().endswith(".json")]
    videos = [Path(f) for f in args.files if not f.lower().endswith(".json")]
    if len(args.files) > 2 or len(jsons) > 1 or len(videos) > 1:
        p.error("expected at most one detections JSON and one video")
    if videos and not jsons:
        p.error("a video needs its detections JSON")

    initial = None
    if jsons:
        detections = jsons[0]
        if videos:
            video = videos[0]
        else:
            try:
                source, video = video_of(detections, root)
            except ValueError as exc:
                raise SystemExit(f"error: {exc}")
            if video is None:
                raise SystemExit(
                    f"error: can't find the video {source or '(not recorded)'} "
                    f"under {root}; pass it as well, or use --root"
                )
        initial = (video, detections)

    ws = Workspace(
        Path(args.workspace).resolve() if args.workspace else root / WORKSPACE_DIR
    )
    try:
        app = create_app(root, initial, ws)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")

    url = f"http://{args.host}:{args.port}"
    if initial:
        url += "/annotate"
    print(f"Studio: {url}")
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

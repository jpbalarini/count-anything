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
(process_video.py --no-render), as background jobs.
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
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from detections_io import (
    DetectionsFile,
    is_detections_file,
    load_detections,
    write_detections,
)
import train_rfdetr
import training_data
from file_lookup import VIDEO_EXTENSIONS, best_video, walk_files
from jobs import JobManager
from options import normalize_name
from rfdetr_detector import MODEL_SIZES, WEIGHTS_FILE

UI_DIR = Path(__file__).parent / "annotator_ui"

JPEG_QUALITY = 90
THUMB_WIDTH = 240
THUMB_QUALITY = 75
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

    def close(self) -> None:
        self.reader.close()


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
    root: Path, models: list[Path] | None = None
) -> tuple[list[dict], list[Path]]:
    """Detections files under `root` (newest first) and the videos.
    The model manifests found are added to `models`."""
    videos: list[Path] = []
    jsons: list[Path] = []
    for path in walk_files(root):
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
        "name": folder.name,
        "classes": data.get("classes", []),
        "threshold": data.get("threshold"),
        "base_model": data.get("base_model"),
        "created": data.get("created"),
        "best": data.get("best"),
        "history": data.get("history", []),
        "stopped_early": data.get("stopped_early", False),
        "epochs_run": (data.get("training") or {}).get("epochs_run"),
        "images": dataset.get("images"),
        "sources": [s.get("detections") for s in dataset.get("sources", [])],
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
    detections: list[str] = Field(min_length=1)
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


class DetectRequest(BaseModel):
    model: str = Field(min_length=1)
    video: str = Field(min_length=1)
    output: str | None = None
    # None: the one chosen for the model when it was trained.
    threshold: float | None = Field(None, ge=0, le=1)
    sample_rate: float | None = Field(None, gt=0)


# -- App ------------------------------------------------------------


class OpenRequest(BaseModel):
    detections: str = Field(min_length=1)
    video: str | None = None


def create_app(root: Path, initial: tuple[Path, Path] | None = None) -> FastAPI:
    root = root.resolve()
    state: dict[str, Session | None] = {"session": None}
    lock = threading.Lock()
    jobs = JobManager()
    # A training run outliving the server would hold the GPU unseen.
    atexit.register(jobs.shutdown)

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

    def open_session(video: Path, detections: Path) -> Session:
        session = Session(video, detections)
        with lock:
            old, state["session"] = state["session"], session
        if old is not None:
            old.close()
        if session.warnings:
            print(
                "warning: detections file doesn't match this video: "
                + ", ".join(session.warnings)
            )
        print(f"Opened {rel(session.detections)} on {rel(session.video)}")
        return session

    if initial is not None:
        open_session(*initial)

    def current(s: str | None = None) -> Session:
        """The open session; `s` (the id the page was loaded with) must
        match it, so a tab never edits files another tab replaced."""
        session = state["session"]
        if session is None:
            raise HTTPException(409, "no files are open")
        if s is not None and s != session.id:
            raise HTTPException(
                409, "other files were opened since this page was loaded"
            )
        return session

    def frame_or_404(session: Session, index: int) -> None:
        if index not in session.store.frames:
            raise HTTPException(
                404, f"frame {index} is not in the detections file"
            )

    app = FastAPI(title="Detections annotator")
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

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
        return FileResponse(UI_DIR / "index.html")

    @app.get("/api/files")
    def files() -> dict:
        found, videos = find_files(root)
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
    def project() -> dict:
        session = current()
        store, reader = session.store, session.reader
        return {
            "id": session.id,
            "root": str(root),
            "video": session.video.name,
            "video_path": rel(session.video),
            "detections": session.detections.name,
            "detections_path": rel(session.detections),
            "warnings": session.warnings,
            "meta": store.info,
            "width": reader.width,
            "height": reader.height,
            "fps": reader.fps,
            "total_frames": reader.total_frames,
            "classes": store.classes(),
            "suggestions": session.suggestions,
            "edited": store.edited,
            # [index, box count (-1 = detection failed)] per frame.
            "frames": [
                [i, -1 if store.frames[i] is None else len(store.frames[i])]
                for i in sorted(store.frames)
            ],
        }

    @app.get("/api/frames/{index}")
    def get_frame(index: int, s: str | None = Query(None)) -> dict:
        session = current(s)
        frame_or_404(session, index)
        return {"index": index, "boxes": session.store.frames[index]}

    @app.put("/api/frames/{index}")
    def put_frame(
        index: int, update: FrameUpdate, s: str | None = Query(None)
    ) -> dict:
        session = current(s)
        frame_or_404(session, index)
        boxes = session.store.set_frame(index, update.boxes)
        return {"index": index, "boxes": boxes}

    @app.put("/api/classes")
    def put_classes(update: ClassesUpdate, s: str | None = Query(None)) -> dict:
        session = current(s)
        session.store.set_classes(update.classes)
        return {"classes": session.store.classes()}

    def jpeg(data: bytes) -> Response:
        # Image URLs carry the session id, so caching can't mix videos.
        return Response(
            data,
            media_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=3600"},
        )

    @app.get("/api/frames/{index}/image")
    def frame_image(index: int, s: str | None = Query(None)) -> Response:
        session = current(s)
        frame_or_404(session, index)
        try:
            return jpeg(session.reader.frame(index))
        except KeyError:
            raise HTTPException(404, f"cannot read frame {index}")

    @app.get("/api/frames/{index}/thumbnail")
    def frame_thumbnail(index: int, s: str | None = Query(None)) -> Response:
        session = current(s)
        frame_or_404(session, index)
        try:
            return jpeg(session.reader.thumbnail(index))
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
        session = state["session"]
        if session is not None and session.detections == detections:
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

    @app.get("/api/models")
    def models() -> dict:
        manifests: list[Path] = []
        find_files(root, manifests)
        found = [m for m in map(read_model, manifests) if m is not None]
        found.sort(key=lambda m: -m["modified"])
        return {
            "models": [{**m, "path": rel(m["path"])} for m in found],
            "defaults": train_defaults(),
        }

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

    @app.delete("/api/models")
    def remove_model(path: str = Query(min_length=1)) -> dict:
        folder = resolve(path)
        if folder == root or not folder.is_relative_to(root):
            raise HTTPException(400, f"{path} is not a model under {root}")
        if read_model(folder / train_rfdetr.MANIFEST) is None:
            raise HTTPException(400, f"{path} is not a trained model")
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
        detections = []
        for name in req.detections:
            path = resolve(name)
            if not path.is_file():
                raise HTTPException(400, f"no such file: {name}")
            detections.append(path)
        videos = [video_for(d) for d in detections]
        if req.output and req.output.strip():
            out = resolve(req.output.strip())
            if out.exists() and any(out.iterdir()):
                raise HTTPException(400, f"{req.output} already exists")
        else:
            name = t.default_output(str(detections[0]), req.model).name
            out = unique_path(root / MODELS_DIR / slug(name))

        command = [
            sys.executable, str(TRAIN_SCRIPT),
            *map(str, detections),
            "--videos", *map(str, videos),
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
        if dry:
            return job_response(None, command, True, result)
        job = start("train", f"Train {out.name}", command, result)
        return job_response(job, command, False, result)

    @app.post("/api/jobs/detect")
    def detect(req: DetectRequest, dry: bool = Query(False)) -> dict:
        model = resolve(req.model)
        if read_model(model / train_rfdetr.MANIFEST) is None:
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
        if dry:
            return job_response(None, command, True, result)
        job = start(
            "detect", f"Detect {video.name} with {model.name}", command, result
        )
        return job_response(job, command, False, result)

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

    try:
        app = create_app(root, initial)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")

    url = f"http://{args.host}:{args.port}"
    print(f"Annotator: {url}" + ("" if initial else " (choose the files there)"))
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

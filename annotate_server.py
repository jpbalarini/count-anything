"""Web UI to review and fix saved detections by hand.

Serves the frames of a video next to a detections file written by
`process_video.py --save-detections`, so boxes can be moved, resized,
relabeled, added and deleted. Every change is written back to the same
file; re-render the video with `--load-detections`.

    python annotate_server.py                      # pick the files in the UI
    python annotate_server.py DETECTIONS.json      # video found from meta.source
    python annotate_server.py VIDEO DETECTIONS.json [--port 8000]

The UI lists the detections files found under --root (default: the
current directory), each matched to the video named in its meta.source.

Only the frames the detector ran on (the ones in the file) can be
edited. The first change keeps a copy of the file as it was
(<name>.orig.json, never overwritten).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import threading
import uuid
import webbrowser
from collections import OrderedDict
from functools import cache
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from detections_io import FORMAT_VERSION, load_detections
from frames import sampling_due
from options import normalize_name

UI_DIR = Path(__file__).parent / "annotator_ui"

JPEG_QUALITY = 90
THUMB_WIDTH = 240
THUMB_QUALITY = 75
# Full-size frames kept in memory (encoded JPEG).
FRAME_CACHE_SIZE = 48
# Short forward jumps are read frame by frame instead of seeking, so
# stepping through samples never depends on the codec seeking exactly.
MAX_READ_AHEAD = 120

# Extra keys this tool keeps in the file's "meta" (ignored by
# process_video.py): classes added in the UI that may not be used on any
# frame yet, and the frames that were changed by hand.
META_CLASSES = "annotator_classes"
META_EDITED = "annotator_edited_frames"

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
# Not searched for files.
SKIP_DIRS = {"node_modules", "__pycache__", "venv", "site-packages"}


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
    """The detections file, held in memory and written on every change."""

    def __init__(self, path: Path, width: int, height: int):
        self.path = path
        self.meta, self.frames = load_detections(path)
        self.width = width
        self.height = height
        self._lock = threading.Lock()

    @property
    def edited(self) -> list[int]:
        return sorted(int(i) for i in self.meta.get(META_EDITED, []))

    def classes(self) -> list[str]:
        """Every class name: the ones added in the UI, then the ones on
        the boxes (most frequent first). Names that only differ in case
        or `_`/`-` (the same class for process_video.py) appear once."""
        counts: dict[str, int] = {}
        for items in self.frames.values():
            for item in items or []:
                counts[item["label"]] = counts.get(item["label"], 0) + 1
        names: dict[str, str] = {}
        for name in self.meta.get(META_CLASSES, []):
            names.setdefault(normalize_name(name), name)
        for name in sorted(counts, key=lambda n: -counts[n]):
            names.setdefault(normalize_name(name), name)
        return list(names.values())

    def _clean(self, box: Box) -> dict:
        x1, y1, x2, y2 = box.box
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        x1, x2 = np.clip([x1, x2], 0, self.width)
        y1, y2 = np.clip([y1, y2], 0, self.height)
        return {
            "label": box.label.strip(),
            "confidence": round(box.confidence, 4),
            "box": [round(float(v), 2) for v in (x1, y1, x2, y2)],
        }

    def set_frame(self, index: int, boxes: list[Box]) -> list[dict]:
        with self._lock:
            if index not in self.frames:
                raise KeyError(index)
            cleaned = [self._clean(b) for b in boxes]
            self.frames[index] = cleaned
            edited = set(self.edited)
            edited.add(index)
            self.meta[META_EDITED] = sorted(edited)
            self._write()
            return cleaned

    def set_classes(self, classes: list[str]) -> None:
        with self._lock:
            self.meta[META_CLASSES] = [c.strip() for c in classes if c.strip()]
            self._write()

    def _write(self) -> None:
        backup = self.path.with_name(f"{self.path.stem}.orig{self.path.suffix}")
        if not backup.exists():
            shutil.copy2(self.path, backup)
        tmp = self.path.with_name(f".{self.path.name}.tmp")
        tmp.write_text(
            json.dumps(
                {
                    "version": FORMAT_VERSION,
                    "meta": self.meta,
                    "frames": {
                        str(i): self.frames[i] for i in sorted(self.frames)
                    },
                },
                separators=(",", ":"),
            )
        )
        os.replace(tmp, self.path)


@cache
def coco_class_names() -> tuple[str, ...]:
    """COCO names, offered as suggestions for rfdetr files (whose labels
    must be COCO classes to load). Empty if rfdetr isn't importable."""
    try:
        from options import load_coco_classes

        return tuple(name for _, name in sorted(load_coco_classes().items()))
    except Exception:
        return ()


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
        meta = self.store.meta
        self.suggestions = (
            list(coco_class_names()) if meta.get("detector") == "rfdetr" else []
        )
        self.warnings = [
            f"{key} {meta[key]} (video: {actual})"
            for key, actual in (
                ("width", self.reader.width),
                ("height", self.reader.height),
                ("total_frames", self.reader.total_frames),
            )
            if meta.get(key) not in (None, actual)
        ]
        threading.Thread(
            target=self.reader.warm_thumbnails,
            args=(sorted(self.store.frames),),
            daemon=True,
        ).start()

    def close(self) -> None:
        self.reader.close()


# -- Finding files --------------------------------------------------


def peek_meta(path: Path) -> dict | None:
    """The "meta" of a detections file, or None if it isn't one.

    Only the start of the file is read: process_video.py writes "meta"
    before "frames", so this stays fast on big files."""
    try:
        with path.open("rb") as f:
            head = f.read(256 * 1024).decode("utf-8", "ignore")
    except OSError:
        return None
    if '"version"' not in head:
        return None
    match = re.search(r'"meta"\s*:\s*', head)
    if not match:
        return None
    try:
        meta, _ = json.JSONDecoder().raw_decode(head, match.end())
    except ValueError:
        return None
    return meta if isinstance(meta, dict) else None


def walk_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if not d.startswith(".") and d not in SKIP_DIRS
        )
        for name in filenames:
            if not name.startswith("."):
                yield Path(dirpath) / name


def best_video(json_path: Path, name: str, videos: list[Path]) -> Path | None:
    """The video called `name`, the closest one to the JSON if several."""
    candidates = [v for v in videos if v.name == name]
    if not candidates:
        return None

    def closeness(video: Path) -> tuple[int, int]:
        common = os.path.commonpath([video, json_path])
        return (-len(Path(common).parts), len(video.parts))

    return min(candidates, key=closeness)


def sample_count(meta: dict) -> int | None:
    """How many frames the detector ran on (same rule as process_video)."""
    total, step = meta.get("total_frames"), meta.get("sample_step") or 1
    if not total:
        return None
    count, next_at = 0, 0.0
    for index in range(total):
        due, next_at = sampling_due(index, next_at, step)
        count += due
    return count


def find_files(root: Path) -> tuple[list[dict], list[Path]]:
    """Detections files under `root` (newest first) and the videos."""
    videos: list[Path] = []
    jsons: list[Path] = []
    for path in walk_files(root):
        suffix = path.suffix.lower()
        if suffix in VIDEO_EXTENSIONS:
            videos.append(path)
        elif suffix == ".json" and not path.name.endswith(".orig.json"):
            jsons.append(path)

    found = []
    for path in jsons:
        meta = peek_meta(path)
        if meta is None or "detector" not in meta:
            continue
        source = meta.get("source")
        video = best_video(path, source, videos) if source else None
        found.append({
            "path": path,
            "detector": meta.get("detector"),
            "source": source,
            "video": video,
            "samples": sample_count(meta),
            "edited": len(meta.get(META_EDITED, [])),
            "modified": path.stat().st_mtime,
        })
    found.sort(key=lambda f: -f["modified"])
    return found, sorted(videos)


# -- App ------------------------------------------------------------


class OpenRequest(BaseModel):
    detections: str = Field(min_length=1)
    video: str | None = None


def create_app(root: Path, initial: tuple[Path, Path] | None = None) -> FastAPI:
    root = root.resolve()
    state: dict[str, Session | None] = {"session": None}
    lock = threading.Lock()

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

    @app.middleware("http")
    async def revalidate_ui(request, call_next):
        # Always revalidate the UI files, so an updated UI is picked up.
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache"
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
            source = (peek_meta(detections) or {}).get("source")
            _, videos = find_files(root)
            video = best_video(detections, source, videos) if source else None
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
            "meta": store.meta,
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
            "file's meta.source is looked up under --root."
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
            source = (peek_meta(detections) or {}).get("source")
            _, found = find_files(root)
            video = best_video(Path(os.path.abspath(detections)), source, found) if source else None
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

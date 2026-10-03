"""locate-anything.cpp detector for process_video.py.

Wraps locate-anything.cpp (https://github.com/mudler/locate-anything.cpp),
an open-vocabulary detector (NVIDIA LocateAnything-3B on ggml). Like the
cloud detector it is meant to run on sampled frames. Two ways to run it:

* Resident engine (`lib_path`): a child process (locate_anything_worker.py)
  loads liblocate_anything once and answers one request per frame, so the
  model is not reloaded every time. If it crashes (e.g. a Metal fault)
  that frame is skipped and a new worker is started.
* `locate-anything-cli` (fallback, no library needed): every frame is
  written to a temporary JPEG and the CLI is run on it

      locate-anything-cli detect --model M --input frame.jpg \\
          --prompt "Locate all the instances that matches the following \\
description: a</c>b." --output boxes.json

  which loads the model on every call.

Both give `{"detections": [{"label": ..., "box": [x1,y1,x2,y2]}]}` (boxes
are pixels of the image they were given). There is no confidence score,
so every detection gets confidence 1.0.
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np
import supervision as sv

CLI_NAME = "locate-anything-cli"
WORKER_SCRIPT = Path(__file__).with_name("locate_anything_worker.py")
WORKER_OK = b"\x00"
DEFAULT_LOCATE_MODE = "hybrid"
DEFAULT_LOCATE_SAMPLE_RATE = 1.0
PROMPT_PREFIX = (
    "Locate all the instances that matches the following description: "
)
CATEGORY_SEPARATOR = "</c>"


def _normalize(name: str) -> str:
    return (
        name.strip().strip(".").lower().replace("_", " ").replace("-", " ")
    )


class _Worker:
    """A locate_anything_worker.py child process holding the model."""

    def __init__(
        self, lib_path: str, model_path: str, threads: int, mode: str,
        prompt: str,
    ):
        # ggml is chatty on stderr; keep it to report why the worker died.
        self._stderr = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(
            [
                sys.executable, str(WORKER_SCRIPT),
                lib_path, model_path, str(threads), mode, prompt,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
        )
        try:
            self._read()  # blocks until the model is loaded
        except (RuntimeError, OSError):
            self.close()
            raise

    def alive(self) -> bool:
        return self.proc.poll() is None

    def _stderr_tail(self) -> str:
        try:
            self._stderr.seek(0)
            lines = self._stderr.read().decode(errors="replace").splitlines()
        except (OSError, ValueError):
            return ""
        return next((ln.strip() for ln in reversed(lines) if ln.strip()), "")

    def _read(self) -> bytes:
        """One reply: the body of an "ok" message, else RuntimeError."""
        header = self.proc.stdout.read(4)
        payload = b""
        if len(header) == 4:
            (size,) = struct.unpack("<I", header)
            payload = self.proc.stdout.read(size)
        if not payload:  # EOF: the worker is gone
            self.proc.wait()
            raise RuntimeError(
                f"worker exited with code {self.proc.returncode}: "
                f"{self._stderr_tail() or 'no output'}"
            )
        status, body = payload[:1], payload[1:]
        if status != WORKER_OK:
            raise RuntimeError(body.decode(errors="replace"))
        return body

    def locate(self, image: bytes) -> list[dict]:
        """Run detection on an encoded image; returns the raw detections."""
        try:
            self.proc.stdin.write(struct.pack("<I", len(image)) + image)
            self.proc.stdin.flush()
        except OSError:  # broken pipe: the worker died
            pass  # _read() below reports why
        return json.loads(self._read())["detections"]

    def close(self) -> None:
        try:
            self.proc.stdin.close()  # EOF tells the worker to shut down
        except OSError:
            pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.proc.stdout.close()
        self._stderr.close()


class LocateAnythingDetector:
    def __init__(
        self,
        model_path: str,
        class_names: dict[int, str],
        mode: str = DEFAULT_LOCATE_MODE,
        threads: int = 0,
        lib_path: str | None = None,
    ):
        if not Path(model_path).is_file():
            raise RuntimeError(f"locate-anything model not found: {model_path}")

        self.model_path = str(model_path)
        self.mode = mode
        self.threads = threads
        self.lib_path = str(lib_path) if lib_path else None
        self.worker: _Worker | None = None
        self.cli: str | None = None
        if self.lib_path is not None:
            if not Path(self.lib_path).is_file():
                raise RuntimeError(
                    f"locate-anything library not found: {self.lib_path}"
                )
        else:
            self.cli = shutil.which(CLI_NAME)
            if self.cli is None:
                raise RuntimeError(
                    f"`{CLI_NAME}` not found in PATH (or give --locate-lib)"
                )
            print(
                f"note: no --locate-lib, using {CLI_NAME}, which reloads "
                "the model on every frame (see the README to load it once)"
            )

        self.name_to_id = {
            _normalize(name): class_id
            for class_id, name in class_names.items()
        }
        self.prompt = (
            PROMPT_PREFIX
            + CATEGORY_SEPARATOR.join(class_names.values())
            + "."
        )

        self.calls = 0
        self.failures = 0
        self._warned_labels: set[str] = set()

        if self.lib_path is not None:
            # Started now so a bad library / model is a startup error
            # and the load time isn't counted as the first frame's.
            self._start_worker()

    @property
    def backend(self) -> str:
        return (
            "resident engine" if self.lib_path else "CLI, reloads per frame"
        )

    def _start_worker(self) -> None:
        self.worker = _Worker(
            self.lib_path, self.model_path, self.threads, self.mode,
            self.prompt,
        )

    def close(self) -> None:
        """Stop the worker (if any); it is restarted if used again."""
        if self.worker is not None:
            self.worker.close()
            self.worker = None

    @staticmethod
    def _empty() -> sv.Detections:
        return sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.empty(0, dtype=np.float32),
            class_id=np.empty(0, dtype=int),
        )

    def _class_id(self, label: str) -> int | None:
        """Map the label the model wrote back to one of our classes."""
        key = _normalize(label)
        if key in self.name_to_id:
            return self.name_to_id[key]
        # Tolerate the model pluralizing / singularizing.
        for name, class_id in self.name_to_id.items():
            if name.rstrip("s") == key.rstrip("s"):
                return class_id
        if len(self.name_to_id) == 1:
            return next(iter(self.name_to_id.values()))
        if key not in self._warned_labels:
            self._warned_labels.add(key)
            print(
                f"warning: locate-anything returned unknown label "
                f"'{label}', ignoring those boxes"
            )
        return None

    def _locate_anything_worker(self, frame_bgr: np.ndarray) -> list[dict]:
        """Raw detections from the resident worker (starting it if needed)."""
        if self.worker is None or not self.worker.alive():
            if self.worker is not None:
                self.worker.close()
            print("note: (re)starting the locate-anything worker")
            self._start_worker()
        ok, jpeg = cv2.imencode(
            ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
        if not ok:
            raise RuntimeError("could not encode the frame")
        return self.worker.locate(jpeg.tobytes())

    def _locate_cli(self, frame_bgr: np.ndarray) -> list[dict]:
        """Raw detections from one `locate-anything-cli detect` call."""
        with tempfile.TemporaryDirectory(prefix="locate_") as tmp:
            image_path = Path(tmp) / "frame.jpg"
            out_path = Path(tmp) / "boxes.json"
            cv2.imwrite(
                str(image_path),
                frame_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

            cmd = [
                self.cli,
                "detect",
                "--model", self.model_path,
                "--input", str(image_path),
                "--prompt", self.prompt,
                "--output", str(out_path),
                "--mode", self.mode,
                "--threads", str(self.threads),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    proc.stderr.strip().splitlines()[-1]
                    if proc.stderr.strip()
                    else f"exit code {proc.returncode}"
                )
            return json.loads(out_path.read_text())["detections"]

    def detect(self, frame_bgr: np.ndarray) -> sv.Detections | None:
        """Return detections, or None if the detection call failed."""
        height, width = frame_bgr.shape[:2]
        self.calls += 1

        try:
            items = (
                self._locate_anything_worker(frame_bgr)
                if self.lib_path
                else self._locate_cli(frame_bgr)
            )
        except (RuntimeError, OSError, ValueError, KeyError) as exc:
            self.failures += 1
            print(
                f"warning: locate-anything ({self.backend}) failed, "
                f"skipping sample: {exc}"
            )
            return None

        boxes, ids = [], []
        for item in items:
            try:
                class_id = self._class_id(item["label"])
                x1, y1, x2, y2 = (float(v) for v in item["box"])
            except (KeyError, TypeError, ValueError):
                continue
            if class_id is None:
                continue
            x1, x2 = sorted((min(max(x1, 0), width), min(max(x2, 0), width)))
            y1, y2 = sorted((min(max(y1, 0), height), min(max(y2, 0), height)))
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            boxes.append([x1, y1, x2, y2])
            ids.append(class_id)

        if not boxes:
            return self._empty()
        return sv.Detections(
            xyxy=np.array(boxes, dtype=np.float32),
            confidence=np.ones(len(boxes), dtype=np.float32),
            class_id=np.array(ids, dtype=int),
        )

    def detect_frames(
        self,
        frames: Iterable[tuple[int, np.ndarray]],
        on_done: Callable[[int], None] | None = None,
    ) -> dict[int, sv.Detections | None]:
        """Detect on many (key, frame) pairs, one frame at a time."""
        results: dict[int, sv.Detections | None] = {}
        try:
            for key, frame in frames:
                results[key] = self.detect(frame)
                if on_done is not None:
                    on_done(len(results))
        finally:
            self.close()
        return results

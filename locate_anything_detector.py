"""locate-anything.cpp detector for process_video.py.

Wraps locate-anything.cpp (https://github.com/mudler/locate-anything.cpp),
an open-vocabulary detector (NVIDIA LocateAnything-3B on ggml). Like the
cloud detector it is meant to run on sampled frames.

Each class is asked for separately ("...description: apple."), so a frame
takes one model call per class. The model accepts several classes in one
prompt (`apple</c>box`), but then only writes the first one as a label
and gives it every box (the crates came back as "apple"), and tends to
repeat boxes until it runs out of tokens.

Two ways to run it:

* Resident engine (`lib_path`): a child process (locate_anything_worker.py)
  loads liblocate_anything once and answers one request (prompt + frame)
  at a time, so the model is not reloaded every time. If it crashes (e.g.
  a Metal fault) that frame is skipped and a new worker is started.
* `locate-anything-cli` (fallback, no library needed): every frame is
  written to a temporary JPEG and the CLI is run on it, once per class

      locate-anything-cli detect --model M --input frame.jpg \\
          --prompt "Locate all the instances that matches the following \\
description: apple." --output boxes.json

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


def class_prompt(name: str) -> str:
    return f"{PROMPT_PREFIX}{name.strip()}."


def _message(data: bytes) -> bytes:
    return struct.pack("<I", len(data)) + data


class _Worker:
    """A locate_anything_worker.py child process holding the model."""

    def __init__(self, lib_path: str, model_path: str, threads: int, mode: str):
        # ggml is chatty on stderr; keep it to report why the worker died.
        self._stderr = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(
            [
                sys.executable, str(WORKER_SCRIPT),
                lib_path, model_path, str(threads), mode,
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

    def locate(self, prompt: str, image: bytes) -> list[dict]:
        """Run detection on an encoded image; returns the raw detections."""
        try:
            self.proc.stdin.write(_message(prompt.encode()) + _message(image))
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

        # One prompt per class (see the module docstring).
        self.prompts = {
            class_id: class_prompt(name) for class_id, name in class_names.items()
        }

        self.calls = 0  # frames
        self.failures = 0
        self.last_error: str | None = None  # why the last failed call failed

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
            self.lib_path, self.model_path, self.threads, self.mode
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

    def _locate_anything_worker(self, jpeg: bytes) -> dict[int, list[dict]]:
        """Raw detections per class from the resident worker (starting it
        if needed)."""
        if self.worker is None or not self.worker.alive():
            if self.worker is not None:
                self.worker.close()
            print("note: (re)starting the locate-anything worker")
            self._start_worker()
        return {
            class_id: self.worker.locate(prompt, jpeg)
            for class_id, prompt in self.prompts.items()
        }

    def _locate_cli(self, jpeg: bytes) -> dict[int, list[dict]]:
        """Raw detections per class, one `locate-anything-cli detect`
        call each."""
        found = {}
        with tempfile.TemporaryDirectory(prefix="locate_") as tmp:
            image_path = Path(tmp) / "frame.jpg"
            out_path = Path(tmp) / "boxes.json"
            image_path.write_bytes(jpeg)
            for class_id, prompt in self.prompts.items():
                cmd = [
                    self.cli,
                    "detect",
                    "--model", self.model_path,
                    "--input", str(image_path),
                    "--prompt", prompt,
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
                found[class_id] = json.loads(out_path.read_text())["detections"]
        return found

    def detect(self, frame_bgr: np.ndarray) -> sv.Detections | None:
        """Return detections, or None if a detection call failed (a
        frame with some classes missing would look like a frame without
        them)."""
        height, width = frame_bgr.shape[:2]
        self.calls += 1

        try:
            ok, jpeg = cv2.imencode(
                ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            if not ok:
                raise RuntimeError("could not encode the frame")
            found = (
                self._locate_anything_worker(jpeg.tobytes())
                if self.lib_path
                else self._locate_cli(jpeg.tobytes())
            )
        except (RuntimeError, OSError, ValueError, KeyError) as exc:
            self.failures += 1
            self.last_error = str(exc)
            print(
                f"warning: locate-anything ({self.backend}) failed, "
                f"skipping sample: {exc}"
            )
            return None

        boxes, ids = [], []
        for class_id, items in found.items():
            seen = set()
            for item in items:
                try:
                    x1, y1, x2, y2 = (float(v) for v in item["box"])
                except (KeyError, TypeError, ValueError):
                    continue
                x1, x2 = sorted((min(max(x1, 0), width), min(max(x2, 0), width)))
                y1, y2 = sorted((min(max(y1, 0), height), min(max(y2, 0), height)))
                # The model sometimes repeats a box word for word.
                key = (round(x1), round(y1), round(x2), round(y2))
                if x2 - x1 < 1 or y2 - y1 < 1 or key in seen:
                    continue
                seen.add(key)
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

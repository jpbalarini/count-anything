"""Background jobs for the annotator: training (train_rfdetr.py) and
detection (process_video.py --no-render), run as subprocesses.

Jobs are only kept in memory: the list starts empty when the server
starts. What a job makes (a model folder, a detections file) is on disk
and stays when the job is removed from the list.

A job's output is kept (the last lines) and parsed for progress:
`@progress {json}` lines (train_rfdetr.py --progress-json) and tqdm
progress bars (process_video.py). Only one job runs at a time, since
both use the GPU.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
import uuid
from collections import deque

PROGRESS_PREFIX = "@progress "
LOG_LINES = 400
# Seconds a job gets to stop (a training run scores the model and saves
# the best weights) before it is killed.
STOP_GRACE = 120
# "Detecting:  45%|████▌     | 272/605 [00:10<00:12, 26.1frame/s]"
TQDM = re.compile(r"^(?P<desc>[^:|]{1,60}):\s*(?P<pct>\d+)%\|.*?\|\s*(?P<n>\d+)/(?P<total>\d+)")


class Job:
    def __init__(self, kind: str, title: str, command: list[str], cwd: str,
                 result: dict | None = None,
                 stop_signal: int = signal.SIGINT):
        self.id = uuid.uuid4().hex[:10]
        self.kind = kind
        self.title = title
        self.command = command
        self.cwd = cwd
        # What the job makes (paths), shown when it is done.
        self.result = result or {}
        self.stop_signal = stop_signal
        self.status = "running"
        self.returncode: int | None = None
        self.started = time.time()
        self.ended: float | None = None
        self.progress: dict = {}
        self.log: deque[str] = deque(maxlen=LOG_LINES)
        self._partial = ""  # a line still being written (tqdm \r updates)
        self._stopping = False
        self._lock = threading.Lock()
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        self._proc = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            # Its own process group: Ctrl-C in the server's terminal
            # doesn't reach it, and it can be stopped as a whole.
            start_new_session=True,
        )
        threading.Thread(target=self._read, daemon=True).start()

    @property
    def running(self) -> bool:
        return self.status == "running"

    def _line(self, line: str, final: bool) -> None:
        """A complete line (`final`) or a progress bar redraw."""
        if line.startswith(PROGRESS_PREFIX):
            try:
                self.progress.update(json.loads(line[len(PROGRESS_PREFIX):]))
            except ValueError:
                pass
            return
        match = TQDM.match(line.strip())
        if match:
            self.progress["bar"] = {
                "label": match["desc"].strip(),
                "done": int(match["n"]),
                "total": int(match["total"]),
            }
        if final:
            if line.strip():
                self.log.append(line.rstrip())
            self._partial = ""
        else:
            self._partial = line.rstrip()

    def _read(self) -> None:
        buffer = ""
        stream = self._proc.stdout
        while True:
            chunk = stream.read1(4096) if hasattr(stream, "read1") else stream.read(4096)
            if not chunk:
                break
            buffer += chunk.decode("utf-8", "replace")
            # \n ends a line; \r redraws it (progress bars).
            while True:
                n, r = buffer.find("\n"), buffer.find("\r")
                ends = [i for i in (n, r) if i >= 0]
                if not ends:
                    break
                cut = min(ends)
                with self._lock:
                    self._line(buffer[:cut], final=buffer[cut] == "\n")
                buffer = buffer[cut + 1:]
        with self._lock:
            if buffer:
                self._line(buffer, final=True)
        code = self._proc.wait()
        with self._lock:
            self.returncode = code
            self.ended = time.time()
            if self._stopping:
                self.status = "stopped"
            else:
                self.status = "done" if code == 0 else "failed"

    def stop(self) -> None:
        """Ask the job to stop (its `stop_signal`; a training run then
        keeps its best checkpoint so far), and kill it if it doesn't."""
        if not self.running or self._stopping:
            return
        self._stopping = True
        try:
            self._proc.send_signal(self.stop_signal)
        except ProcessLookupError:
            return
        threading.Timer(STOP_GRACE, self.kill).start()

    def kill(self) -> None:
        if self._proc.poll() is None:
            try:
                os.killpg(self._proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                self._proc.kill()

    def to_dict(self, log: bool = False) -> dict:
        with self._lock:
            data = {
                "id": self.id,
                "kind": self.kind,
                "title": self.title,
                "status": self.status,
                "stopping": self._stopping and self.running,
                "returncode": self.returncode,
                "started": self.started,
                "ended": self.ended,
                "progress": dict(self.progress),
                "result": self.result,
                "last_line": self._partial or (self.log[-1] if self.log else ""),
            }
            if log:
                data["log"] = list(self.log) + ([self._partial] if self._partial else [])
                data["command"] = self.command
            return data


class JobManager:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def running(self) -> Job | None:
        with self._lock:
            return next((j for j in self.jobs.values() if j.running), None)

    def start(self, kind: str, title: str, command: list[str], cwd: str,
              result: dict | None = None,
              stop_signal: int = signal.SIGINT) -> Job:
        busy = self.running()
        with self._lock:
            if busy is not None:
                raise RuntimeError(f"'{busy.title}' is still running")
            job = Job(kind, title, command, cwd, result, stop_signal)
            self.jobs[job.id] = job
            return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def list(self) -> list[dict]:
        with self._lock:
            jobs = list(reversed(self.jobs.values()))
        return [j.to_dict() for j in jobs]

    def remove(self, job_id: str) -> None:
        """Take a finished job off the list (its files are kept)."""
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.running:
                raise RuntimeError("the job is still running; stop it first")
            del self.jobs[job_id]

    def using(self, path: str) -> Job | None:
        """A running job that has `path` on its command line."""
        with self._lock:
            return next(
                (j for j in self.jobs.values() if j.running and path in j.command),
                None,
            )

    def shutdown(self) -> None:
        for job in self.jobs.values():
            if job.running:
                job.kill()

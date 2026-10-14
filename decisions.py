"""Optional per-object classification with the OpenAI Decisions API.

Every tracked object is cropped once (from the frame where it is seen
best) and sent to `client.decisions.create` with one choice question,
e.g. ripe / unripe / defect. What to ask is read from a YAML file
(--decisions config.yaml), so the same code works for any video:

    model: gpt-6-luna          # optional, the default
    context: >                 # text sent along with each crop
      One apple, cropped from an overhead camera over a packhouse
      conveyor belt.
    question:
      name: quality            # optional
      title: Quality           # HUD section title (default: Quality)
      instructions: Is this apple ready to be packed?
      choices:
        - value: ripe
          description: Red or yellow skin, firm, no visible damage.
          color: "#5DCAA5"     # optional, hex
          good: true           # counts towards the HUD headline %
        - value: unripe
          description: Mostly green skin.
          color: "#C7D36F"
        - value: defect
          description: Bruises, cuts, rot or mold.
          color: "#F0997B"
          flag: true           # always labeled on the video
    min_confidence: 0.3        # below this the object is "unverified"
    crop_padding: 0.15         # extra margin around the box (x box size)
    crop_max_side: 384         # crops are downscaled to this (px)
    image_detail: low          # low | high | auto | original
    concurrency: 16            # requests in flight at once

Needs OPENAI_API_KEY (read from .env too).

Every answer is saved to a JSON file (by default next to the output
video, see --save-decisions) with the frame and box each crop was taken
from. --load-decisions reuses those answers for the objects whose crop
matches (same frame, same box), so re-rendering doesn't call the API
again; objects that don't match are classified as usual.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

DEFAULT_DECISIONS_MODEL = "gpt-6-luna"
DEFAULT_CONCURRENCY = 16

# Used for choices without an explicit color: good choices get the
# first one, the others are cycled through the rest.
GOOD_COLOR = "#5DCAA5"
OTHER_CHOICE_COLORS = ["#F0997B", "#C7D36F", "#AFA9EC", "#85B7EB", "#ED93B1"]


@dataclass
class Choice:
    value: str
    description: str = ""
    color: str = GOOD_COLOR
    good: bool = False
    flag: bool = False


@dataclass
class DecisionsConfig:
    instructions: str
    choices: list[Choice]
    model: str = DEFAULT_DECISIONS_MODEL
    name: str = "classification"
    title: str = "Quality"
    context: str = ""
    min_confidence: float = 0.0
    crop_padding: float = 0.15
    crop_max_side: int = 384
    image_detail: str = "low"
    concurrency: int = DEFAULT_CONCURRENCY
    path: str = ""

    @property
    def choice_by_value(self) -> dict[str, Choice]:
        return {c.value: c for c in self.choices}

    @classmethod
    def load(cls, path: str | Path) -> "DecisionsConfig":
        import yaml

        path = Path(path)
        try:
            raw = yaml.safe_load(path.read_text())
        except FileNotFoundError:
            raise ValueError(f"decisions config not found: {path}")
        except yaml.YAMLError as exc:
            raise ValueError(f"{path}: invalid YAML ({exc})") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: expected a mapping at the top level")

        question = raw.get("question")
        if not isinstance(question, dict):
            raise ValueError(f"{path}: missing `question:` section")
        instructions = str(question.get("instructions") or "").strip()
        if not instructions:
            raise ValueError(f"{path}: question.instructions is required")

        raw_choices = question.get("choices") or []
        if not isinstance(raw_choices, list) or len(raw_choices) < 2:
            raise ValueError(f"{path}: question.choices needs at least 2 entries")

        choices: list[Choice] = []
        other_i = 0
        for i, item in enumerate(raw_choices):
            if isinstance(item, str):
                item = {"value": item}
            if not isinstance(item, dict) or not str(item.get("value", "")).strip():
                raise ValueError(f"{path}: question.choices[{i}] needs a `value`")
            good = bool(item.get("good", False))
            color = item.get("color")
            if not color:
                if good:
                    color = GOOD_COLOR
                else:
                    color = OTHER_CHOICE_COLORS[other_i % len(OTHER_CHOICE_COLORS)]
                    other_i += 1
            color = str(color)
            if not color.startswith("#"):
                color = f"#{color}"
            choices.append(
                Choice(
                    value=str(item["value"]).strip(),
                    description=str(item.get("description") or "").strip(),
                    color=color,
                    good=good,
                    flag=bool(item.get("flag", False)),
                )
            )
        values = [c.value for c in choices]
        if len(set(values)) != len(values):
            raise ValueError(f"{path}: question.choices values must be unique")
        # Without an explicit `good`, the first choice is the good one.
        if not any(c.good for c in choices):
            choices[0].good = True

        config = cls(
            instructions=instructions,
            choices=choices,
            model=str(raw.get("model") or DEFAULT_DECISIONS_MODEL),
            name=str(question.get("name") or "classification"),
            title=str(question.get("title") or "Quality"),
            context=str(raw.get("context") or "").strip(),
            min_confidence=float(raw.get("min_confidence", 0.0)),
            crop_padding=float(raw.get("crop_padding", 0.15)),
            crop_max_side=int(raw.get("crop_max_side", 384)),
            image_detail=str(raw.get("image_detail", "low")),
            concurrency=int(raw.get("concurrency", DEFAULT_CONCURRENCY)),
            path=str(path),
        )
        if not 0 <= config.min_confidence <= 1:
            raise ValueError(f"{path}: min_confidence must be in [0, 1]")
        if config.crop_padding < 0:
            raise ValueError(f"{path}: crop_padding must be >= 0")
        if config.crop_max_side < 32:
            raise ValueError(f"{path}: crop_max_side must be >= 32")
        if config.image_detail not in ("low", "high", "auto", "original"):
            raise ValueError(
                f"{path}: image_detail must be low, high, auto or original"
            )
        if config.concurrency < 1:
            raise ValueError(f"{path}: concurrency must be >= 1")
        return config


@dataclass
class Verdict:
    """The answer for one object.

    `choice` is what the model answered; `value` is what is shown, None
    when unverified (status `refused`, `failed`, or `low_confidence`:
    below min_confidence).
    """

    value: str | None
    confidence: float = 0.0
    probabilities: dict[str, float] = field(default_factory=dict)
    choice: str | None = None
    status: str = "ok"

    @classmethod
    def answered(
        cls,
        choice: str,
        confidence: float,
        probabilities: dict[str, float],
        min_confidence: float,
    ) -> "Verdict":
        if confidence < min_confidence:
            return cls(None, confidence, probabilities, choice, "low_confidence")
        return cls(choice, confidence, probabilities, choice, "ok")


def crop_box(
    frame: np.ndarray, xyxy: np.ndarray, padding: float
) -> np.ndarray | None:
    """The box plus `padding` x its size on each side, clipped to the frame."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    pad_x, pad_y = (x2 - x1) * padding, (y2 - y1) * padding
    x1, y1 = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    x2, y2 = min(w, int(np.ceil(x2 + pad_x))), min(h, int(np.ceil(y2 + pad_y)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return frame[y1:y2, x1:x2].copy()


class DecisionsClassifier:
    """Classifies crops in parallel with the Decisions API."""

    def __init__(self, config: DecisionsConfig):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "--decisions needs the OPENAI_API_KEY environment variable "
                "(or put it in .env)"
            )
        from openai import OpenAI

        self.config = config
        # The SDK retries 429/5xx with backoff.
        self.client = OpenAI(max_retries=6, timeout=60)
        self.calls = 0
        self.failures = 0
        self.refusals = 0
        self.input_tokens = 0
        self._lock = threading.Lock()

    def _data_url(self, crop: np.ndarray) -> str:
        h, w = crop.shape[:2]
        scale = self.config.crop_max_side / max(h, w)
        if scale < 1:
            crop = cv2.resize(
                crop,
                (max(1, round(w * scale)), max(1, round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("could not encode crop")
        data = base64.standard_b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{data}"

    def _question(self) -> dict:
        choices = []
        for c in self.config.choices:
            choice = {"value": c.value}
            if c.description:
                choice["description"] = c.description
            choices.append(choice)
        return {
            "type": "choice",
            "name": self.config.name,
            "instructions": self.config.instructions,
            "choices": choices,
        }

    def classify_one(self, crop: np.ndarray) -> Verdict:
        content: list[dict] = []
        if self.config.context:
            content.append({"type": "input_text", "text": self.config.context})
        content.append(
            {
                "type": "input_image",
                "image_url": self._data_url(crop),
                "detail": self.config.image_detail,
            }
        )
        decision = self.client.decisions.create(
            model=self.config.model,
            input=[{"role": "user", "content": content}],
            questions=[self._question()],
        )
        with self._lock:
            self.input_tokens += decision.usage.input_tokens

        answer = decision.answers[0]
        if answer.type == "refusal":
            with self._lock:
                self.refusals += 1
            return Verdict(None, status="refused")
        if answer.type != "choice":
            raise RuntimeError(f"unexpected answer type {answer.type!r}")

        return Verdict.answered(
            str(answer.choice),
            answer.confidence,
            {str(p.value): p.probability for p in answer.probabilities},
            self.config.min_confidence,
        )

    def classify(
        self,
        crops: dict[int, np.ndarray],
        on_done: Callable[[int], None] | None = None,
    ) -> dict[int, Verdict]:
        """{key: crop} -> {key: Verdict}, `concurrency` requests at a time."""
        verdicts: dict[int, Verdict] = {}
        done = 0
        with ThreadPoolExecutor(max_workers=self.config.concurrency) as pool:
            futures = {
                pool.submit(self.classify_one, crop): key
                for key, crop in crops.items()
            }
            for future in as_completed(futures):
                key = futures[future]
                self.calls += 1
                try:
                    verdicts[key] = future.result()
                except Exception as exc:  # keep going, mark unverified
                    self.failures += 1
                    print(f"warning: decisions call for track {key} failed: {exc}")
                    verdicts[key] = Verdict(None, status="failed")
                done += 1
                if on_done:
                    on_done(done)
        return verdicts

    def stats(self) -> str:
        return (
            f"Decisions calls: {self.calls} ({self.failures} failed, "
            f"{self.refusals} refused), input tokens: {self.input_tokens}"
        )


# ---------------------------------------------------------
# Saved answers (--save-decisions / --load-decisions)
# ---------------------------------------------------------

DECISIONS_FORMAT = "count-anything-decisions"
# Two boxes are the same crop when they overlap at least this much.
SAME_BOX_IOU = 0.9


def fingerprint(config: DecisionsConfig) -> dict:
    """What decides the answers. Saved answers are only reused when this
    matches (min_confidence is applied again on load, so it may change)."""
    return {
        "model": config.model,
        "context": config.context,
        "name": config.name,
        "instructions": config.instructions,
        "choices": [[c.value, c.description] for c in config.choices],
        "crop_padding": config.crop_padding,
        "crop_max_side": config.crop_max_side,
        "image_detail": config.image_detail,
    }


def _fingerprint_hash(config: DecisionsConfig) -> str:
    raw = json.dumps(fingerprint(config), sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def save_decisions(
    path: Path,
    config: DecisionsConfig,
    video: str,
    objects: list[dict],
) -> None:
    """`objects`: {track_id, frame, box, verdict (Verdict), reused}."""
    data = {
        "format": DECISIONS_FORMAT,
        "date_created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "video": video,
        "config_file": config.path,
        "config_hash": _fingerprint_hash(config),
        "config": fingerprint(config) | {"min_confidence": config.min_confidence},
        "objects": [
            {
                "track_id": o["track_id"],
                "frame": o["frame"],
                "box": [round(float(v), 2) for v in o["box"]],
                "reused": o.get("reused", False),
                **{
                    k: v
                    for k, v in asdict(o["verdict"]).items()
                    if k != "value"  # depends on min_confidence
                },
            }
            for o in sorted(objects, key=lambda o: o["track_id"])
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))


class SavedDecisions:
    """Answers loaded from a --save-decisions file, looked up by crop."""

    def __init__(self, path: Path, config: DecisionsConfig):
        try:
            data = json.loads(Path(path).read_text())
        except FileNotFoundError:
            raise ValueError(f"decisions file not found: {path}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid JSON ({exc})") from exc
        if data.get("format") != DECISIONS_FORMAT:
            raise ValueError(f"{path}: not a decisions file")

        self.compatible = data.get("config_hash") == _fingerprint_hash(config)
        self.min_confidence = config.min_confidence
        # {frame: [(box, record), ...]}; failed calls are retried.
        self._by_frame: dict[int, list[tuple[np.ndarray, dict]]] = {}
        for record in data.get("objects", []):
            if record.get("status") == "failed":
                continue
            self._by_frame.setdefault(int(record["frame"]), []).append(
                (np.array(record["box"], dtype=float), record)
            )
        self.size = sum(len(v) for v in self._by_frame.values())

    def lookup(self, frame: int, box: np.ndarray) -> Verdict | None:
        best, best_iou = None, SAME_BOX_IOU
        for saved_box, record in self._by_frame.get(frame, ()):
            iou = _iou(saved_box, box)
            if iou >= best_iou:
                best, best_iou = record, iou
        if best is None:
            return None
        if best["status"] == "refused":
            return Verdict(None, status="refused")
        return Verdict.answered(
            best["choice"],
            best["confidence"],
            best["probabilities"],
            self.min_confidence,
        )


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (
        (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    )
    return inter / union if union > 0 else 0.0

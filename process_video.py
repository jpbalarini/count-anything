"""Count objects in a video (RF-DETR or a cloud model) + ByteTrack.

Counting modes
--------------
  --count-mode crossing   count objects crossing a line (default)
  --count-mode total      count every distinct object seen in the video

Detectors
---------
  --detector local        RF-DETR on every frame (default)
  --detector cloud        a cloud vision model (Anthropic) on sampled
                          frames only, see --sample-rate. Needs the
                          ANTHROPIC_API_KEY environment variable.

Examples
--------
Minimal (cars, motorcycles, buses and trucks, horizontal line mid-frame):

    python process_video.py input.mp4

Only cars and trucks, custom colors, custom line, one-way counting:

    python process_video.py input.mp4 -o out.mp4 \\
        --classes car truck \\
        --colors car=#3B82F6 truck=#22C55E \\
        --line 0 700 1892 700 \\
        --direction in --in-text "EASTBOUND"

Custom HUD (LABEL=CLASS[,CLASS...] or LABEL=all):

    python process_video.py input.mp4 --classes person bicycle \\
        --hud-title "FOOTFALL" \\
        --hud-rows "PEOPLE=person" "CYCLISTS=bicycle" "EVERYONE=all"

Total count (no line), using a cloud model at 2 samples per second:

    python process_video.py input.mp4 --count-mode total \\
        --detector cloud --sample-rate 2 --classes car truck

With --detector cloud, --classes takes any free-form text, not just COCO
classes (quote multi-word names or separate them with commas):

    python process_video.py input.mp4 --count-mode total \\
        --detector cloud --classes "traffic cone" "shipping container"

Run `python process_video.py --list-classes` to see the available class ids
(local detector only).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from dotenv import load_dotenv

from cloud_detector import (
    DEFAULT_CLOUD_CONCURRENCY,
    DEFAULT_CLOUD_MAX_SIDE,
    DEFAULT_CLOUD_MODEL,
    DEFAULT_SAMPLE_RATE,
    CloudDetector,
)

# Load ANTHROPIC_API_KEY etc. from .env next to this script (and from
# the current directory). Variables already set in the shell win.
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv()


# ---------------------------------------------------------
# Defaults
#
# RF-DETR uses the sparse COCO ids (NOT the same as
# Ultralytics YOLO):  3 = car, 4 = motorcycle, 6 = bus,
# 8 = truck.
# ---------------------------------------------------------

DEFAULT_CLASSES = ["car", "motorcycle", "bus", "truck"]

# Colors used for well-known classes when the user doesn't override them.
DEFAULT_CLASS_COLORS = {
    3: "#3B82F6",  # car        - blue
    4: "#06B6D4",  # motorcycle - cyan
    6: "#8B5CF6",  # bus        - purple
    8: "#22C55E",  # truck      - green
}

# Cycled through for any other class that has no explicit color.
FALLBACK_COLORS = [
    "#EF4444",  # red
    "#EAB308",  # yellow
    "#EC4899",  # pink
    "#14B8A6",  # teal
    "#F97316",  # orange
    "#A3E635",  # lime
]

OTHER_COLOR = "#64748B"
DEFAULT_LINE_COLOR = "#F59E0B"  # amber

MODEL_SIZES = ["nano", "small", "medium", "large"]


# ---------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------


def _normalize(name: str) -> str:
    return name.strip().lower().replace("_", " ").replace("-", " ")


def load_coco_classes() -> dict[int, str]:
    from rfdetr.assets.coco_classes import COCO_CLASSES

    return dict(COCO_CLASSES)


def build_free_classes(tokens: list[str]) -> dict[int, str]:
    """Free-form class names for the cloud detector.

    Any text is a valid class. Ids are just the position in the
    list (0, 1, 2, ...), duplicates (ignoring case/`_`/`-`) removed.
    """
    names: dict[str, str] = {}
    for token in tokens:
        token = token.strip()
        if token:
            names.setdefault(_normalize(token), token)
    return dict(enumerate(names.values()))


def resolve_class(token: str, coco: dict[int, str]) -> int:
    """Turn '3' or 'car' (or 'traffic_light') into a class id."""
    token = token.strip()
    wanted = _normalize(token)
    for class_id, name in coco.items():
        if _normalize(name) == wanted:
            return class_id

    if token.lstrip("-").isdigit():
        class_id = int(token)
        if class_id in coco:
            return class_id
        raise ValueError(
            f"unknown class id {class_id} "
            "(see --list-classes)"
        )
    raise ValueError(
        f"unknown class name '{token}' (see --list-classes)"
    )


def split_tokens(values: list[str]) -> list[str]:
    """Allow both `--classes car truck` and `--classes car,truck`."""
    out: list[str] = []
    for value in values:
        out.extend(t for t in value.split(",") if t.strip())
    return out


def parse_color(value: str) -> sv.Color:
    value = value.strip()
    if not value.startswith("#"):
        value = f"#{value}"
    try:
        return sv.Color.from_hex(value)
    except Exception as exc:
        raise ValueError(
            f"invalid hex color '{value}' (expected e.g. #3B82F6)"
        ) from exc


def build_color_map(
    class_ids: list[int],
    color_args: list[str] | None,
    coco: dict[int, str],
    use_defaults: bool = True,
) -> dict[int, str]:
    """Return {class_id: '#RRGGBB'} for every tracked class.

    Priority: explicit --colors, then built-in defaults (COCO ids
    only, so skipped for free-form cloud classes), then a rotating
    fallback palette.

    --colors accepts `CLASS=HEX` entries (CLASS is an id or name)
    and/or bare HEX entries, which are matched to --classes by
    position.
    """
    explicit: dict[int, str] = {}

    for i, item in enumerate(split_tokens(color_args or [])):
        if "=" in item:
            key, hex_value = item.split("=", 1)
            class_id = resolve_class(key, coco)
            if class_id not in class_ids:
                raise ValueError(
                    f"--colors: class '{key}' is not in --classes"
                )
        else:
            if i >= len(class_ids):
                raise ValueError(
                    "--colors: more positional colors than --classes"
                )
            class_id, hex_value = class_ids[i], item

        parse_color(hex_value)  # validate
        explicit[class_id] = (
            hex_value if hex_value.startswith("#") else f"#{hex_value}"
        )

    colors: dict[int, str] = {}
    fallback_i = 0
    for class_id in class_ids:
        if class_id in explicit:
            colors[class_id] = explicit[class_id]
        elif use_defaults and class_id in DEFAULT_CLASS_COLORS:
            colors[class_id] = DEFAULT_CLASS_COLORS[class_id]
        else:
            palette = (
                FALLBACK_COLORS
                if use_defaults
                else list(DEFAULT_CLASS_COLORS.values())
                + FALLBACK_COLORS
            )
            colors[class_id] = palette[fallback_i % len(palette)]
            fallback_i += 1
    return colors


def build_hud_rows(
    row_args: list[str] | None,
    class_ids: list[int],
    coco: dict[int, str],
) -> list[tuple[str, list[int]]]:
    """Return [(label, [class ids counted by this row]), ...].

    Row syntax: `LABEL=all`, `LABEL=CLASS[,CLASS...]`, or just
    `CLASS` (label defaults to the class name).

    Default: a TOTAL row followed by one row per tracked class.
    """
    if not row_args:
        rows = [("TOTAL", list(class_ids))]
        rows += [
            (coco[c].upper(), [c]) for c in class_ids
        ]
        return rows

    rows = []
    for spec in row_args:
        if "=" in spec:
            label, classes_spec = spec.split("=", 1)
        else:
            label, classes_spec = None, spec

        if classes_spec.strip().lower() == "all":
            ids = list(class_ids)
        else:
            ids = [
                resolve_class(t, coco)
                for t in classes_spec.split(",")
                if t.strip()
            ]

        missing = [c for c in ids if c not in class_ids]
        if missing:
            names = ", ".join(coco[c] for c in missing)
            raise ValueError(
                f"--hud-rows '{spec}': {names} not in --classes, "
                "so it would always read 0"
            )
        if not ids:
            raise ValueError(f"--hud-rows '{spec}': no classes given")

        if label is None:
            label = coco[ids[0]].upper()
        rows.append((label.strip(), ids))
    return rows


def sampling_due(
    index: int, next_at: float, step: float
) -> tuple[bool, float]:
    """Is `index` a sampled frame? Returns (due, updated next_at).

    Shared by the cloud pre-pass and the render pass so both pick
    exactly the same frames.
    """
    if index + 1e-9 < next_at:
        return False, next_at
    while next_at <= index + 1e-9:
        next_at += step
    return True, next_at


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Count objects crossing a line in a video "
            "(RF-DETR detection + ByteTrack tracking)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    io = p.add_argument_group("input / output")
    io.add_argument(
        "source",
        nargs="?",
        help="Path to the input video.",
    )
    io.add_argument(
        "-o",
        "--output",
        help=(
            "Path for the annotated video "
            "(default: ./<source name>_counted.mp4)."
        ),
    )
    io.add_argument(
        "--list-classes",
        action="store_true",
        help="Print the available class ids/names and exit.",
    )

    det = p.add_argument_group("detection")
    det.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        metavar="CLASS",
        help=(
            "Class ids or names to detect, e.g. `3 8` or "
            "`car truck` or `car,truck`. With --detector cloud "
            "these are free-form text, not limited to COCO, e.g. "
            "`\"traffic cone\" \"person wearing a helmet\"` "
            f"(default: {' '.join(DEFAULT_CLASSES)})."
        ),
    )
    det.add_argument(
        "--colors",
        nargs="+",
        metavar="CLASS=HEX",
        help=(
            "Color per class, e.g. `car=#3B82F6 8=#22C55E`. "
            "Bare hex values are matched to --classes by position. "
            "Unspecified classes get a default color."
        ),
    )
    det.add_argument(
        "--model",
        choices=MODEL_SIZES,
        default="medium",
        help="RF-DETR model size (default: medium).",
    )
    det.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Detection confidence threshold (default: 0.2).",
    )
    det.add_argument(
        "--track-threshold",
        type=float,
        default=0.25,
        help="ByteTrack activation threshold (default: 0.25).",
    )
    det.add_argument(
        "--min-track-frames",
        type=int,
        default=None,
        help=(
            "Detections (frames, or samples with --detector cloud) "
            "an object must be seen in before it gets a track id "
            "(default: 3 for local, 1 for cloud)."
        ),
    )

    mode = p.add_argument_group("counting mode")
    mode.add_argument(
        "--count-mode",
        choices=["crossing", "total"],
        default="crossing",
        help=(
            "`crossing`: count objects crossing the line. `total`: "
            "count every distinct tracked object seen in the video "
            "(no line is drawn; --line/--direction are ignored). "
            "Default: crossing."
        ),
    )

    cloud = p.add_argument_group("detector (local vs cloud)")
    cloud.add_argument(
        "--detector",
        choices=["local", "cloud"],
        default="local",
        help=(
            "`local`: RF-DETR on every frame. `cloud`: a cloud vision "
            "model on sampled frames only, replacing RF-DETR "
            "(default: local)."
        ),
    )
    cloud.add_argument(
        "--cloud-model",
        default=os.environ.get("CLOUD_MODEL", DEFAULT_CLOUD_MODEL),
        help=(
            "Anthropic model id used with --detector cloud "
            f"(default: $CLOUD_MODEL or {DEFAULT_CLOUD_MODEL})."
        ),
    )
    cloud.add_argument(
        "--sample-rate",
        type=float,
        default=DEFAULT_SAMPLE_RATE,
        metavar="PER_SEC",
        help=(
            "Cloud model calls per second of video, e.g. 2 on a "
            "10 s video makes 20 calls regardless of the video FPS. "
            "Between samples the last result is held on screen "
            f"(default: {DEFAULT_SAMPLE_RATE:g})."
        ),
    )
    cloud.add_argument(
        "--cloud-concurrency",
        type=int,
        default=DEFAULT_CLOUD_CONCURRENCY,
        metavar="N",
        help=(
            "Max simultaneous cloud API calls; all samples are sent "
            "up front, N at a time, and the video is rendered "
            "afterwards. 1 = sequential "
            f"(default: {DEFAULT_CLOUD_CONCURRENCY})."
        ),
    )
    cloud.add_argument(
        "--cloud-max-side",
        type=int,
        default=DEFAULT_CLOUD_MAX_SIDE,
        metavar="PX",
        help=(
            "Frames are downscaled so their longest side is at most "
            f"PX before upload (default: {DEFAULT_CLOUD_MAX_SIDE})."
        ),
    )

    line = p.add_argument_group("counting line")
    line.add_argument(
        "--line",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        help=(
            "Line endpoints in pixels "
            "(default: horizontal line across the middle of the frame). "
            "The line's direction defines what 'in' and 'out' mean."
        ),
    )
    line.add_argument(
        "--line-relative",
        action="store_true",
        help=(
            "Interpret --line values as fractions (0-1) of the frame "
            "width/height, e.g. `--line 0 0.5 1 0.5 --line-relative`."
        ),
    )
    line.add_argument(
        "--line-color",
        default=DEFAULT_LINE_COLOR,
        help=f"Line color as hex (default: {DEFAULT_LINE_COLOR}).",
    )
    line.add_argument(
        "--direction",
        choices=["both", "in", "out"],
        default="both",
        help=(
            "Which crossings to count and display: both in/out, "
            "only in, or only out (default: both)."
        ),
    )
    line.add_argument(
        "--in-text",
        help="Label shown for 'in' crossings (default: 'in').",
    )
    line.add_argument(
        "--out-text",
        help="Label shown for 'out' crossings (default: 'out').",
    )
    line.add_argument(
        "--swap-direction",
        action="store_true",
        help="Swap which side of the line counts as 'in' vs 'out'.",
    )

    hud = p.add_argument_group("HUD / annotations")
    hud.add_argument(
        "--no-hud",
        action="store_true",
        help="Do not draw the HUD panel.",
    )
    hud.add_argument(
        "--no-labels",
        action="store_true",
        help=(
            "Do not draw the text label (class name + track id) on "
            "each box. Boxes are still drawn."
        ),
    )
    hud.add_argument(
        "--hud-title",
        default="OBJECT COUNTER",
        help="HUD title (default: 'OBJECT COUNTER').",
    )
    hud.add_argument(
        "--hud-rows",
        nargs="+",
        metavar="LABEL=CLASSES",
        help=(
            "HUD rows, e.g. `VEHICLES=all CARS=car "
            "'TWO WHEELS=motorcycle,bicycle'`. Default: a TOTAL row "
            "plus one row per class."
        ),
    )
    hud.add_argument(
        "--hud-color",
        help="HUD accent color as hex (default: the line color).",
    )
    size = p.add_argument_group(
        "size filter (hides far-away / tiny objects)"
    )
    size.add_argument(
        "--min-size",
        "--min-display-size",
        nargs=2,
        type=int,
        default=[35, 25],
        metavar=("W", "H"),
        help=(
            "Minimum box size in pixels. Boxes smaller than W x H are "
            "filtered out (default: 35 25). `0 0` disables the "
            "minimum."
        ),
    )
    size.add_argument(
        "--max-size",
        nargs=2,
        type=int,
        default=[0, 0],
        metavar=("W", "H"),
        help=(
            "Maximum box size in pixels; larger boxes are filtered "
            "out. 0 means no limit (default: 0 0)."
        ),
    )
    size.add_argument(
        "--size-filter",
        choices=["display", "all"],
        default="display",
        help=(
            "`display`: filtered objects are only hidden from the "
            "video but still tracked and counted. `all`: they are also "
            "excluded from counting (default: display)."
        ),
    )
    size.add_argument(
        "--size-hysteresis",
        type=float,
        default=0.8,
        metavar="RATIO",
        help=(
            "Anti-flicker. An object must reach the full min size to "
            "appear, but stays visible until it drops below "
            "RATIO x min size (and symmetrically for --max-size). "
            "Use 1.0 for a hard cutoff (default: 0.8)."
        ),
    )
    hud.add_argument(
        "--trace-length",
        type=int,
        default=20,
        help="Trace length in frames, 0 disables traces (default: 20).",
    )

    args = p.parse_args()

    if not args.list_classes and not args.source:
        p.error("the following arguments are required: source")
    if not 0 < args.size_hysteresis <= 1:
        p.error("--size-hysteresis must be in (0, 1]")
    if min(args.min_size + args.max_size) < 0:
        p.error("--min-size / --max-size must be >= 0")
    if args.sample_rate <= 0:
        p.error("--sample-rate must be > 0")
    if args.cloud_concurrency < 1:
        p.error("--cloud-concurrency must be >= 1")
    if args.cloud_max_side < 64:
        p.error("--cloud-max-side must be >= 64")
    if args.min_track_frames is None:
        args.min_track_frames = 1 if args.detector == "cloud" else 3
    return args


# ---------------------------------------------------------
# HUD
# ---------------------------------------------------------


class Hud:
    """Translucent counter panel with a title and N count rows."""

    X, Y = 40, 40
    MIN_WIDTH = 300
    TITLE_SCALE = 0.62
    VALUE_SCALE = 0.78
    LABEL_SCALE = 0.52
    ROW_STEP = 40
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(
        self,
        title: str,
        rows: list[tuple[str, list[int]]],
        accent_bgr: tuple[int, int, int],
    ):
        self.title = title
        self.rows = rows
        self.accent = accent_bgr

        # Size the panel to fit the title and the longest label.
        title_w = cv2.getTextSize(
            title, self.FONT, self.TITLE_SCALE, 2
        )[0][0]
        label_w = max(
            (
                cv2.getTextSize(
                    label, self.FONT, self.LABEL_SCALE, 1
                )[0][0]
                for label, _ in rows
            ),
            default=0,
        )
        self.w = max(
            self.MIN_WIDTH,
            25 + title_w + 25,
            90 + label_w + 25,
        )
        self.h = 75 + self.ROW_STEP * len(rows)

    def draw(
        self,
        frame: np.ndarray,
        counts: dict[int, int],
    ) -> np.ndarray:
        x, y, w, h = self.X, self.Y, self.w, self.h

        # Dark translucent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay, (x, y), (x + w, y + h), (20, 25, 35), -1
        )
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        # Accent bar
        cv2.rectangle(
            frame, (x, y), (x + 5, y + h), self.accent, -1
        )

        # Title
        cv2.putText(
            frame,
            self.title,
            (x + 25, y + 38),
            self.FONT,
            self.TITLE_SCALE,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Separator
        cv2.line(
            frame,
            (x + 25, y + 55),
            (x + w - 25, y + 55),
            (100, 105, 115),
            1,
            cv2.LINE_AA,
        )

        # Rows
        row_y = y + 95
        for label, class_ids in self.rows:
            value = sum(counts.get(c, 0) for c in class_ids)

            cv2.putText(
                frame,
                f"{value:02d}",
                (x + 25, row_y),
                self.FONT,
                self.VALUE_SCALE,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                label,
                (x + 90, row_y),
                self.FONT,
                self.LABEL_SCALE,
                (190, 195, 205),
                1,
                cv2.LINE_AA,
            )
            row_y += self.ROW_STEP

        return frame


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------


def main() -> None:
    args = parse_args()
    use_cloud = args.detector == "cloud"

    if args.list_classes:
        if use_cloud:
            print(
                "With --detector cloud, --classes accepts any "
                "free-form text (e.g. 'traffic cone', "
                "'person wearing a helmet'); there is no fixed list."
            )
            return
        for class_id, name in sorted(load_coco_classes().items()):
            print(f"{class_id:3d}  {name}")
        return

    # -- Validate / resolve options --------------------------------
    # `coco` maps class id -> name for every class we can track.
    # Local: the fixed COCO set. Cloud: whatever the user typed.
    try:
        if use_cloud:
            coco = build_free_classes(split_tokens(args.classes))
            class_ids = list(coco)
        else:
            coco = load_coco_classes()
            class_ids = list(
                dict.fromkeys(  # dedupe, keep order
                    resolve_class(t, coco)
                    for t in split_tokens(args.classes)
                )
            )
        if not class_ids:
            raise ValueError("--classes: no classes given")
        color_map = build_color_map(
            class_ids, args.colors, coco, use_defaults=not use_cloud
        )
        hud_rows = build_hud_rows(args.hud_rows, class_ids, coco)
        line_color = parse_color(args.line_color)
        hud_color = parse_color(args.hud_color or args.line_color)
    except ValueError as exc:
        sys.exit(f"error: {exc}")

    source_path = Path(args.source)
    if not source_path.is_file():
        sys.exit(f"error: source video not found: {source_path}")

    target_path = (
        Path(args.output)
        if args.output
        else Path.cwd() / f"{source_path.stem}_counted.mp4"
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # -- Video ------------------------------------------------------
    video_info = sv.VideoInfo.from_video_path(str(source_path))
    print(video_info)

    # -- Counting line (crossing mode only) -------------------------
    count_total = args.count_mode == "total"
    line_zone = None
    line_zone_annotator = None

    if count_total:
        print("Count mode: total (distinct tracked objects)")
    else:
        if args.line:
            x1, y1, x2, y2 = args.line
            if args.line_relative:
                x1, x2 = x1 * video_info.width, x2 * video_info.width
                y1, y2 = y1 * video_info.height, y2 * video_info.height
        else:
            x1, x2 = 0, video_info.width
            y1 = y2 = video_info.height // 2

        if (x1, y1) == (x2, y2):
            sys.exit(
                "error: --line start and end must be different points"
            )

        if args.swap_direction:
            x1, y1, x2, y2 = x2, y2, x1, y1

        print(f"Counting line: ({x1:g}, {y1:g}) -> ({x2:g}, {y2:g})")

        line_zone = sv.LineZone(
            start=sv.Point(x1, y1),
            end=sv.Point(x2, y2),
        )

        line_zone_annotator = sv.LineZoneAnnotator(
            color=line_color,
            thickness=2,
            text_thickness=2,
            text_scale=0.8,
            custom_in_text=args.in_text,
            custom_out_text=args.out_text,
            display_in_count=args.direction in ("both", "in"),
            display_out_count=args.direction in ("both", "out"),
        )

    # -- Detector ---------------------------------------------------
    if use_cloud:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit(
                "error: --detector cloud needs the ANTHROPIC_API_KEY "
                "environment variable"
            )
        cloud_detector = CloudDetector(
            model=args.cloud_model,
            class_names={c: coco[c] for c in class_ids},
            threshold=args.threshold,
            max_side=args.cloud_max_side,
        )

        # Number of frames between two cloud calls (>= 1 frame).
        sample_step = max(1.0, video_info.fps / args.sample_rate)
        effective_rate = video_info.fps / sample_step
        expected_calls = (
            int(np.ceil((video_info.total_frames or 0) / sample_step))
        )
        print(
            f"Cloud detector: {args.cloud_model}, "
            f"{effective_rate:g} samples/s (one call every "
            f"{sample_step:g} frames, ~{expected_calls} calls, "
            f"{args.cloud_concurrency} in parallel)"
        )

        # Detection doesn't depend on tracking, so every sampled
        # frame is sent to the model up front (in parallel). Tracking,
        # counting and drawing then replay the results in frame order
        # while the output video is written.
        def sampled_frames():
            next_at = 0.0
            for i, frame in enumerate(
                sv.get_video_frames_generator(str(source_path))
            ):
                due, next_at = sampling_due(i, next_at, sample_step)
                if due:
                    yield i, frame

        def report_progress(done: int, _submitted: int) -> None:
            print(f"  cloud calls done: {done}/~{expected_calls}")

        print("Running cloud detection...")
        cloud_results = cloud_detector.detect_frames(
            sampled_frames(),
            args.cloud_concurrency,
            on_done=report_progress,
        )
        if cloud_detector.calls and (
            cloud_detector.failures == cloud_detector.calls
        ):
            sys.exit(
                "error: every cloud call failed (see the warnings "
                "above), nothing to render"
            )
    else:
        import rfdetr

        model_cls_name = f"RFDETR{args.model.capitalize()}"
        print(f"Loading RF-DETR {args.model.capitalize()}...")
        model = getattr(rfdetr, model_cls_name)()
        print(f"RF-DETR {args.model.capitalize()} loaded")

        def detect(frame: np.ndarray) -> sv.Detections:
            # sv.process_video gives BGR, RF-DETR expects RGB.
            return model.predict(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                threshold=args.threshold,
            )

        cloud_results = {}
        sample_step = 1.0
        effective_rate = video_info.fps

    # -- Tracker ----------------------------------------------------
    # The tracker only ever sees sampled frames, so it has to be told
    # the sampled rate, not the video FPS.
    from trackers import ByteTrackTracker

    byte_tracker = ByteTrackTracker(
        frame_rate=effective_rate,
        minimum_consecutive_frames=args.min_track_frames,
        track_activation_threshold=args.track_threshold,
    )

    # -- Annotators -------------------------------------------------
    # Palette is indexed by class id, so it needs to cover
    # 0..max(class id). Anything we don't track stays neutral gray.
    palette = sv.ColorPalette.from_hex(
        [
            color_map.get(i, OTHER_COLOR)
            for i in range(max(class_ids) + 1)
        ]
    )

    bounding_box_annotator = sv.BoxAnnotator(
        color=palette,
        thickness=2,
        color_lookup=sv.ColorLookup.CLASS,
    )
    label_annotator = sv.LabelAnnotator(
        color=palette,
        text_color=sv.Color.WHITE,
        text_thickness=1,
        text_scale=0.6,
        text_padding=5,
        border_radius=4,
        color_lookup=sv.ColorLookup.CLASS,
    )
    trace_annotator = (
        sv.TraceAnnotator(
            color=palette,
            thickness=2,
            trace_length=args.trace_length,
            color_lookup=sv.ColorLookup.CLASS,
        )
        # Traces need a position for every frame; with sampled cloud
        # detections they would just be a dot, so they are skipped.
        if args.trace_length > 0 and not use_cloud
        else None
    )

    hud = (
        None
        if args.no_hud
        else Hud(
            args.hud_title,
            hud_rows,
            (hud_color.b, hud_color.g, hud_color.r),
        )
    )

    min_w, min_h = args.min_size
    max_w, max_h = args.max_size
    hyst = args.size_hysteresis

    # Track ids currently considered "big enough". Once an id is in
    # here it only leaves when it falls below the relaxed
    # (hysteresis) limits, which stops boxes flickering on and off
    # around the threshold.
    visible_ids: set[int] = set()

    def size_mask(detections: sv.Detections) -> np.ndarray:
        widths = detections.xyxy[:, 2] - detections.xyxy[:, 0]
        heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]

        was_visible = np.array(
            [int(t) in visible_ids for t in detections.tracker_id],
            dtype=bool,
        )
        # Already-visible objects get relaxed limits.
        scale = np.where(was_visible, hyst, 1.0)

        mask = (widths >= min_w * scale) & (heights >= min_h * scale)
        if max_w:
            mask &= widths <= max_w / scale
        if max_h:
            mask &= heights <= max_h / scale

        for tracker_id, ok in zip(detections.tracker_id, mask):
            (visible_ids.add if ok else visible_ids.discard)(
                int(tracker_id)
            )
        return mask

    # {class_id: count} shown in the HUD: line crossings in
    # `crossing` mode, distinct tracked objects in `total` mode.
    counts: dict[int, int] = defaultdict(int)
    seen_track_ids: set[int] = set()  # total mode only

    def process(detections: sv.Detections) -> sv.Detections:
        """Track, filter and count one set of detections.

        Returns the detections to draw. Called once per frame for the
        local detector, once per sample for the cloud detector.
        """
        # Keep only the classes we care about
        detections = detections[
            np.isin(detections.class_id, class_ids)
        ]

        detections = byte_tracker.update(detections)

        # Remove detections without confirmed track IDs
        if detections.tracker_id is not None:
            detections = detections[detections.tracker_id >= 0]

        # Size filter (see --min-size / --max-size). By default this
        # only affects what is drawn; with --size-filter all the
        # filtered objects are not counted either.
        if len(detections) > 0:
            keep = size_mask(detections)
            display_detections = detections[keep]
            if args.size_filter == "all":
                detections = display_detections
        else:
            display_detections = detections

        # Counting
        if count_total:
            # Each track id is counted once, the first time it is
            # seen (as a member of the class it had then).
            if len(detections) > 0:
                for class_id, tracker_id in zip(
                    detections.class_id, detections.tracker_id
                ):
                    if int(tracker_id) not in seen_track_ids:
                        seen_track_ids.add(int(tracker_id))
                        counts[int(class_id)] += 1
        else:
            crossed_in, crossed_out = line_zone.trigger(detections)

            if args.direction == "in":
                crossed = crossed_in
            elif args.direction == "out":
                crossed = crossed_out
            else:
                crossed = crossed_in | crossed_out

            if np.any(crossed):
                for class_id in detections.class_id[crossed]:
                    counts[int(class_id)] += 1

        return display_detections

    # Detections currently on screen. With the cloud detector these
    # are held (unchanged) until the next sample arrives.
    display_detections = sv.Detections.empty()
    next_sample_at = 0.0
    samples_done = 0

    # -- Frame callback ---------------------------------------------
    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        nonlocal display_detections, next_sample_at, samples_done

        # Is this a frame we run detection on? Always for the local
        # detector; every `sample_step` frames for the cloud one.
        due, next_sample_at = sampling_due(
            index, next_sample_at, sample_step
        )
        if due:
            detections = (
                cloud_results.get(index) if use_cloud else detect(frame)
            )
            if detections is not None:  # None = cloud call failed
                display_detections = process(detections)
            samples_done += 1

        labels = [
            f"{coco[int(class_id)].upper()} · {int(tracker_id):02d}"
            for class_id, tracker_id in zip(
                display_detections.class_id,
                display_detections.tracker_id,
            )
        ] if display_detections.tracker_id is not None else []

        # Annotation
        annotated = frame.copy()

        if trace_annotator is not None:
            annotated = trace_annotator.annotate(
                scene=annotated, detections=display_detections
            )
        annotated = bounding_box_annotator.annotate(
            scene=annotated, detections=display_detections
        )
        if not args.no_labels:
            annotated = label_annotator.annotate(
                scene=annotated,
                detections=display_detections,
                labels=labels,
            )
        if line_zone_annotator is not None:
            annotated = line_zone_annotator.annotate(
                annotated, line_counter=line_zone
            )

        if hud is not None:
            annotated = hud.draw(annotated, counts)

        return annotated

    # -- Process ----------------------------------------------------
    sv.process_video(
        source_path=str(source_path),
        target_path=str(target_path),
        callback=callback,
    )

    if use_cloud:
        print(
            f"Cloud calls: {cloud_detector.calls} "
            f"({cloud_detector.failures} failed), tokens: "
            f"{cloud_detector.input_tokens} in / "
            f"{cloud_detector.output_tokens} out"
        )

    print(f"Saved to: {target_path}")


if __name__ == "__main__":
    main()

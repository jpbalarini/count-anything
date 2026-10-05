"""Command-line interface for process_video.py.

Argument definitions, their defaults and the validation of the parsed
values. Detector-specific defaults come from the detector modules.
"""

from __future__ import annotations

import argparse
import os

from cloud_detector import (
    DEFAULT_CLOUD_CONCURRENCY,
    DEFAULT_CLOUD_MAX_SIDE,
    DEFAULT_CLOUD_MODEL,
    DEFAULT_SAMPLE_RATE,
)
from locate_anything_detector import (
    DEFAULT_LOCATE_MODE,
    DEFAULT_LOCATE_SAMPLE_RATE,
)
from rfdetr_detector import DEFAULT_MODEL_SIZE, MODEL_SIZES

DEFAULT_CLASSES = ["car", "motorcycle", "bus", "truck"]
DEFAULT_LINE_COLOR = "#F59E0B"  # amber

DETECTORS = ["rfdetr", "cloud", "locate-anything"]
DEFAULT_DETECTOR = "rfdetr"
# Detectors that take free-form class names (anything else uses COCO).
OPEN_VOCAB_DETECTORS = {"cloud", "locate-anything"}
# Detections (frames, or samples when detection is sampled) in a row an
# object may be missed in and still be drawn (--hold-frames), for the
# detectors that sometimes miss objects on a frame. The others (rfdetr)
# don't hold unless asked.
DEFAULT_HOLD_FRAMES = 2

# Detector calls per second of video. Detectors not listed here run on
# every frame unless --sample-rate is given.
DEFAULT_SAMPLE_RATES = {
    "cloud": DEFAULT_SAMPLE_RATE,
    "locate-anything": DEFAULT_LOCATE_SAMPLE_RATE,
}

# Shown at the end of `--help`.
EPILOG = """\
Counting modes
--------------
  --count-mode crossing   count objects crossing a line (default)
  --count-mode total      count every distinct object seen in the video

Detectors
---------
  --detector rfdetr       RF-DETR on every frame (default); with
                          --sample-rate only on sampled frames
  --detector cloud        a cloud vision model (Anthropic) on sampled
                          frames only, see --sample-rate. Needs the
                          ANTHROPIC_API_KEY environment variable.
  --detector locate-anything
                          locate-anything.cpp on sampled frames only.
                          Needs --locate-model path/to/model.gguf, and
                          --locate-lib path/to/liblocate_anything.dylib
                          to load the model once (else
                          `locate-anything-cli` in PATH is used, which
                          reloads it on every frame).

Speed / iteration
-----------------
  --infer-resolution 720  run detection on frames downscaled so the
                          shorter side is 720 px (boxes are mapped back
                          to the full-resolution video)
  --save-detections f.json   write the raw detections to a JSON file
  --load-detections f.json   skip detection, reuse a saved JSON (change
                          colors, HUD, line, classes... and re-render)

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

With --detector cloud or locate-anything, --classes takes any free-form text, not just COCO
classes (quote multi-word names or separate them with commas):

    python process_video.py input.mp4 --count-mode total \\
        --detector cloud --classes "traffic cone" "shipping container"

Detect once at 720p, then tweak the look without detecting again:

    python process_video.py input.mp4 --infer-resolution 720 \\
        --save-detections dets.json
    python process_video.py input.mp4 --load-detections dets.json \\
        --hud-title "NEW TITLE" --colors car=#FF0000

Run `python process_video.py --list-classes` to see the available class ids
(rfdetr detector only).
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Count objects crossing a line in a video "
            "(RF-DETR detection + ByteTrack tracking)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
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
            "`car truck` or `car,truck`. With --detector cloud or "
            "locate-anything these are free-form text, not limited to COCO, e.g. "
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
        default=DEFAULT_MODEL_SIZE,
        help=f"RF-DETR model size (default: {DEFAULT_MODEL_SIZE}).",
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
            "Detections (frames, or samples when detection is sampled) "
            "an object must be seen in before it gets a track id "
            "(default: 3 when detecting on every frame, 1 when sampled)."
        ),
    )

    det.add_argument(
        "--hold-frames",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Keep drawing a tracked object when the detector misses it "
            "in up to N detections (frames, or samples when detection "
            "is sampled) in a row, so frames on which it returns only "
            "some of the objects don't make boxes blink. With "
            "--interpolate the box moves from where it was last found "
            "to where it is found again; without, it stays where it "
            "was last found. Only affects what is drawn, not the "
            f"counts. 0 = off (default: {DEFAULT_HOLD_FRAMES} for cloud "
            "/ locate-anything, 0 for rfdetr)."
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

    speed = p.add_argument_group("inference speed / detections file")
    speed.add_argument(
        "--infer-resolution",
        type=int,
        default=None,
        metavar="PX",
        help=(
            "Run detection on frames downscaled so the shorter side is "
            "PX pixels, e.g. `720` for a 1080p video (faster). Never "
            "upscales. Boxes are mapped back to the full-resolution "
            "video, which is still rendered at its original size. "
            "Default: full resolution."
        ),
    )
    speed.add_argument(
        "--save-detections",
        metavar="JSON",
        help=(
            "Write the raw detections (before tracking/counting) to "
            "this JSON file, so the video can be re-rendered later "
            "with --load-detections."
        ),
    )
    speed.add_argument(
        "--description",
        default="",
        help=(
            "info.description of the --save-detections file (default: "
            "'<detector> detections of <classes> in <video>')."
        ),
    )
    speed.add_argument(
        "--contributor",
        default="",
        help="info.contributor of the --save-detections file.",
    )
    speed.add_argument(
        "--url",
        default="",
        help="info.url of the --save-detections file.",
    )
    speed.add_argument(
        "--load-detections",
        metavar="JSON",
        help=(
            "Do not run any detector: read detections from a file "
            "written by --save-detections and only track, count and "
            "render. Colors, HUD, line, size filter, --classes, "
            "--threshold (raise only), tracker settings, etc. can all "
            "be changed freely. Use the same source video."
        ),
    )

    cloud = p.add_argument_group("detector (rfdetr / cloud / locate-anything)")
    cloud.add_argument(
        "--detector",
        choices=DETECTORS,
        default=DEFAULT_DETECTOR,
        help=(
            "`rfdetr`: RF-DETR, on every frame unless --sample-rate is "
            "given. `cloud`: a cloud vision model on sampled frames. "
            "`locate-anything`: locate-anything.cpp on "
            f"sampled frames (default: {DEFAULT_DETECTOR}). Ignored with "
            "--load-detections."
        ),
    )
    cloud.add_argument(
        "--locate-model",
        default=os.environ.get("LOCATE_ANYTHING_MODEL"),
        metavar="GGUF",
        help=(
            "Path to the locate-anything .gguf model, required with "
            "--detector locate-anything (default: $LOCATE_ANYTHING_MODEL)."
        ),
    )
    cloud.add_argument(
        "--locate-lib",
        default=os.environ.get("LOCATE_ANYTHING_LIB"),
        metavar="LIB",
        help=(
            "Path to liblocate_anything (.dylib / .so, built with "
            "-DLA_SHARED=ON). The model is then loaded once instead of "
            "on every frame (a small gain when the model file is in the "
            "OS cache, see the README). Without it `locate-anything-cli` "
            "is used (default: $LOCATE_ANYTHING_LIB)."
        ),
    )
    cloud.add_argument(
        "--locate-mode",
        choices=["hybrid", "slow", "fast"],
        default=DEFAULT_LOCATE_MODE,
        help=(
            "locate-anything decode mode "
            f"(default: {DEFAULT_LOCATE_MODE})."
        ),
    )
    cloud.add_argument(
        "--locate-threads",
        type=int,
        default=0,
        metavar="N",
        help="locate-anything CPU threads, 0 = auto (default: 0).",
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
        default=None,
        metavar="PER_SEC",
        help=(
            "Detector calls per second of video, e.g. 2 on a 10 s video "
            "makes 20 calls regardless of the video FPS. Between samples "
            f"the last result is held on screen (default: "
            f"{DEFAULT_SAMPLE_RATE:g} for cloud, "
            f"{DEFAULT_LOCATE_SAMPLE_RATE:g} for locate-anything, every "
            "frame for rfdetr)."
        ),
    )
    cloud.add_argument(
        "--interpolate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Move each tracked object linearly between its detections: "
            "smoothly between samples instead of holding each box until "
            "the next sample, and across detections that missed it (see "
            "--hold-frames). --no-interpolate holds the box instead. "
            "Needs the detections after the current one, so the HUD "
            "counts follow the previous sample. With sampled detection "
            "it works best at about 5 samples/s or more. Without "
            "--hold-frames it changes nothing when detecting on every "
            "frame (default: on)."
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
        help=(
            "Label shown for 'in' crossings (default: 'in'). Pass "
            "'none' (or '') to hide the 'in' label on the video; "
            "crossings are still counted."
        ),
    )
    line.add_argument(
        "--out-text",
        help=(
            "Label shown for 'out' crossings (default: 'out'). Pass "
            "'none' (or '') to hide the 'out' label on the video; "
            "crossings are still counted."
        ),
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
    if args.sample_rate is not None and args.sample_rate <= 0:
        p.error("--sample-rate must be > 0")
    if args.hold_frames is not None and args.hold_frames < 0:
        p.error("--hold-frames must be >= 0")
    if args.cloud_concurrency < 1:
        p.error("--cloud-concurrency must be >= 1")
    if args.cloud_max_side < 64:
        p.error("--cloud-max-side must be >= 64")
    if args.infer_resolution is not None and args.infer_resolution < 32:
        p.error("--infer-resolution must be >= 32")
    if args.save_detections and args.load_detections:
        p.error(
            "--save-detections and --load-detections can't be combined"
        )
    return args

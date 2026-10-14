"""Count objects in a video (RF-DETR, a cloud model or locate-anything) + ByteTrack.

Run `python process_video.py --help` for the options and examples
(the argument definitions live in cli.py).
"""

from __future__ import annotations

import bisect
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from dotenv import load_dotenv
from tqdm import tqdm

from cli import (
    DEFAULT_DETECTOR,
    DEFAULT_THRESHOLD,
    DEFAULT_HOLD_FRAMES,
    DEFAULT_SAMPLE_RATES,
    DETECTORS,
    OPEN_VOCAB_DETECTORS,
    parse_args,
)
from cloud_detector import CloudDetector
from decisions import (
    DecisionsClassifier,
    DecisionsConfig,
    SavedDecisions,
    Verdict,
    crop_box,
    save_decisions,
)
from detections_io import (
    DetectionsFile,
    load_detections,
    new_info,
    save_detections,
    to_detections,
)
from frames import (
    CountedLine,
    TrackTimeline,
    inference_size,
    sampling_due,
)
from hud import Hud
from locate_anything_detector import LocateAnythingDetector
from options import (
    OTHER_COLOR,
    build_color_map,
    build_free_classes,
    build_hud_rows,
    is_hidden_label,
    load_coco_classes,
    normalize_name,
    parse_color,
    parse_size_grades,
    resolve_class,
    split_tokens,
)
from rfdetr_detector import (
    RfDetrDetector,
    recommended_threshold,
    weights_path,
)
from showcase import (
    GOOD_HEX,
    ChoiceStyle,
    PanelStats,
    ShowcaseRenderer,
    TrackInfo,
    hex_to_rgb,
)

# Load ANTHROPIC_API_KEY / OPENAI_API_KEY etc. from .env next to this script (and from
# the current directory). Variables already set in the shell win.
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv()


# ---------------------------------------------------------
# Detection
# ---------------------------------------------------------


@dataclass
class DetectionRun:
    """The result of the detection pre-pass.

    `detections_by_frame` maps each sampled frame index to its detections
    in video pixels (None when the detector call failed). `meta` goes into
    the --save-detections info, `stats` is printed at the end.
    """

    detections_by_frame: dict[int, sv.Detections | None]
    meta: dict = field(default_factory=dict)
    stats: str = ""


def detections_from_file(
    loaded: DetectionsFile,
    path: str,
    detector: str,
    coco: dict[int, str],
    threshold: float,
    video_info: sv.VideoInfo,
) -> DetectionRun:
    """The detections of a --load-detections file, matched to the classes
    by name."""
    name_to_id = {normalize_name(n): c for c, n in coco.items()}
    detections_by_frame = {
        index: (
            None
            if items is None
            else to_detections(items, name_to_id, normalize_name, threshold)
        )
        for index, items in loaded.frames.items()
    }
    mismatch = [
        f"{key} {loaded.video[key]} (video: {actual})"
        for key, actual in (
            ("width", video_info.width),
            ("height", video_info.height),
            ("total_frames", video_info.total_frames),
        )
        if loaded.video.get(key) not in (None, actual)
    ]
    if mismatch:
        print(
            "warning: detections file doesn't match this video: "
            + ", ".join(mismatch)
        )
    print(
        f"Loaded detections for {len(detections_by_frame)} frames "
        f"from {path} (detector: {detector}), skipping detection"
    )
    return DetectionRun(detections_by_frame)


def run_detector(
    detector: str,
    args,
    *,
    class_names: dict[int, str],
    rfdetr_detector: RfDetrDetector | None,
    source_path: Path,
    video_info: sv.VideoInfo,
    sample_step: float,
    effective_rate: float,
    expected_calls: int,
) -> DetectionRun:
    """Run `detector` on every sampled frame of the video.

    Detection runs on a copy of the frame downscaled to
    --infer-resolution; the boxes are scaled back so everything
    downstream works in video pixels.
    """
    infer_w, infer_h = inference_size(
        video_info.width, video_info.height, args.infer_resolution
    )
    resized = (infer_w, infer_h) != (video_info.width, video_info.height)
    if resized:
        print(
            f"Inference resolution: {infer_w}x{infer_h} "
            f"(video is {video_info.width}x{video_info.height})"
        )
    scale = np.array(
        [video_info.width / infer_w, video_info.height / infer_h] * 2
    )

    def sampled_frames():
        """(index, frame resized for inference) for every sampled frame."""
        next_at = 0.0
        for i, frame in enumerate(
            sv.get_video_frames_generator(str(source_path))
        ):
            due, next_at = sampling_due(i, next_at, sample_step)
            if due:
                if resized:
                    frame = cv2.resize(
                        frame, (infer_w, infer_h), interpolation=cv2.INTER_AREA
                    )
                yield i, frame

    sampling = (
        f"{effective_rate:g} samples/s (one every {sample_step:g} frames, "
        f"~{expected_calls} samples)"
    )
    unit = "frame" if detector == "rfdetr" else "call"

    def progress() -> tqdm:
        return tqdm(total=expected_calls, desc="Detecting", unit=unit)

    if detector == "cloud":
        raw, run = _detect_cloud(
            args, class_names, sampled_frames(), sampling, progress
        )
    elif detector == "locate-anything":
        raw, run = _detect_locate_anything(
            args, class_names, sampled_frames(), sampling, progress
        )
    else:
        raw, run = _detect_rfdetr(
            args, rfdetr_detector, sampled_frames(), sampling, progress
        )

    for index, detections in raw.items():
        if detections is not None and resized and len(detections) > 0:
            detections.xyxy = (detections.xyxy * scale).astype(np.float32)
        run.detections_by_frame[index] = detections
    return run


def _detect_cloud(args, class_names, frames, sampling, progress):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(
            "error: --detector cloud needs the ANTHROPIC_API_KEY "
            "environment variable"
        )
    cloud_detector = CloudDetector(
        model=args.cloud_model,
        class_names=class_names,
        threshold=args.threshold,
        max_side=args.cloud_max_side,
    )
    print(
        f"Cloud detector: {args.cloud_model}, {sampling}, "
        f"{args.cloud_concurrency} calls in parallel"
    )
    # Detection doesn't depend on tracking, so every sampled frame is
    # sent to the model up front (in parallel). Tracking, counting and
    # drawing then replay the results in frame order.
    print("Running cloud detection...")
    with progress() as bar:
        raw = cloud_detector.detect_frames(
            frames,
            args.cloud_concurrency,
            on_done=lambda done, _submitted: bar.update(done - bar.n),
        )
    if cloud_detector.calls and cloud_detector.failures == cloud_detector.calls:
        sys.exit(
            "error: every cloud call failed (see the warnings above), "
            "nothing to render"
        )
    return raw, DetectionRun(
        {},
        meta={"model": args.cloud_model},
        stats=(
            f"Cloud calls: {cloud_detector.calls} "
            f"({cloud_detector.failures} failed), tokens: "
            f"{cloud_detector.input_tokens} in / "
            f"{cloud_detector.output_tokens} out"
        ),
    )


def _detect_locate_anything(args, class_names, frames, sampling, progress):
    if not args.locate_model:
        sys.exit(
            "error: --detector locate-anything needs --locate-model "
            "(or the LOCATE_ANYTHING_MODEL environment variable)"
        )
    try:
        locate_detector = LocateAnythingDetector(
            model_path=args.locate_model,
            class_names=class_names,
            mode=args.locate_mode,
            threads=args.locate_threads,
            lib_path=args.locate_lib,
        )
    except RuntimeError as exc:
        sys.exit(f"error: {exc}")
    print(
        f"locate-anything detector: {args.locate_model} "
        f"({args.locate_mode}, {locate_detector.backend}), {sampling} x "
        f"{len(locate_detector.prompts)} class(es), one call at a time"
    )
    for prompt in locate_detector.prompts.values():
        print(f"Prompt: {prompt}")

    print("Running locate-anything detection...")
    with progress() as bar:
        raw = locate_detector.detect_frames(
            frames, on_done=lambda done: bar.update(done - bar.n)
        )
    if locate_detector.calls and (
        locate_detector.failures == locate_detector.calls
    ):
        sys.exit(
            "error: every locate-anything call failed (see the warnings "
            "above), nothing to render"
        )
    return raw, DetectionRun(
        {},
        meta={"model": args.locate_model, "locate_mode": args.locate_mode},
        stats=(
            f"locate-anything calls: {locate_detector.calls} "
            f"({locate_detector.failures} failed)"
        ),
    )


def _detect_rfdetr(args, rfdetr_detector, frames, sampling, progress):
    if rfdetr_detector is None:
        rfdetr_detector = RfDetrDetector(args.model, args.threshold)
    print(f"RF-DETR detector: {sampling}")
    print("Running RF-DETR detection...")
    with progress() as bar:
        raw = rfdetr_detector.detect_frames(
            frames, on_done=lambda done: bar.update(done - bar.n)
        )
    meta = {"model": rfdetr_detector.size}
    if args.weights:
        meta["weights"] = str(weights_path(args.weights))
    return raw, DetectionRun({}, meta=meta)


def write_detections(
    path: Path,
    args,
    run: DetectionRun,
    *,
    detector: str,
    sample_step: float,
    coco: dict[int, str],
    class_ids: list[int],
    coco_classes: bool,
    source_path: Path,
    video_info: sv.VideoInfo,
) -> None:
    """Save --save-detections (raw, before tracking)."""
    class_names = ", ".join(coco[c] for c in class_ids)
    info = new_info(
        {
            "detector": detector,
            "sample_step": sample_step,
            "infer_resolution": args.infer_resolution,
            "threshold": args.threshold,
            **run.meta,
        },
        description=args.description
        or f"{detector} detections of {class_names} in {source_path.name}",
        contributor=args.contributor,
        url=args.url,
    )
    video = {
        "file_name": source_path.name,
        "fps": video_info.fps,
        "width": video_info.width,
        "height": video_info.height,
        "total_frames": video_info.total_frames,
    }
    save_detections(
        path,
        info,
        video,
        run.detections_by_frame,
        label_of=lambda class_id: coco.get(class_id, str(class_id)),
        # RF-DETR class ids are COCO's category ids; free-form classes and
        # a fine-tuned model's are numbered from 0.
        category_id_of=(
            (lambda class_id: class_id)
            if coco_classes
            else (lambda class_id: class_id + 1)
        ),
        class_ids=class_ids,
    )
    print(f"Saved detections to: {path}")


# ---------------------------------------------------------
# Showcase style: measurement, classification, renderer
# ---------------------------------------------------------

# Boxes this close (px) to the border of the picture are cut off.
EDGE_MARGIN = 2


@dataclass
class Showcase:
    """Everything the showcase style needs to draw a frame."""

    renderer: ShowcaseRenderer
    panel_stats: PanelStats
    tracks: dict[int, TrackInfo]
    decisions_stats: str = ""

    def draw(
        self, frame: np.ndarray, index: int, detections: sv.Detections
    ) -> np.ndarray:
        objects = (
            list(
                zip(
                    detections.xyxy,
                    detections.class_id.tolist(),
                    detections.tracker_id.tolist(),
                )
            )
            if detections.tracker_id is not None
            else []
        )
        return self.renderer.draw(
            frame,
            index,
            objects,
            self.tracks,
            self.panel_stats.at(index),
            self.panel_stats.last_count_frame(index),
        )


def measure_tracks(
    track_ids: set[int],
    observations: dict[int, list[tuple[int, np.ndarray]]],
    width: int,
    height: int,
    mm_per_px: float | None,
) -> tuple[dict[int, TrackInfo], dict[int, tuple[int, np.ndarray]]]:
    """Size and best view of every track.

    The size is the box diagonal: the median over the frames the box is
    fully inside the picture (else its largest). The best view, used to
    classify the object, is its largest box among those fully inside the
    picture (else its largest). Returns ({id: TrackInfo}, {id: (frame,
    box)}).
    """

    def inside(xyxy: np.ndarray) -> bool:
        x1, y1, x2, y2 = xyxy
        return (
            x1 > EDGE_MARGIN and y1 > EDGE_MARGIN
            and x2 < width - EDGE_MARGIN and y2 < height - EDGE_MARGIN
        )

    def diagonal(xyxy: np.ndarray) -> float:
        return float(np.hypot(xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]))

    tracks: dict[int, TrackInfo] = {}
    best_view: dict[int, tuple[int, np.ndarray]] = {}
    for tracker_id in track_ids:
        obs = observations.get(tracker_id)
        if not obs:
            continue
        full = [o for o in obs if inside(o[1])]
        size_mm = None
        if mm_per_px:
            diagonals = [diagonal(b) for _, b in (full or obs)]
            size_px = float(np.median(diagonals)) if full else max(diagonals)
            size_mm = size_px * mm_per_px
        tracks[tracker_id] = TrackInfo(size_mm=size_mm)
        best_view[tracker_id] = max(
            full or obs,
            key=lambda o: (o[1][2] - o[1][0]) * (o[1][3] - o[1][1]),
        )
    return tracks, best_view


def classify_tracks(
    best_view: dict[int, tuple[int, np.ndarray]],
    config: DecisionsConfig,
    source_path: Path,
    save_path: Path,
    saved: SavedDecisions | None,
    saved_path: str | None,
) -> tuple[dict[int, Verdict], str]:
    """Classify each track's best view with the Decisions API.

    Answers in `saved` (--load-decisions) are reused for matching crops;
    the rest are cropped from the video and sent to the API. All answers
    are written to `save_path`. Returns ({id: Verdict}, API stats line).
    """
    verdicts: dict[int, Verdict] = {}
    reused: set[int] = set()
    if saved is not None:
        if not saved.compatible:
            print(
                f"warning: {saved_path} was made with a different model / "
                "question / crop settings, its answers are not reused"
            )
        else:
            for tracker_id, (frame_index, box) in best_view.items():
                verdict = saved.lookup(frame_index, box)
                if verdict is not None:
                    verdicts[tracker_id] = verdict
                    reused.add(tracker_id)
            print(
                f"Reusing {len(reused)} of {len(best_view)} answers "
                f"from {saved_path}"
            )
    todo = {t: v for t, v in best_view.items() if t not in reused}

    stats = ""
    if todo:
        try:
            classifier = DecisionsClassifier(config)
        except RuntimeError as exc:
            sys.exit(f"error: {exc}")

        # One pass over the video collects every crop.
        wanted: dict[int, list[int]] = defaultdict(list)
        for tracker_id, (frame_index, _) in todo.items():
            wanted[frame_index].append(tracker_id)
        crops: dict[int, np.ndarray] = {}
        last_wanted = max(wanted)
        for i, frame in enumerate(
            tqdm(
                sv.get_video_frames_generator(str(source_path)),
                total=last_wanted + 1,
                desc="Cropping",
                unit="frame",
            )
        ):
            for tracker_id in wanted.get(i, ()):
                crop = crop_box(frame, todo[tracker_id][1], config.crop_padding)
                if crop is not None:
                    crops[tracker_id] = crop
            if i >= last_wanted:
                break

        print(
            f"Decisions API: {config.model}, classifying {len(crops)} "
            f"objects ({config.concurrency} in parallel) with {config.path}"
        )
        with tqdm(total=len(crops), desc="Classifying", unit="obj") as bar:
            verdicts |= classifier.classify(
                crops, on_done=lambda done: bar.update(done - bar.n)
            )
        if classifier.calls and classifier.failures == classifier.calls:
            print(
                "warning: every decisions call failed (see above), "
                "objects are shown as unverified"
            )
        stats = classifier.stats()

    save_decisions(
        save_path,
        config,
        source_path.name,
        [
            {
                "track_id": tracker_id,
                "frame": best_view[tracker_id][0],
                "box": best_view[tracker_id][1],
                "verdict": verdict,
                "reused": tracker_id in reused,
            }
            for tracker_id, verdict in verdicts.items()
        ],
    )
    print(f"Saved decisions to: {save_path}")
    return verdicts, stats


def prepare_showcase(
    args,
    *,
    video_info: sv.VideoInfo,
    source_path: Path,
    target_path: Path,
    samples: list[tuple],
    counted: list[tuple[int, int, int]],
    observations: dict[int, list[tuple[int, np.ndarray]]],
    class_ids: list[int],
    color_map: dict[int, str],
    hud_rows: list[tuple[str, list[int]]],
    size_grades: list[tuple[str, float | None]] | None,
    line_zone: sv.LineZone | None,
    decisions_config: DecisionsConfig | None,
    saved_decisions: SavedDecisions | None,
) -> Showcase:
    """Measure (and optionally classify) the tracks after the tracking
    pass, and set up the showcase renderer and panel."""
    # Every track that is drawn or counted.
    track_ids = {
        int(t)
        for _, shown, *_ in samples
        if shown.tracker_id is not None
        for t in shown.tracker_id
    } | {t for _, t, _ in counted}
    tracks, best_view = measure_tracks(
        track_ids,
        observations,
        video_info.width,
        video_info.height,
        args.mm_per_px,
    )

    decisions_stats = ""
    if decisions_config is not None and best_view:
        verdicts, decisions_stats = classify_tracks(
            best_view,
            decisions_config,
            source_path,
            save_path=(
                Path(args.save_decisions)
                if args.save_decisions
                else target_path.with_name(f"{target_path.stem}_decisions.json")
            ),
            saved=saved_decisions,
            saved_path=args.load_decisions,
        )
        for tracker_id, verdict in verdicts.items():
            tracks[tracker_id].verdict = verdict.value

    choices = (
        [
            ChoiceStyle(c.value, hex_to_rgb(c.color), c.good, c.flag)
            for c in decisions_config.choices
        ]
        if decisions_config
        else None
    )
    # The big number is the total, so rows that count every class would
    # only repeat it.
    panel_rows = [
        (label, ids) for label, ids in hud_rows if set(ids) != set(class_ids)
    ]
    panel_stats = PanelStats(
        counted,
        tracks,
        fps=video_info.fps,
        window_s=args.throughput_window,
        update_s=args.throughput_update,
        rows=panel_rows,
        choices=choices,
        grades=size_grades,
        measure=bool(args.mm_per_px),
    )

    line_points = None
    if line_zone is not None:
        line_points = (
            line_zone.vector.start.x, line_zone.vector.start.y,
            line_zone.vector.end.x, line_zone.vector.end.y,
        )
    try:
        renderer = ShowcaseRenderer(
            video_info.width,
            video_info.height,
            video_info.fps,
            title=args.hud_title,
            accent_hex=(args.hud_color or args.line_color),
            # A single class without --colors gets the showcase teal.
            class_colors=(
                color_map
                if args.colors or len(class_ids) > 1
                else {class_ids[0]: GOOD_HEX}
            ),
            choices=choices,
            quality_title=decisions_config.title if decisions_config else "",
            position=args.hud_position,
            font_arg=args.hud_font,
            show_panel=not args.no_hud,
            show_tags=not args.no_labels,
            line=line_points,
            line_label=(
                None
                if args.in_text is None or is_hidden_label(args.in_text)
                else args.in_text
            ),
            window_s=args.throughput_window,
        )
    except ValueError as exc:
        sys.exit(f"error: {exc}")

    return Showcase(renderer, panel_stats, tracks, decisions_stats)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------


def main() -> None:
    args = parse_args()

    # -- Saved detections (--load-detections) -----------------------
    # The file records which detector produced it, which decides the
    # class namespace (COCO vs free-form) and the sampling.
    loaded: DetectionsFile | None = None
    if args.load_detections:
        try:
            loaded = load_detections(Path(args.load_detections))
        except ValueError as exc:
            sys.exit(f"error: {exc}")
        detector = loaded.info.get("detector")
        if detector not in DETECTORS:
            sys.exit(
                f"error: {args.load_detections}: unknown detector "
                f"{detector!r} (expected one of: {', '.join(DETECTORS)})"
            )
        if args.detector not in (DEFAULT_DETECTOR, detector):
            print(
                f"note: --detector {args.detector} ignored, detections "
                f"file was made with '{detector}'"
            )
    else:
        detector = args.detector

    # A fine-tuned model comes with the threshold chosen when it was
    # trained (best F1 on its validation frames).
    if args.threshold is None:
        recommended = (
            recommended_threshold(args.weights) if args.weights else None
        )
        args.threshold = recommended or DEFAULT_THRESHOLD
        if recommended:
            print(
                f"Threshold {recommended:g}, chosen for this model when it "
                "was trained (change it with --threshold)"
            )

    # A fine-tuned RF-DETR (--weights) knows its own classes, so it is
    # loaded before the classes are resolved.
    rfdetr_detector: RfDetrDetector | None = None
    if loaded is None and args.weights:
        detector = "rfdetr"
        try:
            rfdetr_detector = RfDetrDetector(
                args.model, args.threshold, weights=args.weights
            )
        except ValueError as exc:
            sys.exit(f"error: {exc}")

    # Which names --classes takes (the class namespace):
    # - the stock RF-DETR: the fixed COCO classes (with COCO's ids);
    # - a fine-tuned RF-DETR: the classes it was trained on;
    # - cloud / locate-anything: free-form text, and so is a saved file
    #   of a fine-tuned model (it is matched to the classes by name).
    model_classes = (
        dict(enumerate(rfdetr_detector.class_names))
        if rfdetr_detector is not None
        else None
    )
    free_classes = detector in OPEN_VOCAB_DETECTORS or bool(
        loaded is not None and loaded.info.get("weights")
    )
    coco_classes = model_classes is None and not free_classes

    if args.list_classes:
        if model_classes is not None:
            for class_id, name in model_classes.items():
                print(f"{class_id:3d}  {name}")
            return
        if free_classes:
            print(
                "With --detector cloud or locate-anything, --classes accepts "
                "any free-form text (e.g. 'traffic cone', "
                "'person wearing a helmet'); there is no fixed list."
            )
            return
        for class_id, name in sorted(load_coco_classes().items()):
            print(f"{class_id:3d}  {name}")
        return

    # -- Validate / resolve options --------------------------------
    # `coco` maps class id -> name for every class we can track.
    # RF-DETR: the fixed COCO set, or the fine-tuned model's classes.
    # Cloud/locate-anything: whatever the user typed.
    # There is no default set of classes: what to detect is always
    # given, except where it is already known (the classes a fine-tuned
    # model was trained on, or those chosen when a cloud /
    # locate-anything / fine-tuned file was detected).
    no_classes = (
        "error: --classes is required: say what to detect, e.g. "
        + (
            "--classes \"traffic cone\" (free-form text)"
            if free_classes
            else "--classes car truck (see --list-classes)"
        )
    )
    try:
        if model_classes is not None:
            coco = model_classes
            tokens = (
                split_tokens(args.classes)
                if args.classes
                else list(coco.values())
            )
        elif free_classes:
            if args.classes:
                tokens = split_tokens(args.classes)
            elif loaded is not None:
                tokens = [c["name"] for c in loaded.categories]
            else:
                sys.exit(no_classes)
            coco = build_free_classes(tokens)
        else:
            # A stock RF-DETR file holds every COCO class it found, so it
            # doesn't say what to detect either.
            if not args.classes:
                sys.exit(no_classes)
            coco = load_coco_classes()
            tokens = split_tokens(args.classes)
        class_ids = list(
            dict.fromkeys(  # dedupe, keep order
                resolve_class(t, coco) for t in tokens
            )
        )
        if not class_ids:
            raise ValueError("--classes: no classes given")
        color_map = build_color_map(
            class_ids, args.colors, coco, use_defaults=coco_classes
        )
        hud_rows = build_hud_rows(args.hud_rows, class_ids, coco)
        line_color = parse_color(args.line_color)
        hud_color = parse_color(args.hud_color or args.line_color)
        size_grades = (
            parse_size_grades(args.size_grades) if args.size_grades else None
        )
        # Loaded (and the API key checked) before detection, so a bad
        # config doesn't surface after a slow / paid detection pass.
        decisions_config = (
            DecisionsConfig.load(args.decisions) if args.decisions else None
        )
        saved_decisions = (
            SavedDecisions(Path(args.load_decisions), decisions_config)
            if args.load_decisions
            else None
        )
        if (
            decisions_config
            and not args.no_render
            and not os.environ.get("OPENAI_API_KEY")
            and saved_decisions is None
        ):
            raise ValueError(
                "--decisions needs the OPENAI_API_KEY environment variable "
                "(or put it in .env)"
            )
    except (ValueError, RuntimeError) as exc:
        sys.exit(f"error: {exc}")

    source_path = Path(args.source)
    if not source_path.is_file():
        sys.exit(f"error: source video not found: {source_path}")

    target_path = (
        Path(args.output)
        if args.output
        else Path.cwd() / f"{source_path.stem}_counted.mp4"
    )
    if not args.no_render:
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
            display_in_count=(
                args.direction in ("both", "in")
                and not is_hidden_label(args.in_text)
            ),
            display_out_count=(
                args.direction in ("both", "out")
                and not is_hidden_label(args.out_text)
            ),
        )

    # -- Sampling ---------------------------------------------------
    # The detector runs every `sample_step` frames: every frame (1.0) by
    # default for RF-DETR, a few times per second for the others.
    # A loaded file dictates the step it was recorded with.
    if loaded is not None:
        sample_step = float(loaded.info.get("sample_step", 1.0))
        if args.sample_rate is not None:
            print("note: --sample-rate ignored with --load-detections")
    else:
        sample_rate = args.sample_rate or DEFAULT_SAMPLE_RATES.get(detector)
        sample_step = (
            max(1.0, video_info.fps / sample_rate) if sample_rate else 1.0
        )
    sampled = sample_step > 1.0
    interpolate = args.interpolate
    if interpolate and sampled:
        print(
            "Interpolating boxes between samples "
            "(disable with --no-interpolate)"
        )
    effective_rate = video_info.fps / sample_step
    expected_calls = int(np.ceil((video_info.total_frames or 0) / sample_step))

    if args.min_track_frames is None:
        args.min_track_frames = 1 if sampled else 3

    # -- Detection --------------------------------------------------
    # Every detector (or a loaded file) runs up front, before anything is
    # tracked or drawn.
    if loaded is not None:
        run = detections_from_file(
            loaded, args.load_detections, detector, coco, args.threshold,
            video_info,
        )
    else:
        run = run_detector(
            detector,
            args,
            class_names={c: coco[c] for c in class_ids},
            rfdetr_detector=rfdetr_detector,
            source_path=source_path,
            video_info=video_info,
            sample_step=sample_step,
            effective_rate=effective_rate,
            expected_calls=expected_calls,
        )
    detections_by_frame = run.detections_by_frame

    # Detection is done already; save right away so the (slow / paid)
    # detections survive a failure while rendering.
    if args.save_detections:
        write_detections(
            Path(args.save_detections),
            args,
            run,
            detector=detector,
            sample_step=sample_step,
            coco=coco,
            class_ids=class_ids,
            coco_classes=coco_classes,
            source_path=source_path,
            video_info=video_info,
        )
    if args.no_render:
        if run.stats:
            print(run.stats)
        return

    # -- Tracker ----------------------------------------------------
    # The tracker only ever sees sampled frames, so it has to be told
    # the sampled rate, not the video FPS.
    from trackers import ByteTrackTracker

    # Some detectors return only part of the objects on a frame. A track
    # they miss in up to --hold-frames detections in a row stays on
    # screen: held where it was, or interpolated towards where it is
    # found again (--interpolate).
    hold_frames = args.hold_frames
    if hold_frames is None:
        hold_frames = (
            DEFAULT_HOLD_FRAMES if detector in OPEN_VOCAB_DETECTORS else 0
        )

    byte_tracker = ByteTrackTracker(
        # The tracker has to keep the id of a track across those missed
        # detections (its default is 1 s; the unit is 30 FPS frames).
        lost_track_buffer=max(
            30, math.ceil(hold_frames * 30 / effective_rate)
        ),
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
        # Traces need a position for every frame; with held (sampled)
        # detections they would just be a dot, so they are skipped
        # unless the boxes are interpolated.
        if args.trace_length > 0 and (not sampled or interpolate)
        else None
    )

    hud = (
        None
        if args.no_hud or args.style != "classic"
        else Hud(
            args.hud_title,
            hud_rows,
            (hud_color.b, hud_color.g, hud_color.r),
        )
    )

    min_w, min_h = args.min_size
    max_w, max_h = args.max_size
    hyst = args.size_hysteresis

    # {track id: [its box as a one-row Detections, detections missed in a row]}
    last_shown: dict[int, list] = {}

    def with_held(shown: sv.Detections) -> sv.Detections:
        """`shown` plus the recently seen tracks that are missing from it.

        Not used with --interpolate, where the boxes of a missed track
        are interpolated between its detections instead.
        """
        if hold_frames <= 0 or interpolate:
            return shown

        present: set[int] = set()
        if shown.tracker_id is not None:
            for row, tracker_id in enumerate(shown.tracker_id):
                present.add(int(tracker_id))
                last_shown[int(tracker_id)] = [shown[row], 0]

        held = []
        for tracker_id, entry in list(last_shown.items()):
            if tracker_id in present:
                continue
            entry[1] += 1
            if entry[1] > hold_frames:
                del last_shown[tracker_id]
            else:
                held.append(entry[0])
        return sv.Detections.merge([shown, *held]) if held else shown

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
    # One (frame index, track id, class id) per count, in frame order.
    counted: list[tuple[int, int, int]] = []
    # {track id: [(frame index, xyxy), ...]} of every tracked box, for
    # measuring and classifying (showcase style).
    observations: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)

    def process(detections: sv.Detections, index: int) -> sv.Detections:
        """Track, filter and count one set of detections.

        Returns the detections to draw. Called once per sampled frame
        (every frame unless detection is sampled), in frame order.
        """
        # Keep only the classes we care about
        detections = detections[
            np.isin(detections.class_id, class_ids)
        ]

        detections = byte_tracker.update(detections)

        # Remove detections without confirmed track IDs
        if detections.tracker_id is not None:
            detections = detections[detections.tracker_id >= 0]
            for xyxy, tracker_id in zip(
                detections.xyxy, detections.tracker_id
            ):
                observations[int(tracker_id)].append((index, xyxy.copy()))

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
                        counted.append(
                            (index, int(tracker_id), int(class_id))
                        )
        else:
            crossed_in, crossed_out = line_zone.trigger(detections)

            if args.direction == "in":
                crossed = crossed_in
            elif args.direction == "out":
                crossed = crossed_out
            else:
                crossed = crossed_in | crossed_out

            if np.any(crossed):
                for class_id, tracker_id in zip(
                    detections.class_id[crossed],
                    detections.tracker_id[crossed],
                ):
                    counts[int(class_id)] += 1
                    counted.append((index, int(tracker_id), int(class_id)))

        return with_held(display_detections)

    # -- Tracking pass ----------------------------------------------
    # Tracking, size filter and counting run over every sample up front:
    # interpolating a frame between two samples needs the tracked boxes
    # of the sample after it, and measuring / classifying an object
    # needs all of its boxes. Each sample keeps a snapshot of the counts
    # (HUD and line) as they were right after it, which is what is shown
    # until the next sample. Without --interpolate a sample's boxes are
    # held (unchanged) until the next one.
    # samples: (frame index, boxes to draw, HUD counts, line in, line out)
    samples: list[tuple[int, sv.Detections, dict, int, int]] = []
    for sample_index in sorted(detections_by_frame):
        detections = detections_by_frame[sample_index]
        if detections is None:  # detection call failed
            continue
        shown = process(detections, sample_index)
        samples.append(
            (
                sample_index,
                shown,
                dict(counts),
                line_zone.in_count if line_zone else 0,
                line_zone.out_count if line_zone else 0,
            )
        )
    sample_indices = [s[0] for s in samples]
    timeline = TrackTimeline(
        sample_indices, [s[1] for s in samples], hold_frames
    )

    def state_at(index: int):
        """(boxes, HUD counts, line) to show on frame `index`."""
        pos = bisect.bisect_right(sample_indices, index) - 1
        if pos < 0:  # before the first successful sample
            empty_line = CountedLine(line_zone, 0, 0) if line_zone else None
            return timeline.at(index), {}, empty_line
        _, held_boxes, hud_counts, line_in, line_out = samples[pos]
        boxes = timeline.at(index) if interpolate else held_boxes
        line = (
            CountedLine(line_zone, line_in, line_out)
            if line_zone
            else None
        )
        return boxes, hud_counts, line

    # -- Measurement / classification (showcase style) ---------------
    showcase = None
    if args.style == "showcase":
        showcase = prepare_showcase(
            args,
            video_info=video_info,
            source_path=source_path,
            target_path=target_path,
            samples=samples,
            counted=counted,
            observations=observations,
            class_ids=class_ids,
            color_map=color_map,
            hud_rows=hud_rows,
            size_grades=size_grades,
            line_zone=line_zone,
            decisions_config=decisions_config,
            saved_decisions=saved_decisions,
        )

    # -- Frame callback ---------------------------------------------
    def callback(frame: np.ndarray, index: int) -> np.ndarray:
        display_detections, shown_counts, shown_line = state_at(index)

        if showcase is not None:
            if trace_annotator is not None:
                frame = trace_annotator.annotate(
                    scene=frame.copy(), detections=display_detections
                )
            return showcase.draw(frame, index, display_detections)

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
                annotated, line_counter=shown_line
            )

        if hud is not None:
            annotated = hud.draw(annotated, shown_counts)

        return annotated

    # -- Process ----------------------------------------------------
    sv.process_video(
        source_path=str(source_path),
        target_path=str(target_path),
        callback=callback,
        show_progress=True,
        progress_message="Rendering video",
    )

    if run.stats:
        print(run.stats)
    if showcase is not None and showcase.decisions_stats:
        print(showcase.decisions_stats)

    print(f"Saved to: {target_path}")


if __name__ == "__main__":
    main()

"""Fine-tune RF-DETR on saved detections.

Use a big or slow model (cloud, locate-anything, RF-DETR large, boxes
fixed in the annotator) to label some frames, then train a small RF-DETR
on them and run that on every frame of the same or similar videos:

    python train_rfdetr.py dets.json -o models/apples
    python process_video.py other.mp4 --weights models/apples ...

Each detections file is paired with the video named in it (found under
--root), or with the matching --videos entry. The output folder gets:

    model.json                  classes, training settings, scores
    checkpoint_best_total.pth   the weights to use (--weights)
    dataset/                    the frames and boxes trained on
    metrics.csv, training_config.json   RF-DETR's own logs

Run `python train_rfdetr.py --help` for the options.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import signal
import sys
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path

from file_lookup import best_video, find_videos
from options import split_tokens
from rfdetr_detector import MANIFEST_FILE, MODEL_SIZES, WEIGHTS_FILE
from training_data import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MIN_SCORE,
    DEFAULT_VAL_FRACTION,
    Source,
    build_dataset,
)

MANIFEST = MANIFEST_FILE
MANIFEST_FORMAT = "rfdetr-finetune"
DEFAULT_TRAIN_MODEL = "small"
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 4
DEFAULT_GRAD_ACCUM = 2
DEFAULT_LR = 1e-4
# Checkpoints RF-DETR writes besides checkpoint_best_total.pth (plus a
# checkpoint_<epoch>.ckpt every 10 epochs), removed after training
# unless --keep-checkpoints: they are only needed to resume a run, and
# each takes 1-4x the space of the model itself.
EXTRA_CHECKPOINTS = ("last.ckpt", "last_ema.pth", "checkpoint_best_ema.pth")


def extra_checkpoints(out: Path) -> list[Path]:
    return [out / n for n in EXTRA_CHECKPOINTS] + sorted(
        out.glob("checkpoint_*.ckpt")
    )
# Prefix of the machine-readable progress lines (--progress-json), read
# by the annotator UI.
PROGRESS_PREFIX = "@progress "
# Best checkpoints RF-DETR keeps while training; checkpoint_best_total.pth
# is made from them when training ends normally. After Ctrl-C it isn't,
# so the first one found takes its place.
BEST_CHECKPOINTS = ("checkpoint_best_ema.pth", "checkpoint_best_regular.pth")

# Set by SIGUSR1 (the annotator's Stop button): finish the current batch,
# score the model, and end training normally, keeping the best weights.
stop_requested = threading.Event()


EPILOG = """\
Examples
--------
Train on one detections file (its video is found under the current
folder), then use the model:

    python train_rfdetr.py videos/conveyor/processed/dets.json \\
        -o models/apples --epochs 20
    python process_video.py videos/conveyor/other.mp4 \\
        --weights models/apples --count-mode total

Several files, only the frames fixed by hand, one class, every 2nd frame:

    python train_rfdetr.py a.json b.json --only-edited \\
        --classes apple --every 2 -o models/apples-reviewed
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fine-tune an RF-DETR model on detections files saved with "
            "--save-detections (and fixed in the annotator)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )
    p.add_argument(
        "detections",
        nargs="+",
        metavar="DETECTIONS.json",
        help="Detections files to learn from.",
    )
    p.add_argument(
        "--videos",
        nargs="+",
        metavar="VIDEO",
        help=(
            "The videos of the detections files, in the same order "
            "(default: the video named in each file, looked up under "
            "--root)."
        ),
    )
    p.add_argument(
        "--root",
        default=".",
        help="Folder searched for the videos (default: current directory).",
    )
    p.add_argument(
        "-o",
        "--output",
        help=(
            "Folder for the model (default: models/<first detections "
            "file name>-<model>). Must not exist yet, unless --overwrite."
        ),
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the model in --output if there is one.",
    )

    data = p.add_argument_group("training data")
    data.add_argument(
        "--classes",
        nargs="+",
        metavar="CLASS",
        help=(
            "Classes to learn, e.g. `apple` or `car truck` (default: "
            "every class with boxes). Boxes of other classes are left out."
        ),
    )
    data.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help=(
            "Leave out boxes below this confidence (default: "
            f"{DEFAULT_MIN_SCORE:g}). Cloud / locate-anything boxes and "
            "boxes drawn by hand have 1.0."
        ),
    )
    data.add_argument(
        "--only-edited",
        action="store_true",
        help="Use only the frames changed in the annotator.",
    )
    data.add_argument(
        "--every",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Use every N-th frame of each file (default: 1, all). "
            "Neighbouring frames add little when detecting on every frame."
        ),
    )
    data.add_argument(
        "--val-fraction",
        type=float,
        default=DEFAULT_VAL_FRACTION,
        help=(
            "Share of each video's frames held out to score the model, "
            "in contiguous stretches (default: "
            f"{DEFAULT_VAL_FRACTION:g})."
        ),
    )
    data.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        metavar="PX",
        help=(
            "Shorter side of the stored frames, never upscaled; 0 keeps "
            f"the video's size (default: {DEFAULT_IMAGE_SIZE})."
        ),
    )

    train = p.add_argument_group("training")
    train.add_argument(
        "--model",
        choices=MODEL_SIZES,
        default=DEFAULT_TRAIN_MODEL,
        help=(
            "RF-DETR size to fine-tune; bigger is more accurate and "
            f"slower (default: {DEFAULT_TRAIN_MODEL})."
        ),
    )
    train.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Passes over the training frames (default: {DEFAULT_EPOCHS}).",
    )
    train.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Frames per step; lower it if you run out of memory "
            f"(default: {DEFAULT_BATCH_SIZE})."
        ),
    )
    train.add_argument(
        "--grad-accum",
        type=int,
        default=DEFAULT_GRAD_ACCUM,
        metavar="N",
        help=(
            "Steps accumulated per update; batch size x N is the "
            f"effective batch (default: {DEFAULT_GRAD_ACCUM})."
        ),
    )
    train.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help=f"Learning rate (default: {DEFAULT_LR:g}).",
    )
    train.add_argument(
        "--patience",
        type=int,
        default=0,
        metavar="EPOCHS",
        help=(
            "Stop early when the validation score hasn't improved for "
            "this many epochs (default: 0, off)."
        ),
    )
    train.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Where to train (default: auto, CUDA > Apple GPU > CPU).",
    )
    train.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Data loading processes (default: 2).",
    )
    train.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help=(
            "Keep RF-DETR's other checkpoints (last.ckpt, ...), needed "
            "only to resume training by hand."
        ),
    )
    p.add_argument(
        "--progress-json",
        action="store_true",
        help=argparse.SUPPRESS,  # progress lines for the annotator UI
    )

    args = p.parse_args(argv)
    if args.videos and len(args.videos) != len(args.detections):
        p.error("--videos needs one video per detections file")
    if args.epochs < 1:
        p.error("--epochs must be >= 1")
    if args.batch_size < 1 or args.grad_accum < 1:
        p.error("--batch-size and --grad-accum must be >= 1")
    if not 0 < args.val_fraction <= 0.5:
        p.error("--val-fraction must be in (0, 0.5]")
    if args.every < 1:
        p.error("--every must be >= 1")
    if args.image_size and args.image_size < 64:
        p.error("--image-size must be 0 or >= 64")
    return args


def resolve_sources(
    detections: list[str], videos: list[str] | None, root: Path
) -> list[Source]:
    """Pair every detections file with its video."""
    sources = []
    found: list[Path] | None = None
    for i, name in enumerate(detections):
        path = Path(name)
        if not path.is_file():
            raise ValueError(f"no such file: {path}")
        source = Source(path, Path())
        data = source.load()
        if videos:
            video = Path(videos[i])
        else:
            video_name = data.video.get("file_name")
            if not video_name:
                raise ValueError(f"{path} doesn't name its video; pass --videos")
            if found is None:
                found = find_videos(root.resolve())
            video = best_video(path.resolve(), video_name, found)
            if video is None:
                raise ValueError(
                    f"can't find the video {video_name} of {path} under "
                    f"{root}; pass --videos or --root"
                )
        if not video.is_file():
            raise ValueError(f"no such video: {video}")
        source.video = video
        sources.append(source)
    return sources


def pick_device(wanted: str) -> str:
    if wanted != "auto":
        return wanted
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_output(detections: str, model: str) -> Path:
    stem = Path(detections).stem
    for suffix in ("_dets", "-dets", "_detections"):
        stem = stem.removesuffix(suffix)
    return Path("models") / f"{stem}-{model}"


def prepare_output(out: Path, overwrite: bool) -> None:
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise ValueError(
                f"{out} already exists; choose another --output or pass "
                "--overwrite"
            )
        if not (out / MANIFEST).exists() and not (out / "dataset").exists():
            raise ValueError(
                f"{out} is not empty and doesn't hold a model trained by "
                "this tool; choose another --output"
            )
        for path in (out / MANIFEST, out / WEIGHTS_FILE, *extra_checkpoints(out)):
            path.unlink(missing_ok=True)
    out.mkdir(parents=True, exist_ok=True)


class Reporter:
    """Progress output: a readable line per epoch, plus (with
    --progress-json) one JSON line per update for the UI."""

    def __init__(self, json_lines: bool):
        self.json_lines = json_lines
        self._last_emit = 0.0

    def emit(self, force: bool = True, **fields) -> None:
        if not self.json_lines:
            return
        now = time.monotonic()
        if not force and now - self._last_emit < 0.5:
            return
        self._last_emit = now
        print(PROGRESS_PREFIX + json.dumps(fields), flush=True)


def metric(metrics: dict, key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else round(value, 4)


def progress_callback(reporter: Reporter, history: list[dict]):
    """A Lightning callback that reports batches and validation scores."""
    from pytorch_lightning import Callback
    from tqdm import tqdm

    class Progress(Callback):
        def on_train_epoch_start(self, trainer, module):
            self.started = time.monotonic()
            self.batches = trainer.num_training_batches
            self.losses: list[float] = []
            self.bar = (
                None
                if reporter.json_lines
                else tqdm(
                    total=self.batches,
                    desc=f"Epoch {trainer.current_epoch + 1}/{trainer.max_epochs}",
                    unit="batch",
                    leave=False,
                )
            )

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            if stop_requested.is_set() and not trainer.should_stop:
                # Lightning scores this partial epoch, then ends training
                # as if it was the last one.
                trainer.should_stop = True
                print("Stopping: scoring the model one last time...", flush=True)
                reporter.emit(phase="stopping")
            loss = metric(trainer.callback_metrics, "loss")
            if loss is not None:
                self.losses.append(loss)
            if self.bar is not None:
                self.bar.update(1)
                if loss is not None:
                    self.bar.set_postfix(loss=f"{loss:.3f}")
            reporter.emit(
                force=False,
                phase="training",
                epoch=trainer.current_epoch + 1,
                epochs=trainer.max_epochs,
                batch=batch_idx + 1,
                batches=self.batches,
                loss=loss,
            )

        def on_train_epoch_end(self, trainer, module):
            if self.bar is not None:
                self.bar.close()

        def on_validation_end(self, trainer, module):
            if trainer.sanity_checking:
                return
            m = trainer.callback_metrics
            entry = {
                "epoch": trainer.current_epoch + 1,
                # Lightning logs the epoch's train/loss only after
                # validation, so average the steps here.
                "loss": (
                    round(sum(self.losses) / len(self.losses), 4)
                    if self.losses
                    else None
                ),
                "map": metric(m, "val/mAP_50_95"),
                "map50": metric(m, "val/mAP_50"),
                "f1": metric(m, "val/F1"),
                "precision": metric(m, "val/precision"),
                "recall": metric(m, "val/recall"),
                "seconds": round(time.monotonic() - self.started, 1),
            }
            history.append(entry)
            fmt = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
            print(
                f"Epoch {entry['epoch']}/{trainer.max_epochs}: "
                f"loss {fmt(entry['loss'])}, mAP50 {fmt(entry['map50'])}, "
                f"mAP50-95 {fmt(entry['map'])}, F1 {fmt(entry['f1'])} "
                f"({entry['seconds']:g} s)",
                flush=True,
            )
            reporter.emit(
                phase="validated",
                epochs=trainer.max_epochs,
                history=history,
            )

    return Progress()


@contextlib.contextmanager
def extra_callback(callback):
    """Add `callback` to the Lightning trainer RF-DETR builds (its
    train() has no way to pass one)."""
    import rfdetr.training as training

    original = training.build_trainer

    def build_trainer(*args, **kwargs):
        trainer = original(*args, **kwargs)
        trainer.callbacks.append(callback)
        return trainer

    training.build_trainer = build_trainer
    try:
        yield
    finally:
        training.build_trainer = original


def calibrate_threshold(weights: Path, dataset: Path) -> dict | None:
    """The confidence threshold with the best F1 on the validation frames.

    A fine-tuned model's scores depend on how long it trained (a short
    run may score every object below 0.3), so no fixed threshold suits
    every model. Each prediction is matched to an unmatched box of its
    class (IoU >= 0.5, highest scores first) and F1 is computed at every
    threshold from 0.01 to 0.95. None if nothing is found at any."""
    import cv2
    import numpy as np
    import supervision as sv

    from rfdetr_detector import RfDetrDetector

    split = dataset / "valid"
    doc = json.loads((split / "_annotations.coco.json").read_text())
    truth: dict[int, list] = {}
    for ann in doc["annotations"]:
        x, y, w, h = ann["bbox"]
        truth.setdefault(ann["image_id"], []).append(
            (ann["category_id"] - 1, [x, y, x + w, y + h])
        )
    if not truth:
        return None

    with contextlib.redirect_stdout(sys.stderr):
        detector = RfDetrDetector(threshold=0.01, weights=weights)
    scores: list[float] = []
    matched: list[bool] = []
    total = 0
    for image in doc["images"]:
        frame = cv2.imread(str(split / image["file_name"]))
        if frame is None:
            continue
        det = detector.detect(frame)
        boxes = truth.get(image["id"], [])
        total += len(boxes)
        if not len(det):
            continue
        iou = (
            sv.box_iou_batch(det.xyxy, np.array([b for _, b in boxes]))
            if boxes
            else np.zeros((len(det), 0))
        )
        taken: set[int] = set()
        for i in np.argsort(-det.confidence):
            candidates = [
                (iou[i, j], j)
                for j, (cls, _) in enumerate(boxes)
                if j not in taken and cls == det.class_id[i] and iou[i, j] >= 0.5
            ]
            if candidates:
                taken.add(max(candidates)[1])
            scores.append(float(det.confidence[i]))
            matched.append(bool(candidates))
    if total == 0:
        return None

    scores_arr, matched_arr = np.array(scores), np.array(matched, dtype=bool)
    best = None
    for threshold in np.round(np.arange(0.01, 0.951, 0.01), 2):
        above = scores_arr >= threshold
        tp = int((above & matched_arr).sum())
        fp = int(above.sum()) - tp
        f1 = 2 * tp / (2 * tp + fp + (total - tp))
        if best is None or f1 > best["f1"] + 1e-9:
            best = {
                "threshold": float(threshold),
                "f1": round(f1, 4),
                "precision": round(tp / max(1, tp + fp), 4),
                "recall": round(tp / total, 4),
            }
    return best if best and best["f1"] > 0 else None


def best_epoch(history: list[dict]) -> dict | None:
    scored = [h for h in history if h.get("map") is not None]
    return max(scored, key=lambda h: h["map"]) if scored else None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reporter = Reporter(args.progress_json)
    root = Path(args.root)
    out = Path(args.output) if args.output else default_output(
        args.detections[0], args.model
    )

    try:
        sources = resolve_sources(args.detections, args.videos, root)
        prepare_output(out, args.overwrite)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # -- Dataset ----------------------------------------------------
    print(f"Building the training data in {out / 'dataset'}")
    for source in sources:
        print(f"  {source.detections}  ({source.video})")

    from tqdm import tqdm

    bar = None if args.progress_json else tqdm(desc="Extracting frames", unit="frame")

    def on_frame(done: int, total: int) -> None:
        if stop_requested.is_set():
            raise KeyboardInterrupt
        if bar is not None:
            bar.total = total
            bar.update(done - bar.n)
        reporter.emit(force=done == total, phase="dataset", done=done, total=total)

    try:
        summary = build_dataset(
            sources,
            out / "dataset",
            classes=split_tokens(args.classes) if args.classes else None,
            min_score=args.min_score,
            val_fraction=args.val_fraction,
            every=args.every,
            only_edited=args.only_edited,
            image_size=args.image_size,
            on_progress=on_frame,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Stopped before training, no model was saved.")
        reporter.emit(phase="stopped")
        return 130
    finally:
        if bar is not None:
            bar.close()

    print(
        f"Classes: {', '.join(summary['classes'])}. "
        f"Frames: {summary['images']['train']} train, "
        f"{summary['images']['valid']} validation. Boxes: "
        + ", ".join(f"{n} {c}" for c, n in summary["boxes_per_class"].items())
    )
    reporter.emit(phase="dataset_done", dataset=summary)

    # -- Training ---------------------------------------------------
    warnings.filterwarnings("ignore", category=FutureWarning)
    import rfdetr

    device = pick_device(args.device)
    print(
        f"Fine-tuning RF-DETR {args.model} on {device}: {args.epochs} "
        f"epochs, batch {args.batch_size} x {args.grad_accum}"
    )
    reporter.emit(phase="loading", device=device)
    model = getattr(rfdetr, f"RFDETR{args.model.capitalize()}")()
    if stop_requested.is_set():
        print("Stopped before training, no model was saved.")
        reporter.emit(phase="stopped")
        return 130

    history: list[dict] = []
    started = time.monotonic()
    stopped = False
    with extra_callback(progress_callback(reporter, history)):
        try:
            model.train(
                dataset_dir=str(out / "dataset"),
                output_dir=str(out),
                epochs=args.epochs,
                batch_size=args.batch_size,
                grad_accum_steps=args.grad_accum,
                lr=args.lr,
                device=device,
                num_workers=args.workers,
                early_stopping=args.patience > 0,
                early_stopping_patience=max(args.patience, 1),
                tensorboard=False,
            )
        except KeyboardInterrupt:
            stopped = True
        except SystemExit as exc:
            # Lightning exits after handling Ctrl-C (or the UI's Stop).
            if exc.code in (0, None):
                raise
            stopped = True

    stopped = stopped or stop_requested.is_set()
    weights = out / WEIGHTS_FILE
    if not weights.exists():
        best_so_far = next(
            (out / n for n in BEST_CHECKPOINTS if (out / n).exists()), None
        )
        if best_so_far is not None:
            os.replace(best_so_far, weights)
    if not weights.exists():
        if stopped:
            print("Stopped before the first epoch was scored, no model was saved.")
            reporter.emit(phase="stopped")
            return 130
        print("error: training produced no checkpoint", file=sys.stderr)
        reporter.emit(phase="failed")
        return 1
    if not args.keep_checkpoints:
        for path in extra_checkpoints(out):
            path.unlink(missing_ok=True)

    print("Choosing the confidence threshold on the validation frames...")
    reporter.emit(phase="calibrating")
    calibration = calibrate_threshold(weights, out / "dataset")
    if calibration:
        print(
            f"Threshold {calibration['threshold']:g}: F1 {calibration['f1']}, "
            f"precision {calibration['precision']}, recall "
            f"{calibration['recall']}"
        )
        if calibration["threshold"] < 0.1:
            print(
                "warning: the model is still very unsure of what it finds "
                "(it needs a threshold this low); train for more epochs"
            )
    else:
        print(
            "warning: the model finds none of the validation boxes; "
            "train for more epochs or on more frames"
        )

    best = best_epoch(history)
    manifest = {
        "format": MANIFEST_FORMAT,
        "version": 1,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "base_model": args.model,
        "classes": summary["classes"],
        "weights": WEIGHTS_FILE,
        # process_video.py --weights uses it unless --threshold is given.
        "threshold": calibration["threshold"] if calibration else None,
        "calibration": calibration,
        "stopped_early": stopped,
        "best": best,
        "history": history,
        "training": {
            "epochs": args.epochs,
            "epochs_run": len(history),
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "patience": args.patience,
            "device": device,
            "minutes": round((time.monotonic() - started) / 60, 1),
        },
        "dataset": summary,
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=2))

    if stopped:
        print(f"Training stopped after {len(history)} epochs; keeping the best so far.")
    if best:
        print(
            f"Best epoch {best['epoch']}: mAP50 {best['map50']}, "
            f"mAP50-95 {best['map']}"
        )
    print(f"Saved the model to: {out}")
    print(f"Use it with: python process_video.py VIDEO --weights {out}")
    reporter.emit(phase="done", output=str(out), best=best, stopped=stopped)
    return 0


if __name__ == "__main__":
    # Data loader workers re-import this module (spawn), so everything
    # runs from here.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, lambda *_: stop_requested.set())
    sys.exit(main())

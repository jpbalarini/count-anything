"""The `showcase` look (--style showcase): corner brackets with ID chips
on every object, a dashed count line that pulses on each count, and one
side panel with the total, throughput and average size, the
classification split as a donut (--decisions) and the size grades
(--size-grades).

Shapes are drawn with OpenCV (anti-aliased), text with Pillow so a real
font can be used.
"""

from __future__ import annotations

import bisect
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Colors are RGB; converted to BGR where OpenCV draws them.
WHITE = (244, 244, 242)
MUTED = (154, 160, 166)
PANEL_BG = (15, 18, 20)
SEPARATOR = (40, 52, 58)
PANEL_TOP = (22, 33, 39)  # vertical gradient of the panel background
PANEL_BOTTOM = (10, 15, 19)
PANEL_BORDER = (66, 98, 106)
BAR_DIM = (88, 92, 96)
GOOD_HEX = "#5DCAA5"
GOOD = (93, 202, 165)
GOOD_DARK = (8, 80, 65)
UNVERIFIED = (205, 208, 212)
LINE_IDLE = (225, 225, 225)
CHIP_ALPHA = 0.84
PANEL_ALPHA = 0.96

# (regular, medium, bold) font candidates, tried in order. Each is
# (path, face index in a .ttc, named weight of a variable font).
FONT_CANDIDATES = [
    tuple(
        (str(Path.home() / "Library/Fonts/Manrope-VariableFont_wght.ttf"), 0, w)
        for w in ("Medium", "SemiBold", "Bold")
    ),
    (("/System/Library/Fonts/HelveticaNeue.ttc", 0, None),
     ("/System/Library/Fonts/HelveticaNeue.ttc", 10, None),
     ("/System/Library/Fonts/HelveticaNeue.ttc", 1, None)),
    (("/usr/share/fonts/truetype/inter/Inter-Regular.ttf", 0, None),
     ("/usr/share/fonts/truetype/inter/Inter-Medium.ttf", 0, None),
     ("/usr/share/fonts/truetype/inter/Inter-Bold.ttf", 0, None)),
    (("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0, None),
     ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0, None),
     ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0, None)),
]

PULSE_SECONDS = 0.35  # count line highlight after a count
LAST_ITEM_SECONDS = 0.5  # the "last item" thumbnail changes at most this often
THUMB_SIZE = 150  # px at 1080p


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return rgb[2], rgb[1], rgb[0]


def lerp(a, b, t: float) -> tuple[int, int, int]:
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def fmt_int(value: int) -> str:
    return f"{value:,}"


def fmt_compact(value: float) -> str:
    """24686 -> '24,686', 246860 -> '247k', 1.2e6 -> '1.2M'."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 100_000:
        return f"{value / 1000:.0f}k"
    return fmt_int(round(value))


class Fonts:
    """Pillow fonts at the sizes the overlay uses, scaled to the video."""

    def __init__(self, scale: float, font_arg: str | None):
        regular, medium, bold = self._resolve(font_arg)

        def load(spec, size):
            return self._load(spec, max(8, round(size * scale)))

        self._bold = bold
        self._sized: dict[int, ImageFont.FreeTypeFont] = {}

        self.caps = load(medium, 17)  # small uppercase labels
        self.big = load(bold, 136)
        self.chip = load(medium, 19)
        self.label = load(regular, 21)
        self.value = load(medium, 23)
        self.small = load(regular, 17)
        self.stat = load(bold, 46)
        self.donut = load(bold, 36)
        self.unit = load(medium, 20)
        self.tag = load(medium, 16)

    @staticmethod
    def _load(spec, size: int) -> ImageFont.FreeTypeFont:
        path, index, weight = spec
        font = ImageFont.truetype(path, size, index=index)
        if weight:
            font.set_variation_by_name(weight)
        return font

    def big_fitting(self, text: str, max_width: float) -> ImageFont.FreeTypeFont:
        """The big-number font, shrunk until `text` fits in `max_width`."""
        font, size = self.big, self.big.size
        while font.getlength(text) > max_width and size > 40:
            size -= 6
            if size not in self._sized:
                self._sized[size] = self._load(self._bold, size)
            font = self._sized[size]
        return font

    @staticmethod
    def _resolve(font_arg: str | None):
        if font_arg:
            paths = [p.strip() for p in font_arg.split(",") if p.strip()]
            for path in paths:
                if not Path(path).is_file():
                    raise ValueError(f"--hud-font: file not found: {path}")
            paths += [paths[-1]] * (3 - len(paths))
            return tuple((path, 0, None) for path in paths[:3])
        for candidate in FONT_CANDIDATES:
            if all(Path(spec[0]).is_file() for spec in candidate):
                return candidate
        # matplotlib (an ultralytics dependency) ships DejaVu Sans.
        import matplotlib

        fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        bold = (str(fonts / "DejaVuSans-Bold.ttf"), 0, None)
        return (str(fonts / "DejaVuSans.ttf"), 0, None), bold, bold


# ---------------------------------------------------------
# What the overlay shows
# ---------------------------------------------------------


@dataclass
class ChoiceStyle:
    value: str
    color: tuple[int, int, int]
    good: bool
    flag: bool


@dataclass
class TrackInfo:
    """Per-object facts known before rendering."""

    size_mm: float | None = None
    verdict: str | None = None  # choice value, None = unverified/not run


@dataclass
class PanelState:
    total: int
    recent: int
    rate_per_hour: float | None
    rows: list[tuple[str, int]] = field(default_factory=list)
    choice_counts: list[tuple[ChoiceStyle, int]] = field(default_factory=list)
    unverified: int = 0
    grade_counts: list[tuple[str, int]] = field(default_factory=list)
    avg_mm: float | None = None
    measure: bool = False  # sizes are measured (avg_mm may still be None)
    last_track: int | None = None  # the "last item" shown next to the count
    last_class: int | None = None


class PanelStats:
    """Turns the counting events into what the panel shows on a frame.

    `counted` is one (frame index, track id, class id) per count, in
    frame order. Everything about the counted objects (size, verdict)
    comes from `tracks`.
    """

    def __init__(
        self,
        counted: list[tuple[int, int, int]],
        tracks: dict[int, TrackInfo],
        fps: float,
        window_s: float,
        update_s: float,
        rows: list[tuple[str, list[int]]],
        choices: list[ChoiceStyle] | None,
        grades: list[tuple[str, float | None]] | None,
        measure: bool,
    ):
        self.counted = counted
        self.frames = [c[0] for c in counted]
        self.tracks = tracks
        self.fps = fps
        self.window_s = window_s
        self.update_step = max(1, round(update_s * fps))  # frames
        self.last_item_step = max(1, round(LAST_ITEM_SECONDS * fps))
        self.rows = rows
        self.choices = choices
        self.grades = grades
        self.measure = measure
        self._cache: dict[int, tuple] = {}

    def _summary(self, n: int):
        if n in self._cache:
            return self._cache[n]
        items = self.counted[:n]
        rows = [
            (label, sum(1 for _, _, c in items if c in ids))
            for label, ids in self.rows
        ]
        infos = [self.tracks.get(t) for _, t, _ in items]
        choice_counts, unverified = [], 0
        if self.choices is not None:
            by_value = {c.value: 0 for c in self.choices}
            for info in infos:
                if info is not None and info.verdict in by_value:
                    by_value[info.verdict] += 1
                else:
                    unverified += 1
            choice_counts = [(c, by_value[c.value]) for c in self.choices]
        sizes = [i.size_mm for i in infos if i is not None and i.size_mm]
        grade_counts = []
        if self.grades:
            counts = [0] * len(self.grades)
            for size in sizes:
                counts[grade_index(size, self.grades)] += 1
            grade_counts = [(g[0], c) for g, c in zip(self.grades, counts)]
        avg_mm = statistics.fmean(sizes) if self.measure and sizes else None
        result = (rows, choice_counts, unverified, grade_counts, avg_mm)
        self._cache = {n: result}  # frames only move forward
        return result

    def at(self, index: int) -> PanelState:
        n = bisect.bisect_right(self.frames, index)
        rows, choice_counts, unverified, grade_counts, avg_mm = self._summary(n)
        recent = self._recent(index)

        # Throughput is only recalculated every `update_s`, so it can be
        # read instead of flickering.
        tick = index - index % self.update_step
        window = min(self.window_s, tick / self.fps)
        rate = self._recent(tick) / window * 3600 if window >= 2.0 else None

        # The last counted object, as of the last LAST_ITEM_SECONDS tick,
        # so the thumbnail doesn't flicker on a busy line.
        tick = index - index % self.last_item_step
        n_tick = bisect.bisect_right(self.frames, tick)
        _, last_track, last_class = (
            self.counted[n_tick - 1] if n_tick else (None, None, None)
        )
        return PanelState(
            total=n,
            recent=recent,
            rate_per_hour=rate,
            rows=rows,
            choice_counts=choice_counts,
            unverified=unverified,
            grade_counts=grade_counts,
            avg_mm=avg_mm,
            measure=self.measure,
            last_track=last_track,
            last_class=last_class,
        )

    def _recent(self, index: int) -> int:
        """Counts in the `window_s` seconds up to frame `index`."""
        since = index - self.window_s * self.fps
        return bisect.bisect_right(self.frames, index) - bisect.bisect_right(
            self.frames, since
        )

    def last_count_frame(self, index: int) -> int | None:
        n = bisect.bisect_right(self.frames, index)
        return self.frames[n - 1] if n else None


def grade_index(size_mm: float, grades: list[tuple[str, float | None]]) -> int:
    """Grades are (name, upper bound in mm), the last one unbounded."""
    for i, (_, upper) in enumerate(grades):
        if upper is None or size_mm < upper:
            return i
    return len(grades) - 1


# ---------------------------------------------------------
# Renderer
# ---------------------------------------------------------


class ShowcaseRenderer:
    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        title: str,
        accent_hex: str,
        class_colors: dict[int, str],
        choices: list[ChoiceStyle] | None,
        quality_title: str,
        position: str = "right",
        font_arg: str | None = None,
        show_panel: bool = True,
        show_tags: bool = True,
        line: tuple[float, float, float, float] | None = None,
        line_label: str | None = None,
        window_s: float = 10.0,
    ):
        self.w, self.h, self.fps = width, height, fps
        self.s = min(width, height) / 1080
        self.fonts = Fonts(self.s, font_arg)
        self.title = title.upper()
        self.accent = hex_to_rgb(accent_hex)
        self.class_colors = {k: hex_to_rgb(v) for k, v in class_colors.items()}
        self.choices = {c.value: c for c in choices} if choices else None
        self.quality_title = quality_title
        self.position = position
        self.show_panel = show_panel
        self.show_tags = show_tags
        self.line = line
        self.line_label = line_label
        self.window_s = window_s
        self._masks: dict[tuple, np.ndarray] = {}
        # Panel columns (its height depends on what it shows).
        pw, margin = self.px(560), self.px(32)
        x0 = margin if position == "left" else width - margin - pw
        self.panel_x = (x0, x0 + pw)
        # Object chips are hidden over this band: the panel and the margin
        # between it and the frame edge.
        self.panel_band = (0, x0 + pw) if position == "left" else (x0, width)

    def px(self, value: float) -> int:
        return max(1, round(value * self.s))

    # -- primitives ------------------------------------------------

    @staticmethod
    def _rounded_rect(img, x1, y1, x2, y2, r, color):
        x1, y1, x2, y2, r = (int(round(v)) for v in (x1, y1, x2, y2, r))
        r = max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r),
                       (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)

    @staticmethod
    def _rounded_outline(img, x1, y1, x2, y2, r, color, thickness):
        x1, y1, x2, y2, r = (int(round(v)) for v in (x1, y1, x2, y2, r))
        for a, b in (((x1 + r, y1), (x2 - r, y1)), ((x1 + r, y2), (x2 - r, y2)),
                     ((x1, y1 + r), (x1, y2 - r)), ((x2, y1 + r), (x2, y2 - r))):
            cv2.line(img, a, b, color, thickness, cv2.LINE_AA)
        for (cx, cy), start in (((x1 + r, y1 + r), 180), ((x2 - r, y1 + r), 270),
                                ((x2 - r, y2 - r), 0), ((x1 + r, y2 - r), 90)):
            cv2.ellipse(img, (cx, cy), (r, r), 0, start, start + 90, color,
                        thickness, cv2.LINE_AA)

    def _paste_rounded(self, img, x, y, image, radius):
        h, w = image.shape[:2]
        x, y = int(round(x)), int(round(y))
        if x < 0 or y < 0 or x + w > self.w or y + h > self.h:
            return
        key = (w, h, radius)
        mask = self._masks.get(key)
        if mask is None:
            mask = np.zeros((h, w), np.uint8)
            self._rounded_rect(mask, 0, 0, w - 1, h - 1, radius, 255)
            mask = (mask.astype(np.float32) / 255)[:, :, None]
            self._masks[key] = mask
        region = img[y : y + h, x : x + w]
        region[:] = (image * mask + region * (1 - mask)).astype(np.uint8)

    @property
    def thumb_size(self) -> int:
        return self.px(THUMB_SIZE)

    def thumbnail(self, crop: np.ndarray) -> np.ndarray:
        """A crop as the square "last item" thumbnail (center cut)."""
        h, w = crop.shape[:2]
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        square = crop[y0 : y0 + side, x0 : x0 + side]
        size = self.thumb_size
        interpolation = cv2.INTER_AREA if side > size else cv2.INTER_CUBIC
        return cv2.resize(square, (size, size), interpolation=interpolation)

    def _brackets(self, img, x1, y1, x2, y2, color, thickness):
        bw, bh = x2 - x1, y2 - y1
        arm = int(np.clip(min(bw, bh) * 0.24, self.px(8), self.px(30)))
        c = bgr(color)
        for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                                 (x2, y2, -1, -1), (x1, y2, 1, -1)):
            cx, cy = int(round(cx)), int(round(cy))
            pts = np.array(
                [[cx, cy + dy * arm], [cx, cy], [cx + dx * arm, cy]], np.int32
            )
            cv2.polylines(img, [pts], False, c, thickness, cv2.LINE_AA)

    def _dashed_line(self, img, p1, p2, color, thickness):
        p1, p2 = np.array(p1, float), np.array(p2, float)
        length = float(np.linalg.norm(p2 - p1))
        if length < 1:
            return
        direction = (p2 - p1) / length
        dash, gap = self.px(16), self.px(11)
        pos = 0.0
        while pos < length:
            a = p1 + direction * pos
            b = p1 + direction * min(pos + dash, length)
            cv2.line(img, tuple(map(int, a)), tuple(map(int, b)),
                     bgr(color), thickness, cv2.LINE_AA)
            pos += dash + gap

    # -- frame -----------------------------------------------------

    def draw(
        self,
        frame: np.ndarray,
        index: int,
        objects: list[tuple[np.ndarray, int, int]],
        tracks: dict[int, TrackInfo],
        panel: PanelState | None,
        last_count_frame: int | None,
        thumbnails: dict[int, np.ndarray] | None = None,
    ) -> np.ndarray:
        """`objects` are (xyxy, class id, track id) of the boxes to draw;
        `thumbnails` ({track id: square BGR image, see `thumbnail`}) feed
        the "last item" next to the count."""
        out = frame.copy()
        self._tracks = tracks
        self._thumbnails = thumbnails or {}

        # Objects and the count line first, then the panel over them.
        self._begin(out)
        self._draw_line(index, last_count_frame)
        self._draw_objects(objects, tracks)
        out = self._compose(out)

        if self.show_panel and panel is not None:
            self._begin(out)
            roi = self._draw_panel(panel)
            out = self._compose(out, roi, PANEL_ALPHA)
        return out

    def _begin(self, out: np.ndarray) -> None:
        self._layer = out.copy()  # dark translucent backgrounds
        self._texts: list[tuple] = []  # (x, y, text, font, rgb, anchor, spacing)
        # Drawn opaque after the translucent layer is blended in, so the
        # blend doesn't dim them.
        self._lines_todo: list[tuple] = []
        self._brackets_todo: list[tuple] = []
        self._solid: list[tuple] = []

    def _compose(
        self, out: np.ndarray, roi=None, alpha: float = CHIP_ALPHA
    ) -> np.ndarray:
        x0, y0, x1, y1 = roi or (0, 0, self.w, self.h)
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(self.w, int(np.ceil(x1))), min(self.h, int(np.ceil(y1)))
        if x1 <= x0 or y1 <= y0:
            return out
        region = out[y0:y1, x0:x1]
        cv2.addWeighted(
            self._layer[y0:y1, x0:x1], alpha, region, 1 - alpha, 0, region
        )
        for args in self._lines_todo:
            self._dashed_line(out, *args)
        for args in self._brackets_todo:
            self._brackets(out, *args)
        for kind, args in self._solid:
            if kind == "rect":
                self._rounded_rect(out, *args)
            elif kind == "outline":
                self._rounded_outline(out, *args)
            elif kind == "image":
                self._paste_rounded(out, *args)
            elif kind == "arc":
                center, r, start, end, color, thickness = args
                cv2.ellipse(out, center, (r, r), 0, start, end, color,
                            thickness, cv2.LINE_AA)
            else:
                cv2.line(out, *args, cv2.LINE_AA)
        if not self._texts:
            return out

        image = Image.fromarray(cv2.cvtColor(out[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        for x, y, text, font, color, anchor, spacing in self._texts:
            x, y = x - x0, y - y0
            if spacing:
                for ch in text:
                    draw.text((x, y), ch, font=font, fill=color, anchor=anchor)
                    x += font.getlength(ch) + spacing
            else:
                draw.text((x, y), text, font=font, fill=color, anchor=anchor)
        out[y0:y1, x0:x1] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        return out

    def _text(self, x, y, text, font, color, anchor="ls", spacing=0):
        self._texts.append((x, y, text, font, color, anchor, spacing))

    def _solid_rect(self, *args):
        self._solid.append(("rect", args))

    def _solid_line(self, *args):
        self._solid.append(("line", args))

    def _solid_arc(self, cx, cy, r, start, end, color, thickness):
        self._solid.append(
            ("arc", ((int(cx), int(cy)), int(r), start, end, bgr(color), thickness))
        )

    def _solid_circle(self, x, y, r, color):
        self._solid.append(("rect", (x - r, y - r, x + r, y + r, r, bgr(color))))

    def _chip(self, x, y, text, color, font=None, bg=PANEL_BG, solid=False):
        """A pill with `text`; (x, y) is its top-left. Returns its size.

        Translucent unless `solid` (needed inside the panel, whose
        background is painted on the translucent layer last).
        """
        font = font or self.fonts.chip
        pad_x = self.px(12)
        h = round(font.size * 1.75)
        w = font.getlength(text) + 2 * pad_x
        if solid:
            self._solid_rect(x, y, x + w, y + h, h / 2, bgr(bg))
        else:
            self._rounded_rect(self._layer, x, y, x + w, y + h, h / 2, bgr(bg))
        self._text(x + pad_x, y + h / 2, text, font, color, anchor="lm")
        return w, h

    # -- count line ------------------------------------------------

    def _draw_line(self, index, last_count_frame):
        if self.line is None:
            return
        x1, y1, x2, y2 = self.line
        pulse = 0.0
        if last_count_frame is not None:
            age = (index - last_count_frame) / self.fps
            if 0 <= age < PULSE_SECONDS:
                pulse = (1 - age / PULSE_SECONDS) ** 2
        color = lerp(LINE_IDLE, self.accent, pulse)
        thickness = self.px(2 + 3 * pulse)
        self._lines_todo.append(((x1, y1), (x2, y2), color, thickness))
        if self.line_label:
            # Near the start of the line, nudged inside the frame.
            cx = int(np.clip(x1 + self.px(10), 0, self.w - self.px(160)))
            cy = int(np.clip(y1 + self.px(16), 0, self.h - self.px(40)))
            self._chip(cx, cy, self.line_label.upper(), MUTED)

    # -- objects ---------------------------------------------------

    def _object_color(self, info: TrackInfo | None, class_id: int):
        if self.choices is not None:
            if info is None or info.verdict not in self.choices:
                return UNVERIFIED, None
            choice = self.choices[info.verdict]
            return choice.color, choice
        return self.class_colors.get(class_id, WHITE), None

    def _draw_objects(self, objects, tracks):
        thickness = self.px(2.5)
        offset = self.px(4)
        for xyxy, class_id, track_id in objects:
            info = tracks.get(track_id)
            color, choice = self._object_color(info, class_id)
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            x1, y1, x2, y2 = x1 - offset, y1 - offset, x2 + offset, y2 + offset
            self._brackets_todo.append((x1, y1, x2, y2, color, thickness))

            if not self.show_tags:
                continue
            parts = [f"#{track_id}"]
            flagged = choice is not None and (choice.flag or not choice.good)
            if flagged:
                parts.append(choice.value)
            if info is not None and info.size_mm:
                parts.append(f"{info.size_mm:.0f} mm")
            text = "  ·  ".join(parts)
            chip_h = round(self.fonts.tag.size * 1.75)
            tx = int(max(0, x1))
            chip_w = self.fonts.tag.getlength(text) + self.px(24)
            band = self.panel_band
            if self.show_panel and tx + chip_w > band[0] and tx < band[1]:
                continue
            ty = int(y1 - chip_h - self.px(6))
            if ty < 0:  # no room above: inside the top of the box
                ty = int(y1 + self.px(6))
            self._chip(tx, ty, text,
                       color if flagged else WHITE, font=self.fonts.tag)

    # -- panel -----------------------------------------------------

    def _draw_panel(self, st: PanelState):
        f, px = self.fonts, self.px
        x0, x1 = self.panel_x
        pw, pad, y0 = x1 - x0, px(40), px(32)
        left, right = x0 + pad, x0 + pw - pad
        y = y0 + pad

        def separator(before=26, after=30):
            nonlocal y
            y += px(before)
            self._solid_line((left, y), (right, y), bgr(SEPARATOR), px(1))
            y += px(after)

        def caps(x, y, text, color=MUTED, anchor="ls"):
            self._text(x, y, text.upper(), f.caps, color, anchor=anchor, spacing=px(1.8))

        # Left: title, big number, recent chip. Right: the last item.
        thumb = self.thumb_size
        col_x = right - thumb
        top = y
        caps(left, y + px(16), self.title)
        y += px(16) + px(140)
        total = fmt_int(st.total)
        big = f.big_fitting(total, col_x - left - px(24))
        self._text(left - px(4), y, total, big, WHITE)
        y += px(24)
        chip = f"+{fmt_int(st.recent)} in last {self.window_s:g} s"
        _, ch = self._chip(left, y, chip, GOOD, bg=GOOD_DARK, solid=True)
        y = max(
            y + ch, self._draw_last_item(col_x, top, st.last_track, st.last_class)
        )

        # Stat tiles: throughput and average size, each with an icon
        tiles = [(
            "Throughput",
            "—" if st.rate_per_hour is None else fmt_compact(st.rate_per_hour),
            "",
            "per hour",
            self._icon_bars,
        )]
        if st.measure:
            tiles.append(
                ("Avg size",
                 "—" if st.avg_mm is None else f"{st.avg_mm:.0f}",
                 "" if st.avg_mm is None else "mm",
                 "box diagonal",
                 self._icon_diameter)
            )
        separator()
        tile_w = (right - left) / len(tiles)
        top = y
        inset = px(28)
        for i, (label, value, unit, sub, icon) in enumerate(tiles):
            tile_l = left + i * tile_w
            tile_r = tile_l + tile_w
            tx = tile_l + (inset if i else 0)
            if i:
                self._solid_line((int(tile_l), int(top)),
                                 (int(tile_l), int(top + px(104))),
                                 bgr(SEPARATOR), px(1))
            caps(tx, top + px(16), label)
            self._text(tx, top + px(72), value, f.stat, WHITE)
            if unit:
                self._text(tx + f.stat.getlength(value) + px(6), top + px(72),
                           unit, f.unit, MUTED)
            self._text(tx, top + px(102), sub, f.small, MUTED)
            icon_r = tile_r - (inset if i < len(tiles) - 1 else 0)
            icon(icon_r, top + px(72))
        y = top + px(104)

        # Per-row counts (only when there is more than one row)
        if len(st.rows) > 1:
            separator()
            for i, (label, value) in enumerate(st.rows):
                if i:
                    y += px(36)
                self._text(left, y + px(16), label, f.label, MUTED)
                self._text(right, y + px(16), fmt_int(value), f.value, WHITE, anchor="rs")
            y += px(16)

        # Classification split: donut + legend
        if st.choice_counts:
            separator()
            caps(left, y + px(16), self.quality_title)
            y += px(40)

            verified = sum(c for _, c in st.choice_counts)
            good = sum(c for ch_, c in st.choice_counts if ch_.good)
            total = verified + st.unverified
            diameter, ring = px(150), px(18)
            r = (diameter - ring) / 2
            cx, cy = left + diameter / 2, y + diameter / 2
            segments = [(ch_.color, c) for ch_, c in st.choice_counts]
            segments.append((BAR_DIM, st.unverified))
            segments = [(color, c) for color, c in segments if c]
            if not total:
                self._solid_arc(cx, cy, r, 0, 360, SEPARATOR, ring)
            else:
                gap = 3 if len(segments) > 1 else 0  # degrees
                angle = -90.0
                for color, count in segments:
                    sweep = 360 * count / total
                    self._solid_arc(cx, cy, r, angle + gap / 2,
                                    angle + sweep - gap / 2, color, ring)
                    angle += sweep
            headline = f"{100 * good / verified:.0f}%" if verified else "—"
            good_names = "/".join(ch_.value for ch_, _ in st.choice_counts if ch_.good)
            self._text(cx, cy + px(8), headline, f.donut, WHITE, anchor="ms")
            self._text(cx, cy + px(32), good_names, f.small, MUTED, anchor="ms")

            legend = [(c.value, c.color, WHITE, n) for c, n in st.choice_counts]
            if st.unverified:
                legend.append(("unverified", BAR_DIM, MUTED, st.unverified))
            row_h = px(38)
            lx = left + diameter + px(36)
            ly = cy - row_h * (len(legend) - 1) / 2 + px(8)
            for name, color, text_color, count in legend:
                dot = px(7)
                self._solid_circle(lx + dot, ly - px(8), dot, color)
                self._text(lx + px(28), ly, name, f.label, text_color)
                pct = f"{100 * count / total:.0f}%" if total else ""
                self._text(right, ly, pct, f.label, MUTED, anchor="rs")
                self._text(right - px(66), ly, fmt_int(count), f.value,
                           color if text_color == WHITE else MUTED, anchor="rs")
                ly += row_h
            y += diameter

        # Size grades
        if st.grade_counts:
            separator()
            caps(left, y + px(16), "Size grade")
            y += px(40)
            n = len(st.grade_counts)
            gap = px(14)
            slot = ((right - left) - gap * (n - 1)) / n
            bw = min(slot, px(52))
            max_h = px(110)
            peak = max(c for _, c in st.grade_counts)
            top = y + px(24)
            for i, (name, count) in enumerate(st.grade_counts):
                bx = left + i * (slot + gap) + (slot - bw) / 2
                bh = max_h * count / peak if peak else 0
                base = top + max_h
                is_peak = bool(peak) and count == peak
                color = GOOD if is_peak else BAR_DIM
                if bh >= 1:
                    self._solid_rect(bx, base - bh, bx + bw, base, min(px(6), bh / 2), bgr(color))
                else:
                    self._solid_rect(bx, base - px(3), bx + bw, base, px(1), bgr(SEPARATOR))
                self._text(bx + bw / 2, base - bh - px(8), fmt_int(count), f.value,
                           WHITE if is_peak else MUTED, anchor="ms")
                self._text(bx + bw / 2, base + px(30), name, f.label, MUTED, anchor="ms")
            y = top + max_h + px(30)

        y += pad
        self._panel_background(x0, y0, x0 + pw, y, px(22))
        return x0, y0, x0 + pw, y

    def _draw_last_item(
        self, x, top, track_id: int | None, class_id: int | None
    ) -> float:
        """The "last item" column of the panel header, `THUMB_SIZE` wide,
        starting at (x, top). Returns where it ends."""
        f, px = self.fonts, self.px
        size, radius = self.thumb_size, px(14)
        self._text(x, top + px(16), "LAST ITEM", f.caps, MUTED, spacing=px(1.8))
        y = top + px(34)

        image = self._thumbnails.get(track_id) if track_id is not None else None
        info = self._tracks.get(track_id) if track_id is not None else None
        if image is None:
            self._solid.append(
                ("outline", (x, y, x + size, y + size, radius, bgr(SEPARATOR), px(2)))
            )
            self._text(x + size / 2, y + size / 2, "—", f.donut, MUTED, anchor="mm")
            return y + size

        color, choice = self._object_color(info, class_id)
        self._solid.append(("image", (x, y, image, radius)))
        self._solid.append(
            ("outline", (x, y, x + size - 1, y + size - 1, radius, bgr(color), px(2.5)))
        )
        y += size + px(30)
        self._text(x, y, f"#{track_id}", f.value, WHITE)
        if choice is not None:
            self._text(x + size, y, choice.value, f.label, color, anchor="rs")
        elif self.choices is not None:
            self._text(x + size, y, "unverified", f.small, MUTED, anchor="rs")
        if info is not None and info.size_mm:
            y += px(26)
            self._text(x, y, f"{info.size_mm:.0f} mm", f.small, MUTED)
        return y

    def _panel_background(self, x0, y0, x1, y1, radius):
        """Rounded card: vertical gradient (translucent) and a thin border."""
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(min(y1, self.h))
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            return
        mask = np.zeros((h, w), np.uint8)
        self._rounded_rect(mask, 0, 0, w - 1, h - 1, radius, 255)
        t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
        top = np.array(bgr(PANEL_TOP), np.float32)
        bottom = np.array(bgr(PANEL_BOTTOM), np.float32)
        gradient = np.broadcast_to(top * (1 - t) + bottom * t, (h, w, 3))
        region = self._layer[y0:y1, x0:x1]
        inside = mask > 0
        region[inside] = gradient[inside].astype(np.uint8)
        self._solid.append(
            ("outline", (x0, y0, x1 - 1, y1 - 1, radius, bgr(PANEL_BORDER),
                         self.px(1.5)))
        )

    # -- icons (teal, drawn right-aligned at x=`right`, sitting on `base`)

    def _icon_bars(self, right, base):
        """Three rising bars: throughput."""
        px = self.px
        w, gap = px(6), px(4)
        for i, h in enumerate((px(10), px(18), px(26))):
            x = right - (3 - i) * w - (2 - i) * gap
            self._solid_rect(x, base - h, x + w, base, px(2), bgr(GOOD))

    def _icon_diameter(self, right, base):
        """A circle with its diameter marked: measured size."""
        px = self.px
        r = px(14)
        cx, cy = right - r, base - r
        thickness = px(2)
        self._solid_arc(cx, cy, r, 0, 360, GOOD, thickness)
        a, b = (int(cx - r + px(5)), int(cy)), (int(cx + r - px(5)), int(cy))
        self._solid_line(a, b, bgr(GOOD), thickness)
        for x, d in ((a[0], 1), (b[0], -1)):  # arrow heads
            self._solid_line((x, a[1]), (x + d * px(5), a[1] - px(4)), bgr(GOOD), thickness)
            self._solid_line((x, a[1]), (x + d * px(5), a[1] + px(4)), bgr(GOOD), thickness)

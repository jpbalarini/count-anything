"""The counter panel drawn on the corner of the output video."""

from __future__ import annotations

import cv2
import numpy as np


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

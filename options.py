"""Helpers that turn the --classes / --colors / --hud-rows values into
class ids, colors and HUD rows, plus the built-in color defaults.
"""

from __future__ import annotations

import supervision as sv

# ---------------------------------------------------------
# Defaults
#
# RF-DETR uses the sparse COCO ids (NOT the same as
# Ultralytics YOLO):  3 = car, 4 = motorcycle, 6 = bus,
# 8 = truck.
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Helpers for --classes / --colors / --hud-rows
# ---------------------------------------------------------


def normalize_name(name: str) -> str:
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
            names.setdefault(normalize_name(token), token)
    return dict(enumerate(names.values()))


def resolve_class(token: str, coco: dict[int, str]) -> int:
    """Turn '3' or 'car' (or 'traffic_light') into a class id."""
    token = token.strip()
    wanted = normalize_name(token)
    for class_id, name in coco.items():
        if normalize_name(name) == wanted:
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


def is_hidden_label(text: str | None) -> bool:
    """True when --in-text/--out-text asks to hide that label.

    `None` (flag not given) means "use the default label", so only an
    explicit empty string or 'none' hides it.
    """
    return text is not None and text.strip().lower() in ("", "none")


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

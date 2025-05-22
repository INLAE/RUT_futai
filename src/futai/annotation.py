"""
Создание преднастроенных аннотаторов Supervision.
"""

import supervision as sv
from typing import Any, Dict

from .constants import (
    COLOR_PALETTE_HEX,
    ELLIPSE_THICKNESS,
    TRIANGLE_BASE,
    TRIANGLE_HEIGHT,
    TRIANGLE_OUTLINE_THICKNESS,
    TEXT_COLOR_HEX
)

def build_annotators() -> Dict[str, Any]:
    """
    Возвращает словарь из 3-х Annotator:
      - ellipse     (игроки)
      - label       (текст с tracker_id)
      - triangle     (мяч)
    """
    palette = sv.ColorPalette.from_hex(COLOR_PALETTE_HEX)

    ellipse = sv.EllipseAnnotator(
        color=palette,
        thickness=ELLIPSE_THICKNESS
    )

    label = sv.LabelAnnotator(
        color=palette,
        text_color=sv.Color.from_hex(TEXT_COLOR_HEX),
        text_position=sv.Position.BOTTOM_CENTER
    )

    triangle = sv.TriangleAnnotator(
        color=sv.Color.from_hex(TEXT_COLOR_HEX),
        base=TRIANGLE_BASE,
        height=TRIANGLE_HEIGHT,
        outline_thickness=TRIANGLE_OUTLINE_THICKNESS
    )

    return {
        'ellipse': ellipse,
        'label': label,
        'triangle': triangle
    }
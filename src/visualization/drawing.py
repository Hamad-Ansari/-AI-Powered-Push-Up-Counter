"""Low-level OpenCV drawing primitives for the AI overlay.

All helpers are pure drawing functions: they take a frame, mutate a copy or draw
in place, and never touch application state. Everything scales with the frame
width so the HUD looks the same at 640p and 1080p.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

Color = Tuple[int, int, int]
Point = Tuple[int, int]

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


def scale_factor(frame: np.ndarray, base_width: int = 1280) -> float:
    """UI scale so the HUD keeps a constant relative size."""
    return max(0.45, frame.shape[1] / float(base_width))


def panel(
    frame: np.ndarray,
    top_left: Point,
    size: Tuple[int, int],
    color: Color,
    *,
    alpha: float = 0.62,
    border: Optional[Color] = None,
    radius: int = 12,
    thickness: int = 1,
) -> None:
    """Draw a translucent rounded rectangle.

    Args:
        frame: BGR frame to draw on (modified in place).
        top_left: ``(x, y)`` of the panel's top-left corner.
        size: ``(width, height)`` in pixels.
        color: Fill colour (BGR).
        alpha: Fill opacity in ``[0, 1]``.
        border: Optional border colour.
        radius: Corner radius in pixels.
        thickness: Border thickness.
    """
    height, width = frame.shape[:2]
    x1 = max(0, int(top_left[0]))
    y1 = max(0, int(top_left[1]))
    x2 = min(width, int(top_left[0] + size[0]))
    y2 = min(height, int(top_left[1] + size[1]))
    if x2 <= x1 or y2 <= y1:
        return

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=cv2.FILLED)
    cv2.addWeighted(overlay, float(alpha), frame, 1.0 - float(alpha), 0, frame)

    radius = max(0, min(radius, min(x2 - x1, y2 - y1) // 2))
    if radius and border is not None:
        _rounded_rect_outline(frame, (x1, y1), (x2, y2), radius, border, thickness)


def _rounded_rect_outline(
    frame: np.ndarray,
    top_left: Point,
    bottom_right: Point,
    radius: int,
    color: Color,
    thickness: int,
) -> None:
    x1, y1 = top_left
    x2, y2 = bottom_right
    r = radius
    cv2.line(frame, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
    cv2.line(frame, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
    cv2.ellipse(frame, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(frame, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(frame, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(frame, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)


def text_size(text: str, font_scale: float, thickness: int) -> Tuple[int, int]:
    """Pixel ``(width, height)`` of ``text`` at the given scale."""
    (width, height), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    return width, height + baseline


_ASCII_MAP = {
    "°": " deg",
    "·": "-",
    "—": "-",
    "–": "-",
    "→": "->",
    "≤": "<=",
    "≥": ">=",
    "×": "x",
}


def sanitize_text(text: str) -> str:
    """Make ``text`` renderable by OpenCV's built-in (ASCII-only) Hershey font.

    Common symbols are transliterated and everything else non-ASCII (emoji etc.)
    is dropped, so the overlay never shows ``?`` tofu boxes.
    """
    for source, replacement in _ASCII_MAP.items():
        text = text.replace(source, replacement)
    ascii_only = text.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_only.split())


def draw_text(
    frame: np.ndarray,
    text: str,
    origin: Point,
    *,
    color: Color = (255, 255, 255),
    font_scale: float = 0.6,
    thickness: int = 1,
    shadow: bool = True,
    shadow_color: Color = (0, 0, 0),
    align: str = "left",
) -> Point:
    """Draw text with an optional drop shadow and alignment.

    Alignment is origin-based: ``left`` anchors the text's left edge at
    ``origin``, ``center`` centres it on ``origin[0]`` and ``right`` puts its right
    edge at ``origin[0]``.

    Args:
        frame: Target frame.
        text: Text to render (sanitised to ASCII first).
        origin: Baseline anchor point.
        color: Text colour.
        font_scale: OpenCV font scale.
        thickness: Stroke thickness.
        shadow: Draw a 1-px dark shadow for readability.
        shadow_color: Shadow colour.
        align: ``"left"``, ``"center"`` or ``"right"``.

    Returns:
        The ``(x, y)`` where drawing started.
    """
    text = sanitize_text(text)
    width, height = text_size(text, font_scale, thickness)
    x, y = int(origin[0]), int(origin[1])
    if align == "center":
        x = int(x - width / 2)
    elif align == "right":
        x = int(x - width)

    if shadow:
        cv2.putText(frame, text, (x + 1, y + 1), FONT, font_scale, shadow_color, thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), FONT, font_scale, color, thickness, cv2.LINE_AA)
    return x, y


def progress_bar(
    frame: np.ndarray,
    top_left: Point,
    size: Tuple[int, int],
    ratio: float,
    color: Color,
    *,
    background: Color = (45, 45, 45),
    vertical: bool = False,
    border: Optional[Color] = None,
) -> None:
    """Draw a horizontal or vertical progress bar."""
    x, y = int(top_left[0]), int(top_left[1])
    width, height = int(size[0]), int(size[1])
    ratio = max(0.0, min(1.0, float(ratio)))

    cv2.rectangle(frame, (x, y), (x + width, y + height), background, thickness=cv2.FILLED)
    if vertical:
        filled = int(height * ratio)
        cv2.rectangle(frame, (x, y + height - filled), (x + width, y + height), color, thickness=cv2.FILLED)
    else:
        filled = int(width * ratio)
        cv2.rectangle(frame, (x, y), (x + filled, y + height), color, thickness=cv2.FILLED)
    if border is not None:
        cv2.rectangle(frame, (x, y), (x + width, y + height), border, thickness=1)


def status_chip(
    frame: np.ndarray,
    top_left: Point,
    label: str,
    value: str,
    color: Color,
    *,
    scale: float = 1.0,
    width: Optional[int] = None,
) -> Tuple[int, int]:
    """Draw a compact ``LABEL  value`` indicator row.

    Returns:
        ``(width, height)`` of the rendered chip.
    """
    font_scale = 0.45 * scale
    label_width, label_height = text_size(label.upper(), font_scale, 1)
    value_width, _ = text_size(value, font_scale, 1)
    dot_radius = int(4 * scale)
    padding = int(8 * scale)
    gap = int(10 * scale)
    chip_width = width or (dot_radius * 2 + padding * 3 + label_width + value_width + gap)
    chip_height = int(22 * scale)

    x, y = int(top_left[0]), int(top_left[1])
    panel(frame, (x, y), (chip_width, chip_height), (18, 18, 18), alpha=0.55, radius=int(8 * scale))
    center_y = y + chip_height // 2
    cv2.circle(frame, (x + padding + dot_radius, center_y), dot_radius, color, thickness=cv2.FILLED, lineType=cv2.LINE_AA)
    draw_text(
        frame,
        label.upper(),
        (x + padding * 2 + dot_radius * 2, center_y + int(label_height / 2) - 2),
        color=(200, 200, 200),
        font_scale=font_scale,
    )
    draw_text(
        frame,
        value,
        (x + chip_width - padding, center_y + int(label_height / 2) - 2),
        color=color,
        font_scale=font_scale,
        align="right",
    )
    return chip_width, chip_height


def angle_arc(
    frame: np.ndarray,
    vertex: Sequence[float],
    point_a: Sequence[float],
    point_c: Sequence[float],
    angle_degrees: float,
    color: Color,
    *,
    radius: int = 34,
    label: Optional[str] = None,
    font_scale: float = 0.5,
) -> None:
    """Draw the angle arc at ``vertex`` plus its label.

    Args:
        frame: Target frame.
        vertex: Joint the angle is measured at.
        point_a: First arm end point.
        point_c: Second arm end point.
        angle_degrees: Angle value to display.
        color: Arc colour.
        radius: Arc radius in pixels.
        label: Optional override of the displayed text.
        font_scale: Label font scale.
    """
    import math

    vx, vy = float(vertex[0]), float(vertex[1])
    start = math.degrees(math.atan2(float(point_a[1]) - vy, float(point_a[0]) - vx))
    end = math.degrees(math.atan2(float(point_c[1]) - vy, float(point_c[0]) - vx))

    # OpenCV wants a counter-clockwise sweep from `start`.
    sweep = (end - start) % 360.0
    if sweep > 180.0:
        start, sweep = end, 360.0 - sweep

    cv2.ellipse(
        frame,
        (int(vx), int(vy)),
        (radius, radius),
        0,
        start,
        start + sweep,
        color,
        2,
        cv2.LINE_AA,
    )
    text = label if label is not None else f"{angle_degrees:.0f}"
    mid = math.radians(start + sweep / 2.0)
    label_point = (int(vx + (radius + 12) * math.cos(mid)), int(vy + (radius + 12) * math.sin(mid)))
    draw_text(frame, text, label_point, color=color, font_scale=font_scale, thickness=1)


def badge(
    frame: np.ndarray,
    top_left: Point,
    text: str,
    color: Color,
    *,
    scale: float = 1.0,
    font_scale: Optional[float] = None,
    padding: int = 10,
) -> Tuple[int, int]:
    """Draw a filled pill with centred text. Returns its ``(width, height)``."""
    resolved_font = font_scale if font_scale is not None else 0.6 * scale
    thickness = max(1, int(round(resolved_font * 2)))
    text_width, text_height = text_size(text, resolved_font, thickness)
    width = text_width + padding * 2
    height = text_height + padding
    x, y = int(top_left[0]), int(top_left[1])
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, thickness=cv2.FILLED)
    draw_text(
        frame,
        text,
        (x + width // 2, y + padding + text_height - 4),
        color=(15, 15, 15),
        font_scale=resolved_font,
        thickness=thickness,
        align="center",
        shadow=False,
    )
    return width, height


def resize_to_width(frame: np.ndarray, width: int) -> np.ndarray:
    """Resize a frame to ``width`` keeping the aspect ratio (no-op when equal)."""
    if frame.shape[1] == width:
        return frame
    height = int(round(frame.shape[0] * width / frame.shape[1]))
    return cv2.resize(frame, (width, max(1, height)), interpolation=cv2.INTER_AREA)


def dim_edges(frame: np.ndarray, strength: float = 0.25) -> np.ndarray:
    """Darken the frame borders so white HUD text stays readable."""
    height, width = frame.shape[:2]
    vignette = np.ones((height, width), dtype=np.float32)
    border = int(min(height, width) * 0.18)
    vignette[:border, :] = 1.0 - strength
    vignette[-border:, :] = 1.0 - strength
    vignette[:, :border] = 1.0 - strength
    vignette[:, -border:] = 1.0 - strength
    vignette = cv2.GaussianBlur(vignette, (0, 0), sigmaX=border / 2)
    return cv2.multiply(frame, cv2.merge([vignette] * 3))


__all__ = [
    "FONT",
    "FONT_BOLD",
    "scale_factor",
    "panel",
    "text_size",
    "draw_text",
    "progress_bar",
    "status_chip",
    "angle_arc",
    "badge",
    "resize_to_width",
    "dim_edges",
]

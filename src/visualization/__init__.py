"""Visualization layer: skeleton rendering and the AI fitness HUD overlay."""

from src.visualization.colors import DEFAULT_PALETTE, Color, Palette, mix, with_alpha
from src.visualization.drawing import (
    angle_arc,
    badge,
    dim_edges,
    draw_text,
    panel,
    progress_bar,
    resize_to_width,
    scale_factor,
    status_chip,
    text_size,
)
from src.visualization.overlay import HudData, OverlayRenderer, build_hud_data, issue_to_feedback
from src.visualization.skeleton_drawer import SkeletonDrawer

__all__ = [
    "DEFAULT_PALETTE",
    "Color",
    "Palette",
    "mix",
    "with_alpha",
    "angle_arc",
    "badge",
    "dim_edges",
    "draw_text",
    "panel",
    "progress_bar",
    "resize_to_width",
    "scale_factor",
    "status_chip",
    "text_size",
    "HudData",
    "OverlayRenderer",
    "build_hud_data",
    "issue_to_feedback",
    "SkeletonDrawer",
]

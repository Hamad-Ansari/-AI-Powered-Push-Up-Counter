"""Professional video overlay: counter, status bar, telemetry and feedback.

Layout (mirrors a commercial fitness AI product):

===========================  ==================  =============================
 6                           [ MOVING ]          DET 0.97   FPS 28.4
 Push-Up                                         17/17 kp   34 ms
===========================  ==================  =============================
 [DETECTION] [TRACKING]                                            FORM SCORE
 [FORM]      [VALID REP]                                             [====] 91
                                                                     Excellent
  Feedback toast
 -----------------------------------------------------------------------------
 Elbow 82°  |  Alignment 174°  |  Form 91%  |  FPS 28.5  |  Reps 6 (5 valid)
 -----------------------------------------------------------------------------
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from src.analysis.form_scorer import FormGrade
from src.analysis.movement_analyzer import ExerciseState
from src.analysis.posture_analyzer import PostureIssue, Severity
from config.settings import VisualizationConfig
from src.utils.helpers import clamp
from src.utils.logger import get_logger
from src.visualization.colors import Palette
from src.visualization.drawing import (
    badge,
    draw_text,
    panel,
    progress_bar,
    scale_factor,
    status_chip,
    text_size,
)

logger = get_logger(__name__)


@dataclass(slots=True)
class HudData:
    """Everything the overlay needs for one frame (plain values, no CV objects)."""

    rep_count: int = 0
    valid_reps: int = 0
    invalid_reps: int = 0
    state: ExerciseState = ExerciseState.UNKNOWN
    raw_state: ExerciseState = ExerciseState.UNKNOWN
    detection_confidence: float = 0.0
    visible_keypoints: int = 0
    total_keypoints: int = 17
    fps: float = 0.0
    inference_ms: float = 0.0
    form_score: float = 0.0
    form_grade: FormGrade = FormGrade.POOR
    elbow_angle: float = float("nan")
    body_alignment: float = float("nan")
    arm_diff: float = float("nan")
    feedback: List[str] = field(default_factory=list)
    last_rep_valid: Optional[bool] = None
    last_rep_reason: str = ""
    has_pose: bool = False
    multiple_people: bool = False
    message: str = ""
    session_label: str = ""


class OverlayRenderer:
    """Renders the AI fitness HUD on top of an annotated frame."""

    def __init__(self, config: VisualizationConfig, palette: Optional[Palette] = None) -> None:
        self.config = config
        self.palette = palette or Palette(config)

    def draw(self, frame: np.ndarray, data: HudData) -> np.ndarray:
        """Render the full HUD onto ``frame`` (in place) and return it.

        Args:
            frame: BGR frame already containing the skeleton.
            data: Values to display for this frame.

        Returns:
            The annotated frame.
        """
        if not self.config.hud.enabled:
            return frame

        scale = scale_factor(frame, self.config.hud.base_width)
        height, width = frame.shape[:2]
        margin = int(16 * scale)

        self._draw_rep_counter(frame, data, margin, scale)
        self._draw_state_bar(frame, data, width, margin, scale)
        self._draw_status_panel(frame, data, width, margin, scale)
        self._draw_left_indicators(frame, data, margin, scale, top_offset=int(140 * scale))
        self._draw_form_bar(frame, data, width, margin, scale)
        self._draw_feedback(frame, data, margin, height, scale)
        if self.config.hud.show_bottom_telemetry:
            self._draw_telemetry(frame, data, width, height, scale)
        return frame

    # -------------------------------------------------------------- top left
    def _draw_rep_counter(self, frame: np.ndarray, data: HudData, margin: int, scale: float) -> None:
        panel_width = int(170 * scale)
        panel_height = int(112 * scale)
        panel(frame, (margin, margin), (panel_width, panel_height), self.palette.panel, alpha=self.config.hud.opacity, border=self.palette.panel_border, radius=int(14 * scale))

        number = str(data.rep_count)
        number_scale = 2.4 * scale
        number_thickness = max(2, int(round(3 * scale)))
        center_x = margin + panel_width // 2
        draw_text(
            frame,
            number,
            (center_x, margin + int(58 * scale)),
            color=self.palette.text,
            font_scale=number_scale,
            thickness=number_thickness,
            align="center",
        )
        label_scale = 0.62 * scale
        draw_text(
            frame,
            "PUSH-UP",
            (center_x, margin + int(58 * scale) + int(30 * scale)),
            color=self.palette.accent,
            font_scale=label_scale,
            thickness=max(1, int(round(scale))),
            align="center",
        )
        sub = f"{data.valid_reps} valid · {data.invalid_reps} invalid"
        sub_scale = 0.42 * scale
        draw_text(
            frame,
            sub,
            (center_x, margin + panel_height - int(10 * scale)),
            color=self.palette.text_dim,
            font_scale=sub_scale,
            align="center",
        )

    # ------------------------------------------------------------ top centre
    def _draw_state_bar(self, frame: np.ndarray, data: HudData, width: int, margin: int, scale: float) -> None:
        label = data.state.label if data.state != ExerciseState.MOVING else "MOVING..."
        color = self.palette.state(data.state.status_key)
        badge_scale = 0.75 * scale
        padding = int(14 * scale)
        thickness = max(1, int(round(badge_scale * 2)))
        text_width, text_height = text_size(label, badge_scale, thickness)
        badge_width = text_width + padding * 2
        badge_height = text_height + padding
        x_center = max(margin, width // 2 - badge_width // 2)
        badge(frame, (x_center, margin), label, color, scale=scale, font_scale=badge_scale, padding=padding)
        if data.message:
            draw_text(
                frame,
                data.message,
                (width // 2, margin + badge_height + int(22 * scale)),
                color=self.palette.text,
                font_scale=0.5 * scale,
                align="center",
            )

    # ------------------------------------------------------------- top right
    def _draw_status_panel(self, frame: np.ndarray, data: HudData, width: int, margin: int, scale: float) -> None:
        panel_width = int(190 * scale)
        panel_height = int(104 * scale)
        left = width - margin - panel_width
        panel(frame, (left, margin), (panel_width, panel_height), self.palette.panel, alpha=self.config.hud.opacity, border=self.palette.panel_border, radius=int(14 * scale))

        font_scale = 0.48 * scale
        line_height = int(22 * scale)
        rows = (
            ("DETECTION", f"{data.detection_confidence:.2f}" if data.has_pose else "--", self.palette.good if data.has_pose else self.palette.neutral),
            ("KEYPOINTS", f"{data.visible_keypoints}/{data.total_keypoints}", self.palette.tracking),
            ("FPS", f"{data.fps:.1f}", self.palette.accent),
            ("LATENCY", f"{data.inference_ms:.0f} ms", self.palette.text_dim),
        )
        for index, (label, value, color) in enumerate(rows):
            y = margin + int(26 * scale) + index * line_height
            draw_text(frame, label, (left + int(14 * scale), y), color=self.palette.text_dim, font_scale=font_scale)
            draw_text(frame, value, (left + panel_width - int(14 * scale), y), color=color, font_scale=font_scale, align="right")
        if data.multiple_people:
            draw_text(
                frame,
                "Multiple people - tracking primary",
                (width - margin, margin + panel_height + int(18 * scale)),
                color=self.palette.warning,
                font_scale=0.44 * scale,
                align="right",
            )

    # ------------------------------------------------------------ left chips
    def _draw_left_indicators(
        self,
        frame: np.ndarray,
        data: HudData,
        margin: int,
        scale: float,
        top_offset: int,
    ) -> None:
        chips = (
            ("detection", "ON" if data.has_pose else "LOST", self.palette.good if data.has_pose else self.palette.bad),
            ("tracking", f"{data.visible_keypoints}/{data.total_keypoints}", self.palette.tracking),
            ("form", data.form_grade.value.title(), self.palette.grade(data.form_grade)),
            (
                "valid rep",
                "—" if data.last_rep_valid is None else ("YES" if data.last_rep_valid else "NO"),
                self.palette.neutral if data.last_rep_valid is None else (self.palette.good if data.last_rep_valid else self.palette.bad),
            ),
        )
        for index, (label, value, color) in enumerate(chips):
            top = top_offset + index * int(30 * scale)
            status_chip(frame, (margin, top), label, value, color, scale=scale)

    # ------------------------------------------------------------ right bar
    def _draw_form_bar(self, frame: np.ndarray, data: HudData, width: int, margin: int, scale: float) -> None:
        bar_width = int(26 * scale)
        bar_height = int(240 * scale)
        panel_width = int(120 * scale)
        left = width - margin - panel_width
        top = int(frame.shape[0] * 0.30)
        panel(frame, (left, top), (panel_width, bar_height + int(56 * scale)), self.palette.panel, alpha=self.config.hud.opacity, border=self.palette.panel_border, radius=int(14 * scale))

        draw_text(
            frame,
            "FORM",
            (left + panel_width // 2, top + int(24 * scale)),
            color=self.palette.text_dim,
            font_scale=0.46 * scale,
            align="center",
        )
        bar_left = left + (panel_width - bar_width) // 2
        progress_bar(
            frame,
            (bar_left, top + int(36 * scale)),
            (bar_width, bar_height),
            clamp(data.form_score / 100.0, 0.0, 1.0),
            self.palette.score(data.form_score),
            background=(40, 40, 40),
            vertical=True,
            border=self.palette.panel_border,
        )
        score_text = f"{data.form_score:.0f}"
        draw_text(
            frame,
            score_text,
            (left + panel_width // 2, top + int(36 * scale) + bar_height + int(26 * scale)),
            color=self.palette.score(data.form_score),
            font_scale=0.8 * scale,
            thickness=max(1, int(round(scale))),
            align="center",
        )
        draw_text(
            frame,
            data.form_grade.headline,
            (left + panel_width // 2, top + int(36 * scale) + bar_height + int(48 * scale)),
            color=self.palette.text_dim,
            font_scale=0.4 * scale,
            align="center",
        )

    # --------------------------------------------------------------- feedback
    def _draw_feedback(
        self,
        frame: np.ndarray,
        data: HudData,
        margin: int,
        height: int,
        scale: float,
    ) -> None:
        messages = [line for line in data.feedback if line][: self.config.hud.feedback_history]
        if not messages:
            return
        bottom_limit = height - int(78 * scale) if self.config.hud.show_bottom_telemetry else height - int(20 * scale)
        line_height = int(26 * scale)
        for index, message in enumerate(reversed(messages)):
            severity = _severity_from_message(message)
            color = self.palette.severity(severity)
            y = bottom_limit - index * line_height
            panel(
                frame,
                (margin, y - int(20 * scale)),
                (min(int(560 * scale), frame.shape[1] - 2 * margin), int(26 * scale)),
                self.palette.panel,
                alpha=max(0.35, self.config.hud.opacity - 0.15),
                radius=int(10 * scale),
            )
            draw_text(frame, message, (margin + int(12 * scale), y), color=color, font_scale=0.52 * scale)

    # -------------------------------------------------------------- telemetry
    def _draw_telemetry(self, frame: np.ndarray, data: HudData, width: int, height: int, scale: float) -> None:
        bar_height = int(46 * scale)
        top = height - bar_height - int(10 * scale)
        panel(frame, (int(10 * scale), top), (width - int(20 * scale), bar_height), self.palette.panel, alpha=self.config.hud.opacity, border=self.palette.panel_border, radius=int(12 * scale))

        segments = (
            ("Elbow", _fmt_deg(data.elbow_angle), self.palette.accent),
            ("Alignment", _fmt_deg(data.body_alignment), self.palette.good if _aligned(data.body_alignment) else self.palette.warning),
            ("Arms diff", _fmt_deg(data.arm_diff), self.palette.text),
            ("Form", f"{data.form_score:.0f}%", self.palette.score(data.form_score)),
            ("FPS", f"{data.fps:.1f}", self.palette.text),
            ("Reps", f"{data.rep_count}", self.palette.text),
        )
        usable = width - int(40 * scale)
        slot = usable / max(1, len(segments))
        for index, (label, value, color) in enumerate(segments):
            x = int(20 * scale) + int(index * slot)
            draw_text(frame, label.upper(), (x, top + int(19 * scale)), color=self.palette.text_dim, font_scale=0.4 * scale)
            draw_text(frame, value, (x, top + int(38 * scale)), color=color, font_scale=0.55 * scale, thickness=max(1, int(round(scale))))


def _fmt_deg(value: float) -> str:
    """Format an angle for the telemetry strip."""
    return f"{value:.0f}°" if math.isfinite(value) else "--"


def _aligned(angle: float) -> bool:
    return math.isfinite(angle) and abs(angle - 180.0) <= 15.0


def _severity_from_message(message: str) -> Severity:
    """Recover the severity from an emoji-prefixed feedback line."""
    if message.startswith(Severity.INFO.emoji):
        return Severity.INFO
    if message.startswith(Severity.WARNING.emoji):
        return Severity.WARNING
    if message.startswith(Severity.CRITICAL.emoji):
        return Severity.CRITICAL
    return Severity.INFO


def issue_to_feedback(issue: Optional[PostureIssue]) -> str:
    """Render a posture issue as a single feedback line."""
    if issue is None:
        return ""
    return f"{issue.emoji} {issue.message}"


def build_hud_data(
    *,
    rep_count: int,
    valid_reps: int,
    invalid_reps: int,
    state: ExerciseState,
    raw_state: ExerciseState = ExerciseState.UNKNOWN,
    detection_confidence: float = 0.0,
    visible_keypoints: int = 0,
    total_keypoints: int = 17,
    fps: float = 0.0,
    inference_ms: float = 0.0,
    form_score: float = 0.0,
    form_grade: FormGrade = FormGrade.POOR,
    elbow_angle: float = float("nan"),
    body_alignment: float = float("nan"),
    arm_diff: float = float("nan"),
    feedback: Optional[List[str]] = None,
    last_rep_valid: Optional[bool] = None,
    last_rep_reason: str = "",
    has_pose: bool = False,
    multiple_people: bool = False,
    message: str = "",
    session_label: str = "",
) -> HudData:
    """Convenience constructor used by the pipeline (keyword-only, explicit)."""
    return HudData(
        rep_count=rep_count,
        valid_reps=valid_reps,
        invalid_reps=invalid_reps,
        state=state,
        raw_state=raw_state,
        detection_confidence=detection_confidence,
        visible_keypoints=visible_keypoints,
        total_keypoints=total_keypoints,
        fps=fps,
        inference_ms=inference_ms,
        form_score=form_score,
        form_grade=form_grade,
        elbow_angle=elbow_angle,
        body_alignment=body_alignment,
        arm_diff=arm_diff,
        feedback=list(feedback or []),
        last_rep_valid=last_rep_valid,
        last_rep_reason=last_rep_reason,
        has_pose=has_pose,
        multiple_people=multiple_people,
        message=message,
        session_label=session_label,
    )


__all__ = ["OverlayRenderer", "HudData", "build_hud_data", "issue_to_feedback"]

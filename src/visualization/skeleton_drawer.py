"""Skeleton rendering: joints, limbs, body line and elbow angle arcs."""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from src.analysis.angle_calculator import AngleSet
from config.settings import VisualizationConfig
from src.pose.keypoints import KEYPOINT_INDEX, LIMB_COLORS, Limb, skeleton_pairs
from src.pose.person import PersonPose
from src.utils.helpers import midpoint
from src.utils.logger import get_logger
from src.visualization.colors import Palette, mix
from src.visualization.drawing import angle_arc, scale_factor

logger = get_logger(__name__)


class SkeletonDrawer:
    """Draws a readable skeleton overlay that never hides the athlete.

    Example:
        >>> drawer = SkeletonDrawer(settings.visualization)
        >>> frame = drawer.draw(frame, pose, angles)
    """

    def __init__(
        self,
        config: VisualizationConfig,
        palette: Optional[Palette] = None,
        *,
        alignment_warning: float = 12.0,
        alignment_critical: float = 25.0,
    ) -> None:
        self.config = config
        self.palette = palette or Palette(config)
        self.alignment_warning = float(alignment_warning)
        self.alignment_critical = float(alignment_critical)
        self._pairs = skeleton_pairs()

    def draw(
        self,
        frame: np.ndarray,
        pose: Optional[PersonPose],
        angles: Optional[AngleSet] = None,
        *,
        draw_angles: bool = True,
        draw_alignment: bool = True,
    ) -> np.ndarray:
        """Render the skeleton (and optional geometry) onto ``frame``.

        Args:
            frame: BGR frame; drawn on in place and returned.
            pose: Primary subject (``None`` renders nothing).
            angles: Angles used for the elbow arcs and the body line.
            draw_angles: Draw the elbow angle arcs and labels.
            draw_alignment: Draw the shoulder-hip-ankle body line.

        Returns:
            The annotated frame (same object that was passed in).
        """
        if pose is None:
            return frame

        scale = scale_factor(frame, self.config.hud.base_width)
        thickness = max(1, int(round(self.config.skeleton.line_thickness * scale)))
        joint_radius = max(2, int(round(self.config.skeleton.joint_radius * scale)))
        highlight_radius = max(3, int(round(self.config.skeleton.highlight_radius * scale)))

        self._draw_limbs(frame, pose, thickness)
        self._draw_joints(frame, pose, joint_radius, highlight_radius)
        if draw_alignment and angles is not None:
            self._draw_body_line(frame, pose, angles, thickness)
        if draw_angles and angles is not None:
            self._draw_angle_arcs(frame, pose, angles, scale)
        return frame

    # ------------------------------------------------------------------- limbs
    def _draw_limbs(self, frame: np.ndarray, pose: PersonPose, thickness: int) -> None:
        invalid_color = self.palette.get("skeleton_invalid")
        for index_a, index_b, limb in self._pairs:
            name_a = _name(index_a)
            name_b = _name(index_b)
            point_a = pose.point(name_a)
            point_b = pose.point(name_b)
            if point_a is None or point_b is None:
                continue
            confidence = min(pose.confidence_of(name_a), pose.confidence_of(name_b))
            base = LIMB_COLORS.get(limb, self.palette.accent)
            color = mix(invalid_color, base, max(0.25, confidence))
            cv2.line(
                frame,
                (int(point_a[0]), int(point_a[1])),
                (int(point_b[0]), int(point_b[1])),
                color,
                thickness,
                cv2.LINE_AA,
            )

    # ------------------------------------------------------------------ joints
    def _draw_joints(self, frame: np.ndarray, pose: PersonPose, radius: int, highlight_radius: int) -> None:
        highlight = self.palette.get("skeleton_highlight")
        for name, keypoint in pose.keypoints.items():
            if not keypoint.is_usable():
                continue
            center = (int(keypoint.x), int(keypoint.y))
            critical = name in _CRITICAL
            current_radius = highlight_radius if critical else radius
            color = highlight if critical else self.palette.get("skeleton")
            cv2.circle(frame, center, current_radius, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, center, max(1, current_radius - 2), color, -1, cv2.LINE_AA)
            if self.config.skeleton.draw_confidence_labels:
                cv2.putText(
                    frame,
                    f"{keypoint.confidence:.2f}",
                    (center[0] + current_radius + 2, center[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    self.palette.text_dim,
                    1,
                    cv2.LINE_AA,
                )

    # -------------------------------------------------------------- body line
    def _draw_body_line(self, frame: np.ndarray, pose: PersonPose, angles: AngleSet, thickness: int) -> None:
        shoulder_mid = midpoint(pose.point("left_shoulder"), pose.point("right_shoulder"))
        hip_mid = midpoint(pose.point("left_hip"), pose.point("right_hip"))
        ankle_mid = midpoint(pose.point("left_ankle"), pose.point("right_ankle"))
        if ankle_mid is None:
            knee_mid = midpoint(pose.point("left_knee"), pose.point("right_knee"))
            ankle_mid = knee_mid
        points = [point for point in (shoulder_mid, hip_mid, ankle_mid) if point is not None]
        if len(points) < 2:
            return

        deviation = angles.alignment_deviation
        color = self.palette.good
        if math.isfinite(deviation):
            if deviation > self.alignment_critical:
                color = self.palette.bad
            elif deviation > self.alignment_warning:
                color = self.palette.warning

        for start, end in zip(points, points[1:], strict=False):
            cv2.line(
                frame,
                (int(start[0]), int(start[1])),
                (int(end[0]), int(end[1])),
                color,
                max(1, thickness - 1),
                cv2.LINE_AA,
            )
        if math.isfinite(angles.body_alignment):
            cv2.putText(
                frame,
                f"{angles.body_alignment:.0f} deg",
                (int(hip_mid[0]) + 10, int(hip_mid[1]) - 10) if hip_mid else (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    # ------------------------------------------------------------------- arcs
    def _draw_angle_arcs(self, frame: np.ndarray, pose: PersonPose, angles: AngleSet, scale: float) -> None:
        radius = int(34 * scale)
        for side, angle_value in (("left", angles.left_elbow), ("right", angles.right_elbow)):
            if not math.isfinite(angle_value):
                continue
            shoulder = pose.point(f"{side}_shoulder")
            elbow = pose.point(f"{side}_elbow")
            wrist = pose.point(f"{side}_wrist")
            if shoulder is None or elbow is None or wrist is None:
                continue
            color = self.palette.good if angle_value <= 100 else self.palette.warning if angle_value < 150 else self.palette.tracking
            angle_arc(
                frame,
                elbow,
                shoulder,
                wrist,
                angle_value,
                color,
                radius=radius,
                label=f"{angle_value:.0f}",
                font_scale=0.5 * scale,
            )


_CRITICAL = {
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
}
_NAME_BY_INDEX = {index: name for name, index in KEYPOINT_INDEX.items()}


def _name(index: int) -> str:
    """Canonical keypoint name for a skeleton index."""
    return _NAME_BY_INDEX[index]


__all__ = ["SkeletonDrawer", "Limb"]

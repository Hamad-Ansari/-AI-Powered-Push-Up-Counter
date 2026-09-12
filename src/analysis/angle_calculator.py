"""Joint-angle geometry for push-up analysis.

Everything here is pure geometry on ``(x, y)`` points: no OpenCV, no model
dependencies, fully unit-testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.pose.person import PersonPose
from src.utils.helpers import midpoint
from src.utils.logger import get_logger

logger = get_logger(__name__)

Point = Optional[Sequence[float]]


class AngleResult:
    """Result of a single angle computation.

    Attributes:
        degrees: Angle at the vertex in ``[0, 180]``, or ``NaN`` when invalid.
        valid: ``False`` when the angle could not be computed.
        reason: Human readable explanation when ``valid`` is ``False``.
    """

    __slots__ = ("degrees", "valid", "reason")

    def __init__(self, degrees: float, valid: bool = True, reason: str = "") -> None:
        self.degrees = degrees
        self.valid = valid
        self.reason = reason

    @property
    def value(self) -> float:
        """Alias for :attr:`degrees`."""
        return self.degrees

    def __float__(self) -> float:
        return float(self.degrees)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"AngleResult({self.degrees:.2f}, valid={self.valid}, reason={self.reason!r})"


INVALID_POINTS = AngleResult(float("nan"), False, "Missing keypoint")
ZERO_VECTOR = AngleResult(float("nan"), False, "Zero-length vector (overlapping joints)")
INVALID_COORDINATES = AngleResult(float("nan"), False, "Non-finite coordinates")


def calculate_angle(
    point_a: Point,
    point_b: Point,
    point_c: Point,
    *,
    min_norm: float = 1e-6,
) -> float:
    """Angle at ``point_b`` formed by the segments ``B->A`` and ``B->C``.

    ``BA = A - B``, ``BC = C - B`` and

    ``angle = arccos(dot(BA, BC) / (|BA| * |BC|))`` converted to degrees.

    The function never raises: missing points, non-finite coordinates and
    zero-length vectors all return ``NaN``.

    Args:
        point_a: First end point.
        point_b: Vertex - the joint the angle is measured at.
        point_c: Second end point.
        min_norm: Vectors shorter than this are treated as degenerate.

    Returns:
        Angle in degrees within ``[0, 180]``, or ``NaN`` when it cannot be computed.
    """
    result = calculate_angle_detailed(point_a, point_b, point_c, min_norm=min_norm)
    return result.degrees


def calculate_angle_detailed(
    point_a: Point,
    point_b: Point,
    point_c: Point,
    *,
    min_norm: float = 1e-6,
) -> AngleResult:
    """Same as :func:`calculate_angle` but returns an :class:`AngleResult`.

    Args:
        point_a: First end point.
        point_b: Vertex.
        point_c: Second end point.
        min_norm: Degenerate-vector threshold.

    Returns:
        An :class:`AngleResult` carrying the value, validity flag and reason.
    """
    if point_a is None or point_b is None or point_c is None:
        return AngleResult(float("nan"), False, INVALID_POINTS.reason)

    try:
        a = np.asarray(point_a, dtype=float)[:2]
        b = np.asarray(point_b, dtype=float)[:2]
        c = np.asarray(point_c, dtype=float)[:2]
    except (TypeError, ValueError):
        return AngleResult(float("nan"), False, INVALID_COORDINATES.reason)

    if a.shape != (2,) or b.shape != (2,) or c.shape != (2,):
        return AngleResult(float("nan"), False, INVALID_COORDINATES.reason)
    if not (np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(c).all()):
        return AngleResult(float("nan"), False, INVALID_COORDINATES.reason)

    ba = a - b
    bc = c - b
    norm_ba = float(np.linalg.norm(ba))
    norm_bc = float(np.linalg.norm(bc))
    if norm_ba < min_norm or norm_bc < min_norm:
        return AngleResult(float("nan"), False, ZERO_VECTOR.reason)

    cosine = float(np.dot(ba, bc) / (norm_ba * norm_bc))
    cosine = max(-1.0, min(1.0, cosine))  # guard against float drift outside [-1, 1]
    return AngleResult(math.degrees(math.acos(cosine)), True)


def angle_from_person(pose: Optional[PersonPose], a: str, b: str, c: str) -> float:
    """Angle at joint ``b`` using named keypoints of a :class:`PersonPose`."""
    if pose is None:
        return float("nan")
    return calculate_angle(pose.point(a), pose.point(b), pose.point(c))


@dataclass(slots=True)
class AngleSet:
    """All angles required by the push-up analysis for one frame.

    Attributes:
        left_elbow: Angle at the left elbow (shoulder-elbow-wrist).
        right_elbow: Angle at the right elbow.
        elbow_mean: Mean of the two usable elbow angles.
        elbow_diff: Absolute left/right difference (arm symmetry).
        body_alignment: Angle at the hip midpoint (shoulder-hip-ankle midpoints).
        trunk_angle: Angle at the shoulder midpoint (hip-shoulder-nose).
        alignment_deviation: ``|body_alignment - 180|``.
        available: Names of the angles that could be computed this frame.
    """

    left_elbow: float = float("nan")
    right_elbow: float = float("nan")
    elbow_mean: float = float("nan")
    elbow_diff: float = float("nan")
    body_alignment: float = float("nan")
    trunk_angle: float = float("nan")
    alignment_deviation: float = float("nan")
    available: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ access
    def elbow_angles(self) -> List[float]:
        """Usable elbow angles (0, 1 or 2 entries)."""
        return [value for value in (self.left_elbow, self.right_elbow) if math.isfinite(value)]

    def get(self, name: str) -> float:
        """Angle by attribute name (``NaN`` when unknown)."""
        return float(getattr(self, name, float("nan")))

    def as_dict(self) -> Dict[str, float]:
        """JSON-friendly snapshot used by analytics and reports."""
        return {
            "left_elbow": _round(self.left_elbow),
            "right_elbow": _round(self.right_elbow),
            "elbow_mean": _round(self.elbow_mean),
            "elbow_diff": _round(self.elbow_diff),
            "body_alignment": _round(self.body_alignment),
            "trunk_angle": _round(self.trunk_angle),
            "alignment_deviation": _round(self.alignment_deviation),
        }

    def primary_elbow(self) -> float:
        """Best available elbow angle: mean, else whichever side is visible."""
        if math.isfinite(self.elbow_mean):
            return self.elbow_mean
        angles = self.elbow_angles()
        return angles[0] if angles else float("nan")


def _round(value: float, digits: int = 2) -> float:
    return round(float(value), digits) if math.isfinite(value) else float("nan")


def compute_angles(pose: Optional[PersonPose]) -> AngleSet:
    """Compute the full :class:`AngleSet` for one pose.

    Args:
        pose: The primary subject (may be ``None``).

    Returns:
        An :class:`AngleSet`; every angle that could not be computed is ``NaN``.
    """
    if pose is None:
        return AngleSet()

    left_elbow = angle_from_person(pose, "left_shoulder", "left_elbow", "left_wrist")
    right_elbow = angle_from_person(pose, "right_shoulder", "right_elbow", "right_wrist")

    elbow_values = [value for value in (left_elbow, right_elbow) if math.isfinite(value)]
    elbow_mean = float(np.mean(elbow_values)) if elbow_values else float("nan")
    elbow_diff = abs(left_elbow - right_elbow) if len(elbow_values) == 2 else float("nan")

    shoulder_mid = midpoint(pose.point("left_shoulder"), pose.point("right_shoulder"))
    hip_mid = midpoint(pose.point("left_hip"), pose.point("right_hip"))
    ankle_mid = midpoint(pose.point("left_ankle"), pose.point("right_ankle"))
    knee_mid = midpoint(pose.point("left_knee"), pose.point("right_knee"))
    nose = pose.point("nose")

    body_alignment = calculate_angle(shoulder_mid, hip_mid, ankle_mid)
    if not math.isfinite(body_alignment) and knee_mid is not None:
        # Ankles are frequently occluded behind the body: fall back to the knee.
        body_alignment = calculate_angle(shoulder_mid, hip_mid, knee_mid)

    trunk_angle = calculate_angle(hip_mid, shoulder_mid, nose)
    alignment_deviation = abs(body_alignment - 180.0) if math.isfinite(body_alignment) else float("nan")

    available = [
        name
        for name, value in (
            ("left_elbow", left_elbow),
            ("right_elbow", right_elbow),
            ("body_alignment", body_alignment),
            ("trunk_angle", trunk_angle),
        )
        if math.isfinite(value)
    ]

    return AngleSet(
        left_elbow=left_elbow,
        right_elbow=right_elbow,
        elbow_mean=elbow_mean,
        elbow_diff=elbow_diff,
        body_alignment=body_alignment,
        trunk_angle=trunk_angle,
        alignment_deviation=alignment_deviation,
        available=available,
    )


def body_alignment_score(deviation: float, tolerance: float = 12.0, max_deviation: float = 35.0) -> float:
    """Map an alignment deviation to a ``[0, 1]`` score.

    Args:
        deviation: ``|hip_angle - 180|`` in degrees.
        tolerance: Deviation that is still considered perfect.
        max_deviation: Deviation that maps to a score of ``0``.

    Returns:
        Score in ``[0, 1]`` (``0.0`` when the deviation is unknown).
    """
    if not math.isfinite(deviation):
        return 0.0
    if deviation <= tolerance:
        return 1.0
    span = max(1e-6, max_deviation - tolerance)
    return max(0.0, 1.0 - (deviation - tolerance) / span)


def depth_score(elbow_angle: float, full_angle: float = 90.0, zero_angle: float = 160.0) -> float:
    """Map the deepest elbow angle reached to a ``[0, 1]`` depth score."""
    if not math.isfinite(elbow_angle):
        return 0.0
    if elbow_angle <= full_angle:
        return 1.0
    if elbow_angle >= zero_angle:
        return 0.0
    return (zero_angle - elbow_angle) / max(1e-6, zero_angle - full_angle)


def symmetry_score(elbow_diff: float, threshold: float = 20.0) -> float:
    """Map the left/right elbow difference to a ``[0, 1]`` symmetry score."""
    if not math.isfinite(elbow_diff):
        return 0.5  # unknown: neutral, one arm is simply out of frame
    return max(0.0, 1.0 - elbow_diff / max(1e-6, 2.0 * threshold))


__all__ = [
    "AngleResult",
    "AngleSet",
    "calculate_angle",
    "calculate_angle_detailed",
    "angle_from_person",
    "compute_angles",
    "body_alignment_score",
    "depth_score",
    "symmetry_score",
]

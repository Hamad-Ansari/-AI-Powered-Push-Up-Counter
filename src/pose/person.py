"""Model-agnostic keypoint data structures.

These dataclasses are the single contract between the pose backends and the rest
of the application (adapter pattern). Nothing outside :mod:`src.pose` ever sees
an ``sv.KeyPoints`` or a raw tensor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.pose.keypoints import (
    CRITICAL_KEYPOINTS,
    KEYPOINT_NAMES,
    MIN_VALID_FOR_ANALYSIS,
    NUM_KEYPOINTS,
)

Box = Tuple[float, float, float, float]


@dataclass(slots=True)
class Keypoint:
    """A single anatomical joint.

    Attributes:
        x: Horizontal pixel coordinate.
        y: Vertical pixel coordinate.
        confidence: Model confidence in ``[0, 1]``.
        valid: ``False`` when missing, non-finite or below the confidence gate.
    """

    x: float
    y: float
    confidence: float = 1.0
    valid: bool = True

    def as_tuple(self) -> Tuple[float, float]:
        """Return ``(x, y)``."""
        return (self.x, self.y)

    def as_dict(self) -> Dict[str, float]:
        """Return the ``{"x":..,"y":..,"confidence":..}`` form used in reports."""
        return {"x": round(self.x, 2), "y": round(self.y, 2), "confidence": round(self.confidence, 4)}

    def is_usable(self) -> bool:
        """``True`` when the joint can safely participate in geometry."""
        return bool(self.valid and math.isfinite(self.x) and math.isfinite(self.y))


@dataclass(slots=True)
class PersonPose:
    """Standardized pose of one detected person.

    Attributes:
        keypoints: Name -> :class:`Keypoint` mapping for all 17 COCO joints.
        detection_confidence: Per-instance score from the detector.
        bbox: ``[x1, y1, x2, y2]`` bounding box in pixels (``None`` if unknown).
        frame_id: Index of the frame this pose came from.
        track_id: Temporally stable identity assigned by the subject tracker.
    """

    keypoints: Dict[str, Keypoint]
    detection_confidence: float = 1.0
    bbox: Optional[Box] = None
    frame_id: int = -1
    track_id: int = -1

    # ------------------------------------------------------------------ access
    def get(self, name: str) -> Optional[Keypoint]:
        """Return the :class:`Keypoint` for ``name`` or ``None`` when unknown."""
        return self.keypoints.get(name)

    def point(self, name: str) -> Optional[Tuple[float, float]]:
        """Return ``(x, y)`` for a *usable* joint, otherwise ``None``."""
        kp = self.keypoints.get(name)
        if kp is None or not kp.is_usable():
            return None
        return (kp.x, kp.y)

    def confidence_of(self, name: str) -> float:
        """Confidence of a joint (``0.0`` when the joint is missing)."""
        kp = self.keypoints.get(name)
        return float(kp.confidence) if kp is not None else 0.0

    def required_points(self, names: Sequence[str]) -> Optional[Dict[str, Tuple[float, float]]]:
        """Return all requested joints or ``None`` if any of them is unusable."""
        resolved: Dict[str, Tuple[float, float]] = {}
        for name in names:
            point = self.point(name)
            if point is None:
                return None
            resolved[name] = point
        return resolved

    # ------------------------------------------------------------- statistics
    @property
    def valid_names(self) -> List[str]:
        """Names of all joints that passed validation."""
        return [name for name in KEYPOINT_NAMES if self.point(name) is not None]

    @property
    def valid_count(self) -> int:
        """Number of usable joints."""
        return len(self.valid_names)

    @property
    def mean_confidence(self) -> float:
        """Mean confidence over usable joints (``0.0`` when there are none)."""
        confidences = [kp.confidence for kp in self.keypoints.values() if kp.is_usable()]
        return float(sum(confidences) / len(confidences)) if confidences else 0.0

    def has_minimal_pose(self) -> bool:
        """``True`` when both arms are fully observable (needed for push-ups)."""
        return all(self.point(name) is not None for name in MIN_VALID_FOR_ANALYSIS)

    def critical_missing(self) -> List[str]:
        """Analysis-critical joints that are currently unusable."""
        return [name for name in CRITICAL_KEYPOINTS if self.point(name) is None]

    def tracking_ratio_parts(self) -> Tuple[int, int]:
        """``(usable critical joints, total critical joints)``."""
        usable = sum(1 for name in CRITICAL_KEYPOINTS if self.point(name) is not None)
        return usable, len(CRITICAL_KEYPOINTS)

    def area(self) -> float:
        """Bounding-box area, used by the primary-person selector."""
        if self.bbox is None:
            return 0.0
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def center(self) -> Optional[Tuple[float, float]]:
        """Bounding-box centre, used by the primary-person selector."""
        if self.bbox is None:
            xs = [kp.x for kp in self.keypoints.values() if kp.is_usable()]
            ys = [kp.y for kp in self.keypoints.values() if kp.is_usable()]
            if not xs:
                return None
            return (sum(xs) / len(xs), sum(ys) / len(ys))
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def to_dict(self) -> Dict[str, object]:
        """JSON-serialisable snapshot (used by reports and the REST API)."""
        return {
            "detection_confidence": round(float(self.detection_confidence), 4),
            "bbox": [round(float(v), 2) for v in self.bbox] if self.bbox else None,
            "track_id": int(self.track_id),
            "frame_id": int(self.frame_id),
            "keypoints": {name: kp.as_dict() for name, kp in self.keypoints.items()},
        }


@dataclass(slots=True)
class PoseFrame:
    """Pose detection result for one video frame.

    Attributes:
        people: Every detected person (already normalised and validated).
        primary: The subject currently being analysed (``None`` when nobody is seen).
        frame_id: Frame index in the source.
        timestamp: Wall-clock time of capture.
        inference_ms: Detector latency for this frame.
        multiple_people: ``True`` when more than one person was detected.
    """

    people: List[PersonPose] = field(default_factory=list)
    primary: Optional[PersonPose] = None
    frame_id: int = 0
    timestamp: float = 0.0
    inference_ms: float = 0.0
    multiple_people: bool = False

    @property
    def detected(self) -> bool:
        """``True`` when at least one analysable person is present."""
        return self.primary is not None

    def __len__(self) -> int:
        return len(self.people)


@dataclass(slots=True)
class DetectorStatus:
    """Runtime health of the pose backend (surfaced in the UI)."""

    backend: str = "unknown"
    loaded: bool = False
    device: str = "cpu"
    model_name: str = ""
    error: str = ""
    keypoints_supported: int = NUM_KEYPOINTS

    def describe(self) -> str:
        """One-line human readable status."""
        if self.error:
            return f"{self.backend}: {self.error}"
        return f"{self.backend} | {self.model_name} | {self.device}"


def empty_pose(frame_id: int = -1) -> PersonPose:
    """A fully-invalid :class:`PersonPose` (useful as a null object in tests)."""
    return PersonPose(
        keypoints={name: Keypoint(float("nan"), float("nan"), 0.0, False) for name in KEYPOINT_NAMES},
        detection_confidence=0.0,
        bbox=None,
        frame_id=frame_id,
    )


def count_valid(poses: Iterable[PersonPose]) -> int:
    """How many poses in ``poses`` have a minimal analysable skeleton."""
    return sum(1 for pose in poses if pose.has_minimal_pose())

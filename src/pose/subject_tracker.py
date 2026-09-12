"""Primary-subject selection with temporal stability.

Two responsibilities:

1. Pick *one* person to analyse when several are in frame (highest confidence,
   then largest, then most central - a configurable weighted score).
2. Keep that identity stable across frames: the tracked subject is only replaced
   when a challenger wins by a configurable margin, which prevents the skeleton
   from flickering between two people.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from config.settings import PersonSelectionConfig
from src.pose.person import PersonPose
from src.utils.helpers import box_iou, clamp
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SubjectTracker:
    """Stable primary-subject selection.

    Example:
        >>> tracker = SubjectTracker(PersonSelectionConfig())
        >>> pose = tracker.select([person_a, person_b], frame_size=(720, 1280))
        >>> pose.track_id
        1
    """

    def __init__(self, config: PersonSelectionConfig) -> None:
        self.config = config
        self._next_track_id = 1
        self._tracks: Dict[int, PersonPose] = {}
        self._lost: Dict[int, int] = {}
        self._primary_id: Optional[int] = None
        self._switch_count = 0

    # ------------------------------------------------------------------ score
    def score(self, pose: PersonPose, frame_size: Sequence[int], max_area: float) -> float:
        """Composite preference score in ``[0, 1]`` (higher = better subject).

        Args:
            pose: Candidate person.
            frame_size: ``(height, width)`` of the frame.
            max_area: Largest bounding-box area in this frame (for normalisation).
        """
        weights = self.config.weights
        height, width = float(frame_size[0]), float(frame_size[1])

        confidence = clamp(pose.detection_confidence, 0.0, 1.0)
        size = clamp(pose.area() / max_area, 0.0, 1.0) if max_area > 0 else 0.0

        center = pose.center()
        if center is None:
            centrality = 0.0
        else:
            dx = abs(center[0] - width / 2.0) / (width / 2.0)
            dy = abs(center[1] - height / 2.0) / (height / 2.0)
            centrality = clamp(1.0 - math_hypot(dx, dy) / 1.41421356, 0.0, 1.0)

        total_weight = sum(weights.values()) or 1.0
        return (
            weights.get("confidence", 0.0) * confidence
            + weights.get("size", 0.0) * size
            + weights.get("centrality", 0.0) * centrality
        ) / total_weight

    # ----------------------------------------------------------------- select
    def select(self, people: Sequence[PersonPose], frame_size: Sequence[int]) -> Optional[PersonPose]:
        """Choose the primary subject for this frame and stamp ``track_id``.

        Args:
            people: All validated people detected in the frame.
            frame_size: ``(height, width)`` of the frame.

        Returns:
            The primary :class:`PersonPose`, or ``None`` when nobody is present.
        """
        if not people:
            self._register_miss()
            return None

        max_area = max((pose.area() for pose in people), default=0.0)
        scored = sorted(
            ((self.score(pose, frame_size, max_area), index, pose) for index, pose in enumerate(people)),
            key=lambda item: (-item[0], item[1]),
        )
        best_score, _, best_pose = scored[0]

        primary = self._match_tracked(people, frame_size)
        if primary is not None:
            challenger_score = best_score
            tracked_score = self.score(primary, frame_size, max_area)
            if challenger_score <= tracked_score + self.config.tracking_margin:
                return self._promote(primary)
            logger.debug(
                "Switching primary subject %s -> %s (score %.3f -> %.3f)",
                self._primary_id,
                best_pose.track_id,
                tracked_score,
                challenger_score,
            )
            self._switch_count += 1
        return self._promote(best_pose)

    # ---------------------------------------------------------------- tracking
    def _match_tracked(self, people: Sequence[PersonPose], frame_size: Sequence[int]) -> Optional[PersonPose]:
        """Re-associate known tracks by IoU, then by proximity to the last position."""
        for pose in people:
            best_id, best_iou = -1, 0.0
            for track_id, previous in self._tracks.items():
                iou = box_iou(pose.bbox, previous.bbox)
                if iou > best_iou:
                    best_id, best_iou = track_id, iou
            if best_id >= 0 and best_iou >= self.config.tracking_iou_threshold:
                pose.track_id = best_id
                self._tracks[best_id] = pose
                self._lost.pop(best_id, None)
                continue

            # No spatial overlap: fall back to the nearest previous position.
            fallback_id = self._nearest_track(pose, frame_size)
            if fallback_id is not None:
                pose.track_id = fallback_id
                self._tracks[fallback_id] = pose
                self._lost.pop(fallback_id, None)
            else:
                pose.track_id = self._next_track_id
                self._tracks[self._next_track_id] = pose
                self._next_track_id += 1
        return self._tracks.get(self._primary_id) if self._primary_id is not None else None

    def _nearest_track(self, pose: PersonPose, frame_size: Sequence[int], max_diagonal_ratio: float = 0.30) -> Optional[int]:
        """Return the track whose last centre is closest, within a distance budget."""
        center = pose.center()
        if center is None or not self._tracks:
            return None
        height, width = float(frame_size[0]), float(frame_size[1])
        limit = max_diagonal_ratio * math_hypot(width, height)

        best_id: Optional[int] = None
        best_distance = float("inf")
        for track_id, previous in self._tracks.items():
            previous_center = previous.center()
            if previous_center is None:
                continue
            distance = math_hypot(center[0] - previous_center[0], center[1] - previous_center[1])
            if distance < best_distance:
                best_id, best_distance = track_id, distance
        if best_id is not None and best_distance <= limit:
            return best_id
        return None

    def _promote(self, pose: PersonPose) -> PersonPose:
        self._primary_id = pose.track_id
        return pose

    def _register_miss(self) -> None:
        """Age out tracks that have not been seen for a while."""
        for track_id in list(self._tracks):
            self._lost[track_id] = self._lost.get(track_id, 0) + 1
            if self._lost[track_id] > self.config.max_lost_frames:
                self._tracks.pop(track_id, None)
                self._lost.pop(track_id, None)
                if self._primary_id == track_id:
                    self._primary_id = None

    # ------------------------------------------------------------------ state
    @property
    def primary_track_id(self) -> Optional[int]:
        """Track id of the currently analysed subject."""
        return self._primary_id

    @property
    def active_tracks(self) -> List[int]:
        """All currently live track ids."""
        return sorted(self._tracks)

    @property
    def switch_count(self) -> int:
        """How many times the primary subject changed."""
        return self._switch_count

    def reset(self) -> None:
        """Forget every track (new session)."""
        self._tracks.clear()
        self._lost.clear()
        self._primary_id = None
        self._next_track_id = 1
        self._switch_count = 0


def math_hypot(dx: float, dy: float) -> float:
    """Euclidean length of a 2D vector (kept local to avoid a module import)."""
    return (dx * dx + dy * dy) ** 0.5

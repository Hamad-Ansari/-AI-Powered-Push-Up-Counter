"""Movement analysis: position classification, phase timing and velocity.

The classifier is *debounced*: a single noisy frame can never flip the exercise
state. A new position only becomes official after ``min_stable_frames``
consecutive agreeing frames, which is what stops the rep counter from
double-counting on jittery pose output.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional, Tuple

from src.analysis.angle_calculator import AngleSet
from src.analysis.smoothing import AngleSmoother
from config.settings import MovementConfig, PushUpConfig, SmoothingConfig
from src.analysis.smoothing import create_smoother
from src.utils.helpers import moving_statistics
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExerciseState(str, Enum):
    """Push-up movement state."""

    UNKNOWN = "UNKNOWN"
    UP = "UP"
    MOVING = "MOVING"
    DOWN = "DOWN"

    @property
    def label(self) -> str:
        """Display label used by the HUD and the dashboard."""
        return {
            ExerciseState.UNKNOWN: "WAITING FOR POSE",
            ExerciseState.UP: "UP",
            ExerciseState.MOVING: "MOVING",
            ExerciseState.DOWN: "DOWN",
        }[self]

    @property
    def status_key(self) -> str:
        """Colour key used to look the state up in the palette."""
        return {
            ExerciseState.UNKNOWN: "neutral",
            ExerciseState.UP: "good",
            ExerciseState.MOVING: "warning",
            ExerciseState.DOWN: "tracking",
        }[self]


class MovementDirection(str, Enum):
    """Direction of travel of the elbow angle."""

    NONE = "none"
    DESCENDING = "descending"
    ASCENDING = "ascending"


@dataclass(slots=True)
class CycleRecord:
    """Aggregated metrics of one UP -> DOWN -> UP attempt."""

    started_at: float = 0.0
    ended_at: float = 0.0
    start_angle: float = float("nan")
    min_elbow_angle: float = float("inf")
    max_elbow_angle: float = float("-inf")
    max_alignment_deviation: float = 0.0
    mean_alignment_deviation: float = float("nan")
    max_arm_diff: float = 0.0
    mean_arm_diff: float = float("nan")
    tracking_ratio: float = 1.0
    frames: int = 0
    frames_with_pose: int = 0
    alignment_samples: List[float] = field(default_factory=list)
    arm_diff_samples: List[float] = field(default_factory=list)

    # ------------------------------------------------------------- accumulate
    def observe(
        self,
        elbow_angle: float,
        alignment_deviation: float,
        arm_diff: float,
        has_pose: bool,
        timestamp: float,
    ) -> None:
        """Fold one frame of the ongoing cycle into the record."""
        if self.started_at == 0.0:
            self.started_at = timestamp
            self.start_angle = elbow_angle
        self.ended_at = timestamp
        self.frames += 1
        if not has_pose:
            return
        self.frames_with_pose += 1
        if math.isfinite(elbow_angle):
            self.min_elbow_angle = min(self.min_elbow_angle, elbow_angle)
            self.max_elbow_angle = max(self.max_elbow_angle, elbow_angle)
        if math.isfinite(alignment_deviation):
            self.max_alignment_deviation = max(self.max_alignment_deviation, alignment_deviation)
            self.alignment_samples.append(alignment_deviation)
        if math.isfinite(arm_diff):
            self.max_arm_diff = max(self.max_arm_diff, arm_diff)
            self.arm_diff_samples.append(arm_diff)

    # -------------------------------------------------------------- properties
    @property
    def duration(self) -> float:
        """Cycle duration in seconds."""
        return max(0.0, self.ended_at - self.started_at)

    @property
    def elbow_travel(self) -> float:
        """Total elbow flexion achieved during the cycle, in degrees."""
        if not (math.isfinite(self.min_elbow_angle) and math.isfinite(self.max_elbow_angle)):
            return 0.0
        return max(0.0, self.max_elbow_angle - self.min_elbow_angle)

    def finalize(self) -> CycleRecord:
        """Compute the derived means and return ``self``."""
        alignment_stats = moving_statistics(self.alignment_samples)
        arm_stats = moving_statistics(self.arm_diff_samples)
        self.mean_alignment_deviation = alignment_stats["mean"]
        self.mean_arm_diff = arm_stats["mean"]
        if math.isinf(self.min_elbow_angle):
            self.min_elbow_angle = float("nan")
        if math.isinf(self.max_elbow_angle):
            self.max_elbow_angle = float("nan")
        self.tracking_ratio = (self.frames_with_pose / self.frames) if self.frames else 0.0
        return self

    def to_dict(self) -> dict:
        """JSON-friendly snapshot for reports and analytics."""
        return {
            "duration_s": round(self.duration, 3),
            "start_angle": _r(self.start_angle),
            "min_elbow_angle": _r(self.min_elbow_angle),
            "max_elbow_angle": _r(self.max_elbow_angle),
            "elbow_travel": round(self.elbow_travel, 2),
            "max_alignment_deviation": _r(self.max_alignment_deviation),
            "mean_alignment_deviation": _r(self.mean_alignment_deviation),
            "max_arm_diff": _r(self.max_arm_diff),
            "mean_arm_diff": _r(self.mean_arm_diff),
            "tracking_ratio": round(self.tracking_ratio, 3),
            "frames": self.frames,
            "frames_with_pose": self.frames_with_pose,
        }


def _r(value: float, digits: int = 2) -> float:
    return round(float(value), digits) if math.isfinite(value) else float("nan")


def classify_position(
    elbow_angle: Optional[float],
    up_threshold: float = 150.0,
    down_threshold: float = 90.0,
) -> ExerciseState:
    """Classify a single (smoothed) elbow angle into a raw position.

    Args:
        elbow_angle: Smoothed elbow angle in degrees (``None``/NaN = no pose).
        up_threshold: Angle at or above which the arms count as extended.
        down_threshold: Angle at or below which the chest counts as lowered.

    Returns:
        :attr:`ExerciseState.UP`, :attr:`ExerciseState.DOWN`,
        :attr:`ExerciseState.MOVING` or :attr:`ExerciseState.UNKNOWN`.
    """
    if elbow_angle is None or not math.isfinite(elbow_angle):
        return ExerciseState.UNKNOWN
    if elbow_angle >= up_threshold:
        return ExerciseState.UP
    if elbow_angle <= down_threshold:
        return ExerciseState.DOWN
    return ExerciseState.MOVING


class StableStateClassifier:
    """Debounce filter: confirms a state only after N agreeing frames.

    Example:
        >>> classifier = StableStateClassifier(min_stable_frames=3)
        >>> for _ in range(3):
        ...     state = classifier.update(ExerciseState.DOWN)
        >>> state
        <ExerciseState.DOWN: 'DOWN'>
    """

    def __init__(self, min_stable_frames: int = 3, initial: ExerciseState = ExerciseState.UNKNOWN) -> None:
        self.min_stable_frames = max(1, int(min_stable_frames))
        self.confirmed = initial
        self._candidate: ExerciseState = initial
        self._streak = 0

    def update(self, raw: ExerciseState) -> ExerciseState:
        """Feed one raw classification and return the confirmed state."""
        if raw == self.confirmed:
            self._candidate = raw
            self._streak = 0
            return self.confirmed
        if raw == self._candidate:
            self._streak += 1
        else:
            self._candidate = raw
            self._streak = 1
        if self._streak >= self.min_stable_frames:
            self.confirmed = raw
            self._streak = 0
        return self.confirmed

    @property
    def candidate(self) -> ExerciseState:
        """State that is currently accumulating evidence."""
        return self._candidate

    @property
    def streak(self) -> int:
        """Consecutive frames the candidate has been observed."""
        return self._streak

    def reset(self, initial: ExerciseState = ExerciseState.UNKNOWN) -> None:
        """Forget all accumulated evidence."""
        self.confirmed = initial
        self._candidate = initial
        self._streak = 0


@dataclass(slots=True)
class MovementState:
    """Per-frame movement snapshot consumed by the counter and the HUD."""

    raw_angle: float = float("nan")
    smoothed_angle: float = float("nan")
    raw_position: ExerciseState = ExerciseState.UNKNOWN
    position: ExerciseState = ExerciseState.UNKNOWN
    velocity: float = 0.0
    direction: MovementDirection = MovementDirection.NONE
    is_jerky: bool = False
    phase_duration: float = 0.0
    phase_transition: bool = False
    previous_position: ExerciseState = ExerciseState.UNKNOWN


class MovementAnalyzer:
    """Smooths the elbow angle and derives movement features.

    Responsibilities:

    * apply the configured smoother to the raw elbow angle;
    * classify the position (raw + debounced);
    * estimate angular velocity and detect jerky motion;
    * measure how long the current phase has lasted.
    """

    def __init__(
        self,
        pushup_config: PushUpConfig,
        smoothing_config: SmoothingConfig,
        movement_config: Optional[MovementConfig] = None,
    ) -> None:
        self.pushup = pushup_config
        self.movement = movement_config or MovementConfig()
        self.smoother: AngleSmoother = create_smoother(smoothing_config)
        self.classifier = StableStateClassifier(pushup_config.stable_frames)
        self._angles: Deque[Tuple[float, float]] = deque(maxlen=max(2, self.movement.velocity_window))
        self._phase_started: Optional[float] = None
        self._last_position: ExerciseState = ExerciseState.UNKNOWN
        self._frame_count = 0

    # ------------------------------------------------------------------ update
    def update(self, raw_angle: Optional[float], timestamp: Optional[float] = None) -> MovementState:
        """Process one frame.

        Args:
            raw_angle: Raw elbow angle (``None`` when no pose).
            timestamp: Frame timestamp in seconds (defaults to a frame counter).

        Returns:
            The resulting :class:`MovementState`.
        """
        self._frame_count += 1
        stamp = float(self._frame_count) / 30.0 if timestamp is None else float(timestamp)

        smoothed = self.smoother.update(raw_angle)
        raw_position = classify_position(smoothed, self.pushup.up_threshold, self.pushup.down_threshold)
        confirmed = self.classifier.update(raw_position)

        if smoothed is not None and math.isfinite(smoothed):
            self._angles.append((stamp, float(smoothed)))
        else:
            self._angles.clear()

        velocity = self._velocity()
        direction = MovementDirection.NONE
        if abs(velocity) > 1.0:
            direction = MovementDirection.DESCENDING if velocity < 0 else MovementDirection.ASCENDING

        transition = confirmed != self._last_position
        if transition or self._phase_started is None:
            self._phase_started = stamp
        phase_duration = stamp - (self._phase_started if self._phase_started is not None else stamp)

        state = MovementState(
            raw_angle=float(raw_angle) if raw_angle is not None and math.isfinite(raw_angle) else float("nan"),
            smoothed_angle=float(smoothed) if smoothed is not None else float("nan"),
            raw_position=raw_position,
            position=confirmed,
            velocity=velocity,
            direction=direction,
            is_jerky=abs(velocity) > self.jerk_threshold,
            phase_duration=phase_duration,
            phase_transition=transition,
            previous_position=self._last_position,
        )
        self._last_position = confirmed
        return state

    def analyze(self, angles: AngleSet, timestamp: Optional[float] = None) -> MovementState:
        """Convenience wrapper that takes the whole :class:`AngleSet`."""
        return self.update(angles.primary_elbow(), timestamp)

    @property
    def jerk_threshold(self) -> float:
        """Angular velocity (deg/s) above which the movement counts as jerky."""
        return max(60.0, self.movement.min_phase_duration * 400.0)

    # ---------------------------------------------------------------- velocity
    def _velocity(self) -> float:
        """Angular velocity in degrees/second over the recent window."""
        if len(self._angles) < 2:
            return 0.0
        (t0, a0), (t1, a1) = self._angles[0], self._angles[-1]
        dt = t1 - t0
        if dt <= 1e-6:
            return 0.0
        return (a1 - a0) / dt

    # ------------------------------------------------------------------- state
    @property
    def position(self) -> ExerciseState:
        """Currently confirmed position."""
        return self.classifier.confirmed

    @property
    def smoother_name(self) -> str:
        """Name of the active smoothing filter."""
        return self.smoother.name

    def reset(self) -> None:
        """Reset smoothing, classification and phase timing."""
        self.smoother.reset()
        self.classifier.reset()
        self._angles.clear()
        self._phase_started = None
        self._last_position = ExerciseState.UNKNOWN


__all__ = [
    "ExerciseState",
    "MovementDirection",
    "MovementState",
    "MovementAnalyzer",
    "StableStateClassifier",
    "CycleRecord",
    "classify_position",
]

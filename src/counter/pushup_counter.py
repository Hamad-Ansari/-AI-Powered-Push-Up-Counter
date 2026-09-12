"""Finite-state push-up repetition counter with validity filtering.

A repetition is only counted when a *complete* cycle happens::

    UP -> MOVING -> DOWN -> MOVING -> UP

and every validation rule passes. Partial movements, noise spikes, lost tracking
and poor body alignment are recorded as invalid attempts instead of reps.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from src.analysis.angle_calculator import AngleSet
from src.analysis.form_analyzer import FormAnalyzer
from src.analysis.form_scorer import FormResult
from src.analysis.movement_analyzer import (
    CycleRecord,
    ExerciseState,
    MovementAnalyzer,
    MovementState,
    classify_position,
)
from src.analysis.movement_analyzer import MovementDirection  # noqa: F401  (re-exported for callers)
from src.analysis.posture_analyzer import PostureResult
from config.settings import MovementConfig, PushUpConfig, SmoothingConfig, ValidationConfig
from src.pose.person import PersonPose
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class RepValidationResult:
    """Outcome of the rep validation rules.

    Attributes:
        valid: ``True`` when the repetition should be counted.
        reason: Short human-readable reason (``"Valid repetition"`` when valid).
        code: Machine-readable reason code.
        details: All rule violations found (empty when valid).
    """

    valid: bool
    reason: str = ""
    code: str = ""
    details: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-friendly snapshot."""
        return {"valid": self.valid, "reason": self.reason, "code": self.code, "details": list(self.details)}

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.valid


class RepValidator:
    """Applies the invalid-rep rules from ``config.yaml`` (``validation:``)."""

    def __init__(self, config: ValidationConfig, pushup_config: Optional[PushUpConfig] = None) -> None:
        self.config = config
        self.pushup = pushup_config or PushUpConfig()

    def is_valid_pushup(self, cycle: CycleRecord) -> RepValidationResult:
        """Decide whether a completed cycle counts as a repetition.

        Args:
            cycle: Metrics accumulated between UP and the next UP.

        Returns:
            A :class:`RepValidationResult` with the verdict and every violation.
        """
        violations: List[tuple[str, str, str]] = []  # (code, reason, detail)

        min_elbow = cycle.min_elbow_angle
        if not math.isfinite(min_elbow):
            violations.append(("no_data", "Keypoints lost during the movement", "No usable elbow angle."))

        if math.isfinite(min_elbow) and min_elbow > self.config.min_depth_angle:
            violations.append(
                (
                    "insufficient_depth",
                    "Insufficient depth",
                    f"Elbow only reached {min_elbow:.0f}° (needs ≤ {self.config.min_depth_angle:.0f}°).",
                )
            )

        if cycle.elbow_travel < self.pushup.min_elbow_bend:
            violations.append(
                (
                    "insufficient_elbow_bend",
                    "Insufficient elbow bend",
                    f"Elbow travelled {cycle.elbow_travel:.0f}° (needs ≥ {self.pushup.min_elbow_bend:.0f}°).",
                )
            )

        if cycle.duration < self.pushup.min_rep_duration:
            violations.append(
                (
                    "too_fast",
                    "Movement too fast (likely noise)",
                    f"Cycle lasted {cycle.duration:.2f}s (minimum {self.pushup.min_rep_duration:.2f}s).",
                )
            )

        if cycle.duration > self.pushup.max_cycle_duration:
            violations.append(
                (
                    "cycle_timeout",
                    "Movement too slow or abandoned",
                    f"Cycle lasted {cycle.duration:.2f}s (maximum {self.pushup.max_cycle_duration:.2f}s).",
                )
            )

        if cycle.tracking_ratio < self.config.min_tracking_ratio:
            violations.append(
                (
                    "tracking_lost",
                    "Keypoints disappeared during the movement",
                    f"Pose visible in {cycle.tracking_ratio * 100:.0f}% of the cycle.",
                )
            )

        if cycle.max_alignment_deviation > self.config.max_alignment_deviation:
            violations.append(
                (
                    "poor_alignment",
                    "Body alignment broke during the rep",
                    f"Body line deviated {cycle.max_alignment_deviation:.0f}° from straight.",
                )
            )

        if math.isfinite(cycle.max_arm_diff) and cycle.max_arm_diff > self.config.max_arm_asymmetry:
            violations.append(
                (
                    "uneven_arms",
                    "Arms moved unevenly",
                    f"Left/right elbow difference reached {cycle.max_arm_diff:.0f}°.",
                )
            )

        if self.config.max_cadence_rpm > 0 and cycle.duration > 0:
            rpm = 60.0 / cycle.duration
            if rpm > self.config.max_cadence_rpm:
                violations.append(
                    (
                        "cadence_too_high",
                        "Cadence above the configured maximum",
                        f"{rpm:.1f} reps/min > {self.config.max_cadence_rpm:.0f} reps/min.",
                    )
                )

        if violations:
            code, reason, detail = violations[0]
            return RepValidationResult(
                valid=False,
                reason=reason,
                code=code,
                details=[f"{item[1]}: {item[2]}" for item in violations],
            )
        return RepValidationResult(valid=True, reason="Valid repetition", code="valid")


@dataclass(slots=True)
class CounterUpdate:
    """Everything the UI needs after processing one frame."""

    rep_count: int = 0
    valid_reps: int = 0
    invalid_reps: int = 0
    state: ExerciseState = ExerciseState.UNKNOWN
    previous_state: ExerciseState = ExerciseState.UNKNOWN
    raw_position: ExerciseState = ExerciseState.UNKNOWN
    elbow_angle: float = float("nan")
    smoothed_angle: float = float("nan")
    body_alignment: float = float("nan")
    arm_diff: float = float("nan")
    form: Optional[FormResult] = None
    posture: Optional[PostureResult] = None
    movement: Optional[MovementState] = None
    rep_completed: bool = False
    last_validation: Optional[RepValidationResult] = None
    cycle_seconds: float = 0.0
    detection_confidence: float = 0.0
    has_pose: bool = False
    message: str = ""

    @property
    def feedback(self) -> str:
        """Primary coaching message for this frame."""
        if self.posture is not None:
            issue = self.posture.primary_feedback()
            if issue is not None:
                return f"{issue.emoji} {issue.message}"
        return ""


class PushUpCounter:
    """Push-up state machine + repetition counter.

    Example:
        >>> counter = PushUpCounter(PushUpConfig(), ValidationConfig())
        >>> for angle in [170, 160, 120, 95, 85, 80, 85, 95, 120, 160, 170]:
        ...     update = counter.update(angle)
        >>> counter.rep_count
        1
    """

    def __init__(
        self,
        pushup_config: PushUpConfig,
        validation_config: Optional[ValidationConfig] = None,
        movement_analyzer: Optional[MovementAnalyzer] = None,
        form_analyzer: Optional[FormAnalyzer] = None,
        smoothing_config: Optional[SmoothingConfig] = None,
        movement_config: Optional[MovementConfig] = None,
        *,
        up_threshold: Optional[float] = None,
        down_threshold: Optional[float] = None,
        min_stable_frames: Optional[int] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create the counter.

        Args:
            pushup_config: State-machine thresholds.
            validation_config: Invalid-rep rules.
            movement_analyzer: Optional shared movement analyzer (smoothing state).
            form_analyzer: Optional shared form analyzer.
            smoothing_config: Smoothing settings for the shared movement analyzer.
            movement_config: Movement analysis settings (phase timing, velocity).
            up_threshold: Optional override of ``pushup_config.up_threshold``.
            down_threshold: Optional override of ``pushup_config.down_threshold``.
            min_stable_frames: Optional override of ``pushup_config.stable_frames``.
            clock: Time source (injectable for deterministic tests).
        """
        self.pushup_config = pushup_config
        if up_threshold is not None:
            self.pushup_config.up_threshold = float(up_threshold)
        if down_threshold is not None:
            self.pushup_config.down_threshold = float(down_threshold)
        if min_stable_frames is not None:
            self.pushup_config.stable_frames = int(min_stable_frames)

        self.validator = RepValidator(validation_config or ValidationConfig(), self.pushup_config)
        self.movement = movement_analyzer or MovementAnalyzer(
            self.pushup_config,
            smoothing_config or SmoothingConfig(),
            movement_config,
        )
        self.form_analyzer = form_analyzer
        self._clock = clock

        self._rep_count = 0
        self._valid_reps = 0
        self._invalid_reps = 0
        self._state = ExerciseState.UNKNOWN
        self._previous_state = ExerciseState.UNKNOWN
        self._reached_down = False
        self._cycle: Optional[CycleRecord] = None
        self._last_validation: Optional[RepValidationResult] = None
        self._session_started: Optional[float] = None
        self._history: List[dict] = []

    # ---------------------------------------------------------------- properties
    @property
    def rep_count(self) -> int:
        """Total counted (valid) repetitions."""
        return self._rep_count

    @property
    def valid_reps(self) -> int:
        """Alias of :attr:`rep_count` (repetitions that passed validation)."""
        return self._valid_reps

    @property
    def invalid_reps(self) -> int:
        """Attempts that were rejected by the validation rules."""
        return self._invalid_reps

    @property
    def current_state(self) -> ExerciseState:
        """Currently confirmed exercise state."""
        return self._state

    @property
    def previous_state(self) -> ExerciseState:
        """Exercise state of the previous frame."""
        return self._previous_state

    @property
    def current_cycle(self) -> Optional[CycleRecord]:
        """Metrics of the repetition in progress."""
        return self._cycle

    @property
    def last_validation(self) -> Optional[RepValidationResult]:
        """Validation result of the most recent completed cycle."""
        return self._last_validation

    @property
    def history(self) -> List[dict]:
        """One record per completed cycle (valid or not)."""
        return list(self._history)

    # ------------------------------------------------------------------ methods
    def classify_position(self, angle: Optional[float]) -> ExerciseState:
        """Classify an elbow angle with the configured thresholds."""
        return classify_position(angle, self.pushup_config.up_threshold, self.pushup_config.down_threshold)

    def update(
        self,
        elbow_angle: Optional[float],
        posture_data: Optional[object] = None,
        *,
        angles: Optional[AngleSet] = None,
        pose: Optional[PersonPose] = None,
        timestamp: Optional[float] = None,
    ) -> CounterUpdate:
        """Process one frame.

        Args:
            elbow_angle: Raw elbow angle in degrees (``None`` when no pose).
            posture_data: Optional :class:`AngleSet` (legacy positional argument).
            angles: Joint angles for this frame (preferred over ``posture_data``).
            pose: Primary subject, used for the detection confidence readout.
            timestamp: Frame timestamp; defaults to the injected clock.

        Returns:
            A :class:`CounterUpdate` describing the new state.
        """
        if angles is None and isinstance(posture_data, AngleSet):
            angles = posture_data
        angles = angles or AngleSet()
        stamp = self._clock() if timestamp is None else float(timestamp)
        if self._session_started is None:
            self._session_started = stamp

        movement_state = self.movement.update(elbow_angle, stamp)
        self._previous_state = self._state
        self._state = movement_state.position

        # A finite (smoothed) elbow angle means the arm joints were tracked this
        # frame - that is the real "did we see the movement" signal, and it holds
        # whether or not a pose object was supplied (bare-angle callers included).
        angle_tracked = math.isfinite(movement_state.smoothed_angle)
        has_pose = pose is not None or angle_tracked
        self._observe_cycle(angles, movement_state, stamp, angle_tracked)

        transition = self._previous_state != self._state
        rep_completed = False
        if transition:
            rep_completed = self._handle_transition(stamp)

        form_result = self._score(angles, movement_state, pose)
        posture_result = self._posture(angles, movement_state, pose)

        message = ""
        if rep_completed:
            message = (
                f"Rep {self._rep_count} counted"
                if self._last_validation is not None and self._last_validation.valid
                else f"Invalid rep: {self._last_validation.reason if self._last_validation else 'unknown'}"
            )

        return CounterUpdate(
            rep_count=self._rep_count,
            valid_reps=self._valid_reps,
            invalid_reps=self._invalid_reps,
            state=self._state,
            previous_state=self._previous_state,
            raw_position=movement_state.raw_position,
            elbow_angle=angles.primary_elbow(),
            smoothed_angle=movement_state.smoothed_angle,
            body_alignment=angles.body_alignment,
            arm_diff=angles.elbow_diff,
            form=form_result,
            posture=posture_result,
            movement=movement_state,
            rep_completed=rep_completed,
            last_validation=self._last_validation,
            cycle_seconds=self._cycle.duration if self._cycle is not None else 0.0,
            detection_confidence=float(pose.detection_confidence) if pose is not None else 0.0,
            has_pose=has_pose,
            message=message,
        )

    def validate_rep(self, posture_data: object) -> RepValidationResult:
        """Validate a cycle.

        Accepts a :class:`CycleRecord` (preferred) or an :class:`AngleSet`, in
        which case the instantaneous angles are treated as the cycle summary.
        """
        if isinstance(posture_data, CycleRecord):
            return self.validator.is_valid_pushup(posture_data)
        if isinstance(posture_data, AngleSet):
            # A single frame carries no travel information, so the cycle is built
            # from the instantaneous angles only. Such a "cycle" can never pass the
            # elbow-travel rule, which is the correct verdict for a partial rep.
            cycle = CycleRecord()
            cycle.observe(
                posture_data.primary_elbow(),
                posture_data.alignment_deviation,
                posture_data.elbow_diff,
                True,
                0.0,
            )
            cycle.observe(
                posture_data.primary_elbow(),
                posture_data.alignment_deviation,
                posture_data.elbow_diff,
                True,
                self.pushup_config.min_rep_duration,
            )
            return self.validator.is_valid_pushup(cycle.finalize())
        raise TypeError(f"validate_rep expects CycleRecord or AngleSet, got {type(posture_data).__name__}")

    def reset(self, *, full: bool = True) -> None:
        """Reset the counters (full new-session reset by default).

        Args:
            full: When ``True`` (default) also clear the movement state, smoothing
                and session clock - i.e. start a brand-new session. When ``False``
                only the repetition tallies and history are cleared, matching
                :meth:`reset_counter_only`.
        """
        self._rep_count = 0
        self._valid_reps = 0
        self._invalid_reps = 0
        self._last_validation = None
        self._reached_down = False
        self._cycle = None
        self._history.clear()
        if full:
            self._state = ExerciseState.UNKNOWN
            self._previous_state = ExerciseState.UNKNOWN
            self.movement.reset()
            self._session_started = None

    def reset_counter_only(self) -> None:
        """Reset just the rep numbers, keeping the movement state."""
        self._rep_count = 0
        self._valid_reps = 0
        self._invalid_reps = 0
        self._history.clear()
        self._last_validation = None

    def session_duration(self) -> float:
        """Seconds since the first processed frame."""
        if self._session_started is None:
            return 0.0
        return max(0.0, self._clock() - self._session_started)

    def reps_per_minute(self) -> float:
        """Cadence of valid reps over the session so far."""
        duration = self.session_duration()
        if duration <= 1e-6:
            return 0.0
        return self._valid_reps * 60.0 / duration

    # ----------------------------------------------------------------- internals
    def _observe_cycle(
        self,
        angles: AngleSet,
        movement_state: MovementState,
        stamp: float,
        tracked: bool,
    ) -> None:
        if self._cycle is None:
            return
        self._cycle.observe(
            movement_state.smoothed_angle,
            angles.alignment_deviation,
            angles.elbow_diff,
            tracked,
            stamp,
        )

    def _handle_transition(self, stamp: float) -> bool:
        """Advance the repetition FSM; returns ``True`` when a rep was resolved."""
        if self._state == ExerciseState.DOWN and self._previous_state in {ExerciseState.UP, ExerciseState.MOVING}:
            if not self._reached_down:
                self._reached_down = True
                self._cycle = CycleRecord(started_at=stamp)
                logger.debug("Push-up depth detected at %.2fs", stamp)
            return False

        if self._state == ExerciseState.UP and self._previous_state in {ExerciseState.DOWN, ExerciseState.MOVING}:
            if not (self._reached_down and self.pushup_config.require_down_before_up):
                return False
            self._reached_down = False
            cycle = self._cycle or CycleRecord(started_at=stamp)
            cycle.ended_at = max(cycle.ended_at, stamp)
            cycle.frames += 1
            cycle.frames_with_pose += 1
            cycle.finalize()
            validation = self.validate_rep(cycle)
            self._last_validation = validation
            self._history.append({"timestamp": stamp, "validation": validation.to_dict(), "cycle": cycle.to_dict()})
            if validation.valid:
                self._rep_count += 1
                self._valid_reps += 1
                logger.info("Rep %d counted (%.2fs cycle, depth %.0f°)", self._rep_count, cycle.duration, cycle.min_elbow_angle)
            else:
                self._invalid_reps += 1
                logger.info("Invalid rep: %s", validation.reason)
            self._cycle = None
            return True

        return False

    def _score(self, angles: AngleSet, movement_state: MovementState, pose: Optional[PersonPose]) -> Optional[FormResult]:
        if self.form_analyzer is None:
            return None
        return self.form_analyzer.analyze(pose, angles, movement_state, self._cycle)

    def _posture(self, angles: AngleSet, movement_state: MovementState, pose: Optional[PersonPose]) -> Optional[PostureResult]:
        if self.form_analyzer is None:
            return None
        return self.form_analyzer.analyze_posture(pose, angles, movement_state, self._cycle)

    # ------------------------------------------------------------------ summary
    def summary(self) -> dict:
        """Snapshot of the counter state (used by reports and tests)."""
        return {
            "rep_count": self._rep_count,
            "valid_reps": self._valid_reps,
            "invalid_reps": self._invalid_reps,
            "state": self._state.value,
            "previous_state": self._previous_state.value,
            "session_duration_s": round(self.session_duration(), 2),
            "reps_per_minute": round(self.reps_per_minute(), 2),
            "history": list(self._history),
        }


__all__ = ["PushUpCounter", "CounterUpdate", "RepValidator", "RepValidationResult"]

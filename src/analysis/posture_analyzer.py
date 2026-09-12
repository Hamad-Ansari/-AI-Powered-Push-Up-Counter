"""Push-up posture analysis.

Turns raw geometry into *issues* a coach would call out: hips too high, hips
sagging, shallow depth, uneven arms, jerky tempo, lost tracking... Each issue
carries a severity (used for colour coding) and a ready-to-display message.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from src.analysis.angle_calculator import AngleSet
from src.analysis.movement_analyzer import ExerciseState, MovementState
from config.settings import MovementConfig, PostureConfig
from src.pose.person import PersonPose
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Severity(str, Enum):
    """How urgent a posture issue is."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

    @property
    def color_key(self) -> str:
        """Palette key for this severity."""
        return {Severity.INFO: "tracking", Severity.WARNING: "warning", Severity.CRITICAL: "bad"}[self]

    @property
    def emoji(self) -> str:
        """Emoji used by the dashboard feedback panel."""
        return {Severity.INFO: "🔵", Severity.WARNING: "🟡", Severity.CRITICAL: "🔴"}[self]


class IssueCode(str, Enum):
    """Machine-readable posture issue identifiers."""

    GOOD_FORM = "good_form"
    NO_PERSON = "no_person"
    TRACKING_LOST = "tracking_lost"
    HIPS_HIGH = "hips_high"
    HIPS_LOW = "hips_low"
    BODY_MISALIGNED = "body_misaligned"
    SHALLOW_DEPTH = "shallow_depth"
    ARMS_NOT_BENT = "arms_not_bent"
    UNEVEN_ARMS = "uneven_arms"
    EXTEND_ARMS = "extend_arms"
    TOO_FAST = "too_fast"
    HEAD_POSITION = "head_position"


@dataclass(slots=True)
class PostureIssue:
    """A single posture observation."""

    code: IssueCode
    severity: Severity
    message: str
    detail: str = ""

    @property
    def emoji(self) -> str:
        """Severity emoji."""
        return self.severity.emoji

    @property
    def color_key(self) -> str:
        """Palette key for rendering."""
        return self.severity.color_key

    def to_dict(self) -> dict:
        """JSON-friendly snapshot."""
        return {
            "code": self.code.value,
            "severity": self.severity.value,
            "message": self.message,
            "detail": self.detail,
        }


@dataclass(slots=True)
class PostureResult:
    """Complete posture assessment for one frame."""

    issues: List[PostureIssue] = field(default_factory=list)
    alignment_deviation: float = float("nan")
    hip_angle: float = float("nan")
    arm_diff: float = float("nan")
    elbow_angle: float = float("nan")
    depth_angle: float = float("nan")
    position: ExerciseState = ExerciseState.UNKNOWN
    has_pose: bool = False
    missing_joints: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ access
    @property
    def worst_severity(self) -> Severity:
        """Most severe issue currently active."""
        if not self.issues:
            return Severity.INFO
        order = {Severity.INFO: 0, Severity.WARNING: 1, Severity.CRITICAL: 2}
        return max(self.issues, key=lambda issue: order[issue.severity]).severity

    @property
    def is_good(self) -> bool:
        """``True`` when nothing worse than INFO is reported."""
        return self.worst_severity == Severity.INFO

    def codes(self) -> List[IssueCode]:
        """Codes of all active issues."""
        return [issue.code for issue in self.issues]

    def has(self, code: IssueCode) -> bool:
        """Whether a specific issue is active."""
        return code in self.codes()

    def primary_feedback(self) -> Optional[PostureIssue]:
        """The most important issue to show first (``None`` when there are none)."""
        return self.issues[0] if self.issues else None

    def to_dict(self) -> dict:
        """JSON-friendly snapshot."""
        return {
            "issues": [issue.to_dict() for issue in self.issues],
            "worst_severity": self.worst_severity.value,
            "alignment_deviation": _r(self.alignment_deviation),
            "hip_angle": _r(self.hip_angle),
            "arm_diff": _r(self.arm_diff),
            "elbow_angle": _r(self.elbow_angle),
            "position": self.position.value,
            "has_pose": self.has_pose,
            "missing_joints": list(self.missing_joints),
        }


def _r(value: float, digits: int = 2) -> float:
    return round(float(value), digits) if math.isfinite(value) else float("nan")


class PostureAnalyzer:
    """Rule-based posture critic.

    Example:
        >>> analyzer = PostureAnalyzer(PostureConfig())
        >>> result = analyzer.analyze(pose, angles, movement)
        >>> result.issues[0].message
        "Keep your body straight"
    """

    def __init__(self, config: PostureConfig, movement_config: Optional[MovementConfig] = None) -> None:
        self.config = config
        self.movement_config = movement_config or MovementConfig()

    def analyze(
        self,
        pose: Optional[PersonPose],
        angles: AngleSet,
        movement: Optional[MovementState] = None,
        *,
        depth_angle: Optional[float] = None,
    ) -> PostureResult:
        """Assess the current posture.

        Args:
            pose: Primary subject (``None`` when nobody is detected).
            angles: Angles computed for this frame.
            movement: Optional movement snapshot (velocity, phase duration).
            depth_angle: Deepest elbow angle of the ongoing repetition, when known.

        Returns:
            A :class:`PostureResult` with issues ordered by importance.
        """
        result = PostureResult(
            alignment_deviation=angles.alignment_deviation,
            hip_angle=angles.body_alignment,
            arm_diff=angles.elbow_diff,
            elbow_angle=angles.primary_elbow(),
            depth_angle=float(depth_angle) if depth_angle is not None else float("nan"),
            position=movement.position if movement is not None else ExerciseState.UNKNOWN,
            has_pose=pose is not None,
        )

        if pose is None:
            result.issues.append(
                PostureIssue(
                    IssueCode.NO_PERSON,
                    Severity.WARNING,
                    "No person detected - step into the frame",
                    "Get your whole body visible, side-on to the camera.",
                )
            )
            return result

        result.missing_joints = pose.critical_missing()
        if result.missing_joints and not pose.has_minimal_pose():
            result.issues.append(
                PostureIssue(
                    IssueCode.TRACKING_LOST,
                    Severity.CRITICAL,
                    "Tracking lost - keep your arms and hips in frame",
                    "Missing joints: " + ", ".join(name.replace("_", " ") for name in result.missing_joints),
                )
            )
            return result

        self._check_alignment(result)
        self._check_arms(result)
        self._check_tempo(result, movement)

        if not result.issues:
            result.issues.append(
                PostureIssue(
                    IssueCode.GOOD_FORM,
                    Severity.INFO,
                    "Good form! Keep going 💪",
                    "Core tight, body straight, controlled tempo.",
                )
            )
        else:
            result.issues.sort(key=lambda issue: {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}[issue.severity])
        return result

    # ------------------------------------------------------------------ rules
    def _check_alignment(self, result: PostureResult) -> None:
        """Hips too high / hips sagging / general misalignment."""
        deviation = result.alignment_deviation
        if not math.isfinite(deviation):
            result.issues.append(
                PostureIssue(
                    IssueCode.BODY_MISALIGNED,
                    Severity.WARNING,
                    "Keep your body straight",
                    "Hip, shoulder and ankle alignment is not visible.",
                )
            )
            return

        tolerance = self.config.alignment_tolerance
        hip_angle = result.hip_angle
        if math.isfinite(hip_angle) and hip_angle < self.config.hips_high_angle:
            result.issues.append(
                PostureIssue(
                    IssueCode.HIPS_HIGH,
                    Severity.CRITICAL,
                    "Don't raise your hips",
                    f"Hip angle {hip_angle:.0f}° - squeeze your glutes and straighten the body line.",
                )
            )
        elif math.isfinite(hip_angle) and hip_angle > self.config.hips_low_angle:
            result.issues.append(
                PostureIssue(
                    IssueCode.HIPS_LOW,
                    Severity.CRITICAL,
                    "Don't drop your hips",
                    f"Hip angle {hip_angle:.0f}° - brace your core to stop the lower back sagging.",
                )
            )
        elif deviation > tolerance:
            severity = Severity.CRITICAL if deviation > self.config.alignment_max_deviation else Severity.WARNING
            result.issues.append(
                PostureIssue(
                    IssueCode.BODY_MISALIGNED,
                    severity,
                    "Keep your body straight",
                    f"Body line is {deviation:.0f}° away from straight.",
                )
            )

    def _check_arms(self, result: PostureResult) -> None:
        """Depth, elbow flexion and left/right symmetry."""
        elbow = result.elbow_angle
        if math.isfinite(elbow) and elbow > self.config.depth_zero_angle:
            result.issues.append(
                PostureIssue(
                    IssueCode.ARMS_NOT_BENT,
                    Severity.WARNING,
                    "Bend your elbows - lower the chest",
                    f"Elbow angle {elbow:.0f}° is still almost locked out.",
                )
            )
        elif math.isfinite(result.depth_angle) and result.depth_angle > self.config.depth_full_angle + 10.0:
            result.issues.append(
                PostureIssue(
                    IssueCode.SHALLOW_DEPTH,
                    Severity.WARNING,
                    "Lower your chest more",
                    f"Deepest elbow angle this rep: {result.depth_angle:.0f}° (target ≤ {self.config.depth_full_angle:.0f}°).",
                )
            )

        if math.isfinite(elbow) and result.position == ExerciseState.UP and elbow < self.config.depth_zero_angle - 15.0:
            result.issues.append(
                PostureIssue(
                    IssueCode.EXTEND_ARMS,
                    Severity.WARNING,
                    "Extend your arms fully at the top",
                    f"Elbow angle {elbow:.0f}° - lock out before the next rep.",
                )
            )

        if math.isfinite(result.arm_diff) and result.arm_diff > self.config.symmetry_threshold:
            result.issues.append(
                PostureIssue(
                    IssueCode.UNEVEN_ARMS,
                    Severity.WARNING,
                    "Try to move both arms evenly",
                    f"Left/right elbow difference: {result.arm_diff:.0f}°.",
                )
            )

    def _check_tempo(self, result: PostureResult, movement: Optional[MovementState]) -> None:
        """Jerky or implausibly fast movement."""
        if movement is None:
            return
        if movement.is_jerky:
            result.issues.append(
                PostureIssue(
                    IssueCode.TOO_FAST,
                    Severity.WARNING,
                    "Slow down and control the movement",
                    f"Angular velocity {abs(movement.velocity):.0f}°/s.",
                )
            )
        elif (
            movement.phase_duration < self.movement_config.min_phase_duration
            and movement.position == ExerciseState.DOWN
            and abs(movement.velocity) > 240.0
        ):
            result.issues.append(
                PostureIssue(
                    IssueCode.TOO_FAST,
                    Severity.INFO,
                    "Control the descent",
                    "The lowering phase is very short.",
                )
            )

__all__ = ["PostureAnalyzer", "PostureIssue", "PostureResult", "IssueCode", "Severity"]

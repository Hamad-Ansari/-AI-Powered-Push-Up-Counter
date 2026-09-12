"""Form analysis facade: posture issues + weighted 0-100 form score.

:class:`FormAnalyzer` is what the pipeline calls once per frame. It composes the
posture critic (:mod:`src.analysis.posture_analyzer`) and the weighted scorer
(:mod:`src.analysis.form_scorer`) into a single structured result.
"""

from __future__ import annotations

import math
from typing import List, Optional

from src.analysis.angle_calculator import AngleSet
from src.analysis.form_scorer import FormGrade, FormResult, FormScorer, cycle_depth_angle
from src.analysis.movement_analyzer import CycleRecord, MovementState
from src.analysis.posture_analyzer import IssueCode, PostureAnalyzer, PostureIssue, PostureResult
from config.settings import FormConfig, MovementConfig, PostureConfig
from src.pose.person import PersonPose
from src.utils.logger import get_logger

logger = get_logger(__name__)

_ENCOURAGEMENTS = {
    FormGrade.EXCELLENT: "Perfect technique - keep this tempo.",
    FormGrade.GOOD: "Solid work, small refinements only.",
    FormGrade.NEEDS_IMPROVEMENT: "Focus on the cues below.",
    FormGrade.POOR: "Reset, slow down and rebuild the movement.",
}


class FormAnalyzer:
    """Produces the structured form analysis used by the UI and the reports.

    Example:
        >>> analyzer = FormAnalyzer(FormConfig(), PostureConfig())
        >>> result = analyzer.analyze(pose, angles, movement)
        >>> result.form_score
        92.4
        >>> result.form_status
        <FormGrade.EXCELLENT: 'EXCELLENT'>
    """

    def __init__(
        self,
        form_config: FormConfig,
        posture_config: Optional[PostureConfig] = None,
        movement_config: Optional[MovementConfig] = None,
    ) -> None:
        self.posture_config = posture_config or PostureConfig()
        self.movement_config = movement_config or MovementConfig()
        self.posture = PostureAnalyzer(self.posture_config, self.movement_config)
        self.scorer = FormScorer(form_config, self.posture_config)

    def analyze(
        self,
        pose: Optional[PersonPose],
        angles: AngleSet,
        movement: Optional[MovementState] = None,
        cycle: Optional[CycleRecord] = None,
    ) -> FormResult:
        """Run the full form analysis for one frame.

        Args:
            pose: Primary subject (``None`` when nobody is detected).
            angles: Joint angles for this frame.
            movement: Movement snapshot (state, velocity, phase duration).
            cycle: Metrics accumulated for the repetition in progress.

        Returns:
            A :class:`FormResult` with the score, grade and feedback messages.
        """
        depth_angle = cycle_depth_angle(cycle, angles.primary_elbow())
        posture_result = self.posture.analyze(pose, angles, movement, depth_angle=depth_angle)

        depth_component = self.scorer.depth_component(depth_angle)
        alignment_component = self.scorer.alignment_component(angles.alignment_deviation)
        symmetry_component = self.scorer.symmetry_component(angles.elbow_diff)
        control_component = self.scorer.control_component(
            cycle_duration=cycle.duration if cycle is not None and cycle.frames else None,
            movement=movement,
        )

        if pose is None:
            # No pose: keep the last-known shape but zero the measurable parts.
            depth_component = 0.0
            alignment_component = 0.0
            control_component = min(control_component, 0.2)

        score_value, components = self.scorer.score(
            depth=depth_component,
            alignment=alignment_component,
            symmetry=symmetry_component,
            control=control_component,
        )
        grade = self.scorer.classify(score_value)
        feedback = self._build_feedback(posture_result, grade)

        return FormResult(
            form_score=score_value,
            form_status=grade,
            feedback=feedback,
            body_alignment_score=alignment_component * 100.0,
            depth_score=depth_component * 100.0,
            symmetry_score=symmetry_component * 100.0,
            movement_score=control_component * 100.0,
            components=components,
        )

    def analyze_posture(
        self,
        pose: Optional[PersonPose],
        angles: AngleSet,
        movement: Optional[MovementState] = None,
        cycle: Optional[CycleRecord] = None,
    ) -> PostureResult:
        """Expose the posture critic directly (used by the rep validator)."""
        depth_angle = cycle_depth_angle(cycle, angles.primary_elbow())
        return self.posture.analyze(pose, angles, movement, depth_angle=depth_angle)

    # ---------------------------------------------------------------- feedback
    @staticmethod
    def _build_feedback(posture_result: PostureResult, grade: FormGrade) -> List[str]:
        """Render posture issues into display-ready strings."""
        feedback: List[str] = []
        for issue in posture_result.issues:
            feedback.append(f"{issue.emoji} {issue.message}")
        if not feedback:
            feedback.append(f"{grade.emoji} {_ENCOURAGEMENTS[grade]}")
        elif grade in {FormGrade.EXCELLENT, FormGrade.GOOD}:
            feedback.append(f"{grade.emoji} {_ENCOURAGEMENTS[grade]}")
        return feedback


def feedback_from_issue(issue: Optional[PostureIssue]) -> str:
    """Single-line feedback for the video overlay."""
    if issue is None:
        return "Good form"
    return f"{issue.emoji} {issue.message}"


def is_acceptable_form(result: FormResult, minimum: float = 60.0) -> bool:
    """``True`` when the form score reaches the ``minimum`` threshold."""
    return math.isfinite(result.form_score) and result.form_score >= minimum


__all__ = ["FormAnalyzer", "FormResult", "FormGrade", "feedback_from_issue", "is_acceptable_form", "IssueCode"]

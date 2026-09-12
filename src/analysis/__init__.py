"""Analysis layer: angles, smoothing, movement, posture and form scoring."""

from src.analysis.angle_calculator import (
    AngleResult,
    AngleSet,
    angle_from_person,
    body_alignment_score,
    calculate_angle,
    calculate_angle_detailed,
    compute_angles,
    depth_score,
    symmetry_score,
)
from src.analysis.form_analyzer import FormAnalyzer, feedback_from_issue, is_acceptable_form
from src.analysis.form_scorer import FormGrade, FormResult, FormScorer, cycle_depth_angle
from src.analysis.movement_analyzer import (
    CycleRecord,
    ExerciseState,
    MovementAnalyzer,
    MovementDirection,
    MovementState,
    StableStateClassifier,
    classify_position,
)
from src.analysis.posture_analyzer import (
    IssueCode,
    PostureAnalyzer,
    PostureIssue,
    PostureResult,
    Severity,
)
from src.analysis.smoothing import (
    AngleSmoother,
    ExponentialMovingAverageSmoother,
    MovingAverageSmoother,
    PassthroughSmoother,
    create_smoother,
    smooth_series,
)

__all__ = [
    "AngleResult",
    "AngleSet",
    "angle_from_person",
    "body_alignment_score",
    "calculate_angle",
    "calculate_angle_detailed",
    "compute_angles",
    "depth_score",
    "symmetry_score",
    "FormAnalyzer",
    "feedback_from_issue",
    "is_acceptable_form",
    "FormGrade",
    "FormResult",
    "FormScorer",
    "cycle_depth_angle",
    "CycleRecord",
    "ExerciseState",
    "MovementAnalyzer",
    "MovementDirection",
    "MovementState",
    "StableStateClassifier",
    "classify_position",
    "IssueCode",
    "PostureAnalyzer",
    "PostureIssue",
    "PostureResult",
    "Severity",
    "AngleSmoother",
    "ExponentialMovingAverageSmoother",
    "MovingAverageSmoother",
    "PassthroughSmoother",
    "create_smoother",
    "smooth_series",
]

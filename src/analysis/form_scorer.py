"""Form scoring: a 0-100 quality score with configurable weights and bands.

``FORM SCORE = depth * w_d + alignment * w_a + symmetry * w_s + control * w_c``

The weights and the classification bands come from ``config.yaml``
(``form.weights`` / ``form.bands``), never from literals in the code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

from src.analysis.angle_calculator import body_alignment_score, depth_score, symmetry_score
from src.analysis.movement_analyzer import CycleRecord, MovementState
from config.settings import FormConfig, PostureConfig
from src.utils.helpers import clamp
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FormGrade(str, Enum):
    """Classification of the form score."""

    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    NEEDS_IMPROVEMENT = "NEEDS IMPROVEMENT"
    POOR = "POOR"

    @property
    def emoji(self) -> str:
        """Emoji used by the dashboard."""
        return {
            FormGrade.EXCELLENT: "🟢",
            FormGrade.GOOD: "🔵",
            FormGrade.NEEDS_IMPROVEMENT: "🟡",
            FormGrade.POOR: "🔴",
        }[self]

    @property
    def color_key(self) -> str:
        """Palette key for rendering."""
        return {
            FormGrade.EXCELLENT: "good",
            FormGrade.GOOD: "tracking",
            FormGrade.NEEDS_IMPROVEMENT: "warning",
            FormGrade.POOR: "bad",
        }[self]

    @property
    def headline(self) -> str:
        """Short headline shown above the feedback list."""
        return {
            FormGrade.EXCELLENT: "Excellent form",
            FormGrade.GOOD: "Good form",
            FormGrade.NEEDS_IMPROVEMENT: "Needs improvement",
            FormGrade.POOR: "Poor form",
        }[self]


@dataclass(slots=True)
class FormResult:
    """Structured form analysis output.

    Attributes:
        form_score: Weighted total in ``[0, 100]``.
        form_status: :class:`FormGrade` classification.
        feedback: Ordered list of coaching messages (emoji-prefixed).
        body_alignment_score: 0-100 body-line sub-score.
        depth_score: 0-100 depth sub-score.
        symmetry_score: 0-100 arm symmetry sub-score.
        movement_score: 0-100 tempo/control sub-score.
        components: Normalised ``[0, 1]`` sub-scores before weighting.
    """

    form_score: float = 0.0
    form_status: FormGrade = FormGrade.POOR
    feedback: list = field(default_factory=list)
    body_alignment_score: float = 0.0
    depth_score: float = 0.0
    symmetry_score: float = 0.0
    movement_score: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        """JSON-friendly snapshot (reports, REST API, analytics)."""
        return {
            "form_score": round(self.form_score, 1),
            "form_status": self.form_status.value,
            "feedback": list(self.feedback),
            "body_alignment_score": round(self.body_alignment_score, 1),
            "depth_score": round(self.depth_score, 1),
            "symmetry_score": round(self.symmetry_score, 1),
            "movement_score": round(self.movement_score, 1),
            "components": {key: round(value, 4) for key, value in self.components.items()},
        }


class FormScorer:
    """Weighted form scoring and grade classification."""

    def __init__(self, config: FormConfig, posture_config: Optional[PostureConfig] = None) -> None:
        self.config = config
        self.posture_config = posture_config or PostureConfig()

    # ------------------------------------------------------------- components
    def depth_component(self, elbow_angle: float) -> float:
        """Normalised depth score for an elbow angle."""
        return depth_score(elbow_angle, self.posture_config.depth_full_angle, self.posture_config.depth_zero_angle)

    def alignment_component(self, deviation: float) -> float:
        """Normalised body-alignment score for ``|hip_angle - 180|``."""
        return body_alignment_score(
            deviation,
            self.posture_config.alignment_tolerance,
            self.posture_config.alignment_max_deviation,
        )

    def symmetry_component(self, elbow_diff: float) -> float:
        """Normalised arm-symmetry score for a left/right elbow difference."""
        return symmetry_score(elbow_diff, self.posture_config.symmetry_threshold)

    def control_component(
        self,
        cycle_duration: Optional[float] = None,
        movement: Optional[MovementState] = None,
    ) -> float:
        """Normalised movement-control score.

        Combines a tempo term (how close the repetition duration is to the ideal)
        with a smoothness term (penalising very high angular velocity).
        """
        control = self.config.control
        ideal = max(0.5, float(control.get("ideal_cycle_duration", 2.5)))
        minimum = max(0.2, float(control.get("min_cycle_duration", 0.8)))
        jerk_free = max(10.0, float(control.get("jerk_free_velocity", 60.0)))

        duration = cycle_duration if cycle_duration is not None else 0.0
        if movement is not None and duration <= 0.0:
            duration = max(movement.phase_duration, minimum)

        if duration <= 0.0:
            tempo = 0.5  # unknown: neutral
        elif duration < minimum:
            tempo = clamp(duration / minimum, 0.0, 1.0) * 0.4
        elif duration <= ideal:
            tempo = 0.4 + 0.6 * clamp((duration - minimum) / max(1e-6, ideal - minimum), 0.0, 1.0)
        else:
            tempo = max(0.55, 1.0 - clamp((duration - ideal) / (2.0 * ideal), 0.0, 0.45))

        velocity = abs(movement.velocity) if movement is not None else 0.0
        smoothness = 1.0 if velocity <= jerk_free else max(0.2, jerk_free / velocity)
        return clamp(0.7 * tempo + 0.3 * smoothness, 0.0, 1.0)

    # ----------------------------------------------------------------- scoring
    def score(
        self,
        *,
        depth: float,
        alignment: float,
        symmetry: float,
        control: float,
    ) -> tuple[float, Dict[str, float]]:
        """Combine normalised components into the weighted 0-100 score.

        Args:
            depth: Normalised depth score in ``[0, 1]``.
            alignment: Normalised alignment score in ``[0, 1]``.
            symmetry: Normalised symmetry score in ``[0, 1]``.
            control: Normalised control score in ``[0, 1]``.

        Returns:
            ``(total_score, components)`` where ``components`` echoes the inputs.
        """
        weights = self.config.weights
        total_weight = sum(weights.values()) or 1.0
        components = {"depth": depth, "alignment": alignment, "symmetry": symmetry, "control": control}
        total = sum(components[name] * weights.get(name, 0.0) for name in components)
        return clamp(total / total_weight * 100.0, 0.0, 100.0), components

    def classify(self, score_value: float) -> FormGrade:
        """Map a 0-100 score onto a :class:`FormGrade` using the configured bands."""
        bands = self.config.bands
        if score_value >= bands.get("excellent", 90.0):
            return FormGrade.EXCELLENT
        if score_value >= bands.get("good", 75.0):
            return FormGrade.GOOD
        if score_value >= bands.get("needs_improvement", 60.0):
            return FormGrade.NEEDS_IMPROVEMENT
        return FormGrade.POOR


def cycle_depth_angle(cycle: Optional[CycleRecord], fallback: float) -> float:
    """Deepest elbow angle of the ongoing cycle, falling back to the live angle."""
    if cycle is not None and math.isfinite(cycle.min_elbow_angle):
        return cycle.min_elbow_angle
    return fallback


__all__ = ["FormGrade", "FormResult", "FormScorer", "cycle_depth_angle"]

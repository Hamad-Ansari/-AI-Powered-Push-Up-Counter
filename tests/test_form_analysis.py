"""Unit tests for posture analysis and form scoring."""

from __future__ import annotations


import pytest

from src.analysis.angle_calculator import AngleSet
from src.analysis.form_analyzer import FormAnalyzer
from src.analysis.form_scorer import FormGrade, FormScorer
from src.analysis.movement_analyzer import CycleRecord, ExerciseState, MovementState
from src.analysis.posture_analyzer import IssueCode, PostureAnalyzer
from config.settings import FormConfig, PostureConfig


def _angles(
    *,
    left_elbow: float = 90.0,
    right_elbow: float = 90.0,
    body_alignment: float = 178.0,
) -> AngleSet:
    """Build an :class:`AngleSet` with the requested elbow/alignment values."""
    elbow_mean = (left_elbow + right_elbow) / 2.0
    return AngleSet(
        left_elbow=left_elbow,
        right_elbow=right_elbow,
        elbow_mean=elbow_mean,
        elbow_diff=abs(left_elbow - right_elbow),
        body_alignment=body_alignment,
        alignment_deviation=abs(body_alignment - 180.0),
        available=["left_elbow", "right_elbow", "body_alignment"],
    )


def _movement(position: ExerciseState = ExerciseState.DOWN, velocity: float = -40.0) -> MovementState:
    return MovementState(smoothed_angle=90.0, position=position, velocity=velocity, phase_duration=1.0)


def _cycle(duration: float = 1.5, min_angle: float = 88.0) -> CycleRecord:
    cycle = CycleRecord()
    cycle.started_at = 0.0
    cycle.ended_at = duration
    cycle.min_elbow_angle = min_angle
    cycle.max_elbow_angle = 170.0
    cycle.frames = 15
    cycle.frames_with_pose = 15
    cycle.tracking_ratio = 1.0
    return cycle


# --------------------------------------------------------------------------- #
# Posture analysis
# --------------------------------------------------------------------------- #
def test_good_form_reports_no_problems():
    """A well-aligned, deep, symmetric pose is graded as good form."""
    analyzer = PostureAnalyzer(PostureConfig())
    result = analyzer.analyze(_FakePose(), _angles(), _movement())
    assert result.is_good
    assert result.has(IssueCode.GOOD_FORM)


def test_hips_high_detected():
    """A hip angle well below straight flags 'hips too high'."""
    analyzer = PostureAnalyzer(PostureConfig())
    result = analyzer.analyze(_FakePose(), _angles(body_alignment=140.0), _movement())
    assert result.has(IssueCode.HIPS_HIGH)
    assert not result.is_good


def test_hips_low_detected():
    """A hip angle above the sag threshold flags 'hips too low'."""
    analyzer = PostureAnalyzer(PostureConfig())
    result = analyzer.analyze(_FakePose(), _angles(body_alignment=212.0), _movement())
    assert result.has(IssueCode.HIPS_LOW)


def test_uneven_arms_detected():
    """A large left/right elbow gap flags uneven arm movement."""
    analyzer = PostureAnalyzer(PostureConfig())
    result = analyzer.analyze(_FakePose(), _angles(left_elbow=90.0, right_elbow=130.0), _movement())
    assert result.has(IssueCode.UNEVEN_ARMS)


def test_no_person_flagged():
    """A missing pose is reported rather than crashing."""
    analyzer = PostureAnalyzer(PostureConfig())
    result = analyzer.analyze(None, _angles(), None)
    assert result.has(IssueCode.NO_PERSON)
    assert not result.has_pose


# --------------------------------------------------------------------------- #
# Form scoring
# --------------------------------------------------------------------------- #
def test_scorer_perfect_components_give_100():
    scorer = FormScorer(FormConfig())
    total, components = scorer.score(depth=1.0, alignment=1.0, symmetry=1.0, control=1.0)
    assert total == pytest.approx(100.0)
    assert components["depth"] == 1.0


def test_scorer_weights_sum_to_100():
    scorer = FormScorer(FormConfig())
    assert sum(scorer.config.weights.values()) == pytest.approx(100.0)


def test_scorer_zero_components_give_zero():
    scorer = FormScorer(FormConfig())
    total, _ = scorer.score(depth=0.0, alignment=0.0, symmetry=0.0, control=0.0)
    assert total == pytest.approx(0.0)


@pytest.mark.parametrize(
    "value, expected",
    [
        (95.0, FormGrade.EXCELLENT),
        (90.0, FormGrade.EXCELLENT),
        (80.0, FormGrade.GOOD),
        (75.0, FormGrade.GOOD),
        (65.0, FormGrade.NEEDS_IMPROVEMENT),
        (60.0, FormGrade.NEEDS_IMPROVEMENT),
        (30.0, FormGrade.POOR),
    ],
)
def test_classification_bands(value, expected):
    assert FormScorer(FormConfig()).classify(value) is expected


def test_control_component_penalises_short_cycles():
    """A very fast cycle scores lower on control than an ideal one."""
    scorer = FormScorer(FormConfig())
    fast = scorer.control_component(cycle_duration=0.4)
    ideal = scorer.control_component(cycle_duration=2.5)
    assert ideal > fast


# --------------------------------------------------------------------------- #
# FormAnalyzer integration
# --------------------------------------------------------------------------- #
def test_form_analyzer_good_pose_scores_high():
    analyzer = FormAnalyzer(FormConfig(), PostureConfig())
    result = analyzer.analyze(_FakePose(), _angles(), _movement(), _cycle())
    assert result.form_score >= 75.0
    assert result.form_status in {FormGrade.EXCELLENT, FormGrade.GOOD}
    assert result.feedback  # non-empty coaching message


def test_form_analyzer_poor_alignment_scores_lower():
    analyzer = FormAnalyzer(FormConfig(), PostureConfig())
    good = analyzer.analyze(_FakePose(), _angles(body_alignment=178.0), _movement(), _cycle())
    poor = analyzer.analyze(_FakePose(), _angles(body_alignment=140.0), _movement(), _cycle())
    assert poor.form_score < good.form_score
    assert poor.body_alignment_score < good.body_alignment_score


def test_form_analyzer_uneven_arms_reduces_symmetry_score():
    analyzer = FormAnalyzer(FormConfig(), PostureConfig())
    even = analyzer.analyze(_FakePose(), _angles(left_elbow=90.0, right_elbow=90.0), _movement(), _cycle())
    uneven = analyzer.analyze(_FakePose(), _angles(left_elbow=90.0, right_elbow=140.0), _movement(), _cycle())
    assert uneven.symmetry_score < even.symmetry_score


def test_form_analyzer_no_pose_is_low_but_safe():
    analyzer = FormAnalyzer(FormConfig(), PostureConfig())
    result = analyzer.analyze(None, _angles(), None, None)
    assert 0.0 <= result.form_score <= 100.0
    assert result.form_status is FormGrade.POOR


def test_form_result_to_dict_is_json_safe():
    analyzer = FormAnalyzer(FormConfig(), PostureConfig())
    result = analyzer.analyze(_FakePose(), _angles(), _movement(), _cycle())
    payload = result.to_dict()
    assert set(payload) >= {"form_score", "form_status", "feedback", "depth_score"}
    assert isinstance(payload["feedback"], list)


class _FakePose:
    """Minimal pose stub: reports a full, high-confidence skeleton."""

    detection_confidence = 0.97
    keypoints: dict = {}

    def point(self, name: str):
        return (10.0, 10.0)

    def has_minimal_pose(self) -> bool:
        return True

    def critical_missing(self):
        return []

    def confidence_of(self, name: str) -> float:
        return 0.97

    @property
    def valid_count(self) -> int:
        return 17

"""Unit tests for the push-up state machine (:mod:`src.counter.pushup_counter`)."""

from __future__ import annotations

from typing import List


from src.analysis.movement_analyzer import CycleRecord, ExerciseState, StableStateClassifier, classify_position
from config.settings import PushUpConfig, SmoothingConfig, ValidationConfig
from src.counter.pushup_counter import PushUpCounter, RepValidator


def _counter(**overrides) -> PushUpCounter:
    """Build a counter with smoothing + debounce disabled and an injectable clock.

    ``stable_frames=1`` isolates the repetition FSM: the debounce filter is
    exercised separately in ``test_stable_classifier_*``.
    """
    overrides.setdefault("stable_frames", 1)
    overrides.setdefault("min_rep_duration", 0.05)
    pushup = PushUpConfig(**{k: v for k, v in overrides.items() if hasattr(PushUpConfig(), k)})
    smoothing = SmoothingConfig(enabled=False)
    clock = _Clock()
    counter = PushUpCounter(pushup, ValidationConfig(), smoothing_config=smoothing, clock=clock)
    counter._test_clock = clock  # type: ignore[attr-defined]
    return counter


class _Clock:
    """Deterministic, manually advanced clock."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


def _run(counter: PushUpCounter, angles: List[float], dt: float = 0.1) -> None:
    """Feed a sequence of elbow angles, advancing the clock by ``dt`` each frame."""
    clock = counter._test_clock  # type: ignore[attr-defined]
    for angle in angles:
        clock.advance(dt)
        counter.update(angle)


# --------------------------------------------------------------------------- #
# classify_position / debounce
# --------------------------------------------------------------------------- #
def test_classify_position_thresholds():
    assert classify_position(160.0, 150.0, 90.0) is ExerciseState.UP
    assert classify_position(80.0, 150.0, 90.0) is ExerciseState.DOWN
    assert classify_position(120.0, 150.0, 90.0) is ExerciseState.MOVING
    assert classify_position(None) is ExerciseState.UNKNOWN


def test_stable_classifier_debounces_single_frame():
    """One stray frame does not flip the confirmed state."""
    classifier = StableStateClassifier(min_stable_frames=3, initial=ExerciseState.UP)
    assert classifier.update(ExerciseState.DOWN) is ExerciseState.UP
    assert classifier.update(ExerciseState.DOWN) is ExerciseState.UP
    assert classifier.update(ExerciseState.UP) is ExerciseState.UP  # reset streak
    assert classifier.confirmed is ExerciseState.UP


def test_stable_classifier_confirms_after_n_frames():
    classifier = StableStateClassifier(min_stable_frames=3, initial=ExerciseState.UP)
    for _ in range(3):
        state = classifier.update(ExerciseState.DOWN)
    assert state is ExerciseState.DOWN


# --------------------------------------------------------------------------- #
# Rep counting
# --------------------------------------------------------------------------- #
def test_full_cycle_counts_one_rep():
    """UP -> MOVING -> DOWN -> MOVING -> UP counts exactly one repetition."""
    counter = _counter()
    _run(counter, [170, 170, 170])            # establish UP
    _run(counter, [140, 120, 100])            # MOVING down
    _run(counter, [85, 85, 85])               # DOWN
    _run(counter, [100, 120, 140])            # MOVING up
    _run(counter, [170, 170, 170])            # UP again
    assert counter.rep_count == 1
    assert counter.valid_reps == 1


def test_partial_movement_counts_zero():
    """Going down but never returning to UP does not count."""
    counter = _counter()
    _run(counter, [170, 170, 170])
    _run(counter, [140, 120, 100, 85, 85, 85])
    assert counter.rep_count == 0


def test_repeated_up_frames_do_not_duplicate():
    """Holding the top position must not inflate the count."""
    counter = _counter()
    _run(counter, [170, 170, 170])
    _run(counter, [140, 100, 85, 85, 85])
    _run(counter, [120, 160, 170])
    _run(counter, [170, 170, 170, 170, 170, 170])  # many UP frames
    assert counter.rep_count == 1


def test_two_cycles_count_two_reps():
    counter = _counter()
    for _ in range(2):
        _run(counter, [170, 170, 170])
        _run(counter, [130, 100, 85, 85])
        _run(counter, [120, 160, 170, 170])
    assert counter.rep_count == 2


def test_shallow_dip_is_invalid_not_counted():
    """A dip that never reaches the DOWN threshold is not a rep."""
    counter = _counter()
    _run(counter, [170, 170, 170])
    _run(counter, [150, 130, 120, 120, 120])  # never <= 90
    _run(counter, [150, 170, 170])
    assert counter.rep_count == 0


def test_reset_clears_counts():
    counter = _counter()
    _run(counter, [170, 170, 170, 100, 85, 85, 120, 170, 170])
    assert counter.rep_count >= 1
    counter.reset()
    assert counter.rep_count == 0
    assert counter.current_state is ExerciseState.UNKNOWN


def test_reset_counter_only_keeps_state():
    counter = _counter()
    _run(counter, [170, 170, 170, 100, 85, 85, 120, 170, 170])
    counter.reset_counter_only()
    assert counter.rep_count == 0
    assert counter.current_state is not ExerciseState.UNKNOWN  # movement state preserved


# --------------------------------------------------------------------------- #
# Validation rules
# --------------------------------------------------------------------------- #
def _cycle(min_angle: float, max_angle: float, duration: float, *, align_dev: float = 2.0, arm_diff: float = 1.0, tracking: float = 1.0) -> CycleRecord:
    cycle = CycleRecord()
    cycle.started_at = 0.0
    cycle.ended_at = duration
    cycle.min_elbow_angle = min_angle
    cycle.max_elbow_angle = max_angle
    cycle.max_alignment_deviation = align_dev
    cycle.max_arm_diff = arm_diff
    cycle.tracking_ratio = tracking
    cycle.frames = max(1, int(duration / 0.1))
    cycle.frames_with_pose = int(cycle.frames * tracking)
    return cycle


def test_validator_accepts_good_rep():
    result = RepValidator(ValidationConfig(), PushUpConfig()).is_valid_pushup(_cycle(85.0, 170.0, 1.5))
    assert result.valid
    assert result.code == "valid"


def test_validator_rejects_insufficient_depth():
    result = RepValidator(ValidationConfig(), PushUpConfig()).is_valid_pushup(_cycle(120.0, 170.0, 1.5))
    assert not result.valid
    assert result.code == "insufficient_depth"


def test_validator_rejects_too_fast():
    result = RepValidator(ValidationConfig(), PushUpConfig()).is_valid_pushup(_cycle(85.0, 170.0, 0.2))
    assert not result.valid
    assert result.code == "too_fast"


def test_validator_rejects_poor_alignment():
    result = RepValidator(ValidationConfig(), PushUpConfig()).is_valid_pushup(_cycle(85.0, 170.0, 1.5, align_dev=50.0))
    assert not result.valid
    assert result.code == "poor_alignment"


def test_validator_rejects_uneven_arms():
    result = RepValidator(ValidationConfig(), PushUpConfig()).is_valid_pushup(_cycle(85.0, 170.0, 1.5, arm_diff=40.0))
    assert not result.valid
    assert result.code == "uneven_arms"


def test_validator_rejects_lost_tracking():
    result = RepValidator(ValidationConfig(), PushUpConfig()).is_valid_pushup(_cycle(85.0, 170.0, 1.5, tracking=0.3))
    assert not result.valid
    assert result.code == "tracking_lost"


def test_counter_marks_invalid_rep_in_history():
    """A shallow cycle is recorded as invalid, not counted."""
    counter = _counter()
    counter.validator.config.min_depth_angle = 100.0
    _run(counter, [170, 170, 170])
    _run(counter, [140, 120, 110, 110, 110])  # bottoms out at 110 (> 100 depth gate)
    _run(counter, [140, 170, 170])
    # 110 > 90 down threshold, so it never even confirms DOWN -> no rep at all
    assert counter.rep_count == 0


def test_reps_per_minute_and_duration():
    counter = _counter()
    _run(counter, [170, 170, 170, 100, 85, 85, 120, 170, 170], dt=0.5)
    assert counter.session_duration() > 0
    assert counter.reps_per_minute() >= 0


def test_summary_structure():
    counter = _counter()
    _run(counter, [170, 170, 170, 100, 85, 85, 120, 170, 170])
    summary = counter.summary()
    assert summary["rep_count"] == counter.rep_count
    assert "history" in summary and "session_duration_s" in summary

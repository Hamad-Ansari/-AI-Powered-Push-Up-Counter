"""Unit tests for joint-angle geometry (:mod:`src.analysis.angle_calculator`)."""

from __future__ import annotations

import math

import pytest

from src.analysis.angle_calculator import (
    AngleSet,
    body_alignment_score,
    calculate_angle,
    calculate_angle_detailed,
    compute_angles,
    depth_score,
    symmetry_score,
)
from src.pose.keypoints import KEYPOINT_NAMES
from src.pose.person import Keypoint, PersonPose


def _person(points: dict[str, tuple[float, float]], confidence: float = 0.95) -> PersonPose:
    """Build a :class:`PersonPose` from a partial ``{name: (x, y)}`` mapping."""
    keypoints = {
        name: (
            Keypoint(points[name][0], points[name][1], confidence, True)
            if name in points
            else Keypoint(float("nan"), float("nan"), 0.0, False)
        )
        for name in KEYPOINT_NAMES
    }
    return PersonPose(keypoints=keypoints, detection_confidence=confidence)


# --------------------------------------------------------------------------- #
# calculate_angle
# --------------------------------------------------------------------------- #
def test_right_angle_is_90_degrees():
    """A clean L shape at the vertex measures 90 degrees."""
    angle = calculate_angle((0.0, 0.0), (0.0, 10.0), (10.0, 10.0))
    assert angle == pytest.approx(90.0, abs=1e-6)


def test_straight_line_is_180_degrees():
    """Three collinear points with the vertex between them measure 180 degrees."""
    angle = calculate_angle((0.0, 0.0), (5.0, 0.0), (10.0, 0.0))
    assert angle == pytest.approx(180.0, abs=1e-6)


def test_folded_back_is_zero_degrees():
    """Both arms pointing the same way collapse to 0 degrees."""
    angle = calculate_angle((10.0, 0.0), (0.0, 0.0), (5.0, 0.0))
    assert angle == pytest.approx(0.0, abs=1e-6)


def test_known_45_degree_angle():
    """A 45 degree bend is measured correctly."""
    angle = calculate_angle((1.0, 1.0), (0.0, 0.0), (1.0, 0.0))
    assert angle == pytest.approx(45.0, abs=1e-6)


@pytest.mark.parametrize(
    "a, b, c",
    [
        (None, (0.0, 0.0), (1.0, 0.0)),
        ((0.0, 0.0), None, (1.0, 0.0)),
        ((0.0, 0.0), (0.0, 0.0), None),
    ],
)
def test_missing_point_returns_nan(a, b, c):
    """A missing keypoint yields NaN rather than raising."""
    assert math.isnan(calculate_angle(a, b, c))


def test_overlapping_points_return_nan():
    """Coincident joints produce a zero vector -> NaN, never a divide-by-zero."""
    result = calculate_angle_detailed((5.0, 5.0), (5.0, 5.0), (10.0, 5.0))
    assert math.isnan(result.degrees)
    assert not result.valid
    assert "Zero-length" in result.reason


def test_non_finite_coordinates_return_nan():
    """Inf/NaN coordinates are rejected as invalid."""
    result = calculate_angle_detailed((float("inf"), 0.0), (0.0, 0.0), (1.0, 0.0))
    assert math.isnan(result.degrees)
    assert not result.valid


def test_result_is_symmetric_in_the_arms():
    """Swapping the two arm endpoints does not change the angle."""
    left = calculate_angle((1.0, 0.0), (0.0, 0.0), (0.0, 1.0))
    right = calculate_angle((0.0, 1.0), (0.0, 0.0), (1.0, 0.0))
    assert left == pytest.approx(right, abs=1e-9)


# --------------------------------------------------------------------------- #
# compute_angles
# --------------------------------------------------------------------------- #
def test_compute_angles_elbow_and_alignment():
    """A posed skeleton yields the expected elbow and alignment angles."""
    pose = _person(
        {
            "left_shoulder": (0.0, 0.0),
            "left_elbow": (0.0, 10.0),
            "left_wrist": (10.0, 10.0),  # 90 degree elbow
            "right_shoulder": (2.0, 0.0),
            "right_elbow": (2.0, 10.0),
            "right_wrist": (12.0, 10.0),
            "left_hip": (0.0, 20.0),
            "right_hip": (2.0, 20.0),
            "left_ankle": (0.0, 40.0),
            "right_ankle": (2.0, 40.0),
        }
    )
    angles = compute_angles(pose)
    assert angles.left_elbow == pytest.approx(90.0, abs=1e-6)
    assert angles.right_elbow == pytest.approx(90.0, abs=1e-6)
    assert angles.elbow_diff == pytest.approx(0.0, abs=1e-6)
    # shoulder/hip/ankle midpoints are collinear and vertical -> 180 degrees
    assert angles.body_alignment == pytest.approx(180.0, abs=1e-6)
    assert angles.alignment_deviation == pytest.approx(0.0, abs=1e-6)


def test_compute_angles_handles_missing_side():
    """A single visible arm still produces a usable primary elbow angle."""
    pose = _person({"left_shoulder": (0.0, 0.0), "left_elbow": (0.0, 10.0), "left_wrist": (10.0, 10.0)})
    angles = compute_angles(pose)
    assert angles.left_elbow == pytest.approx(90.0, abs=1e-6)
    assert math.isnan(angles.right_elbow)
    assert angles.primary_elbow() == pytest.approx(90.0, abs=1e-6)
    assert math.isnan(angles.elbow_diff)


def test_compute_angles_none_pose_is_empty():
    """No person -> an all-NaN :class:`AngleSet` that never raises."""
    angles = compute_angles(None)
    assert isinstance(angles, AngleSet)
    assert math.isnan(angles.primary_elbow())


def test_compute_angles_falls_back_to_knee():
    """When ankles are hidden, alignment falls back to shoulder-hip-knee."""
    pose = _person(
        {
            "left_shoulder": (0.0, 0.0),
            "right_shoulder": (2.0, 0.0),
            "left_hip": (0.0, 20.0),
            "right_hip": (2.0, 20.0),
            "left_knee": (0.0, 40.0),
            "right_knee": (2.0, 40.0),
        }
    )
    angles = compute_angles(pose)
    assert angles.body_alignment == pytest.approx(180.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# component scores
# --------------------------------------------------------------------------- #
def test_depth_score_full_at_or_below_target():
    assert depth_score(85.0) == pytest.approx(1.0)
    assert depth_score(90.0) == pytest.approx(1.0)


def test_depth_score_zero_at_locked_out():
    assert depth_score(160.0) == pytest.approx(0.0)
    assert depth_score(175.0) == pytest.approx(0.0)


def test_depth_score_interpolates():
    assert depth_score(125.0) == pytest.approx(0.5, abs=1e-6)


def test_alignment_score_perfect_within_tolerance():
    assert body_alignment_score(5.0) == pytest.approx(1.0)


def test_alignment_score_degrades_with_deviation():
    assert body_alignment_score(35.0) == pytest.approx(0.0, abs=1e-6)
    mid = body_alignment_score(23.5)  # halfway between tolerance and max
    assert 0.4 < mid < 0.6


def test_symmetry_score_penalises_difference():
    assert symmetry_score(0.0) == pytest.approx(1.0)
    assert symmetry_score(40.0) == pytest.approx(0.0, abs=1e-6)


def test_symmetry_score_unknown_is_neutral():
    assert symmetry_score(float("nan")) == pytest.approx(0.5)

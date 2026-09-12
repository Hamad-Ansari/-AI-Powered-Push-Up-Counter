"""Dependency-free synthetic pose source (demo mode, tests, CI).

This adapter implements exactly the same contract as the RF-DETR adapter, so the
whole pipeline - angles, state machine, form scoring, overlay, export - can be
exercised without torch, without a GPU and without downloading weights.

It generates a *geometrically consistent* side-view push-up: ankle/knee/hip/
shoulder lie on one body axis, the wrists stay planted on the floor and the
elbows bend as the shoulder descends. The elbow angle therefore really does go
from ~178 deg (UP) to ~88 deg (DOWN), which is what makes it a valid test bench
for the counter and the form scorer.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.pose.base import BasePoseAdapter
from src.pose.keypoints import KEYPOINT_NAMES
from src.pose.person import Keypoint, PersonPose
from src.utils.logger import get_logger

logger = get_logger(__name__)

#: Elbow angle (deg) at the extremes of the movement.
_UP_ELBOW_DEG = 176.0
_DOWN_ELBOW_DEG = 88.0
#: Where each joint sits along the shoulder->ankle plank (shoulder = 0).
_PLANK = {"shoulder": 0.00, "hip": 0.44, "knee": 0.74, "ankle": 1.00}
_HEAD_LEN = 0.13
_SIDE_SPREAD = 0.05  # left/right separation in a 2D side view (fraction of plank)


def _lerp(a: Tuple[float, float], b: Tuple[float, float], t: float) -> Tuple[float, float]:
    """Linear interpolation between two points."""
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _solve_elbow(
    shoulder: Tuple[float, float],
    wrist: Tuple[float, float],
    arm_length: float,
    bend_sign: float,
) -> Tuple[float, float]:
    """Two-link inverse kinematics: elbow for a fixed upper/forearm length.

    The shoulder and wrist are pinned, so the elbow lies on the perpendicular
    bisector at ``sqrt(L^2 - (|SW|/2)^2)``. ``bend_sign`` picks which side the
    elbow bows to (towards the floor for a push-up).
    """
    mid = ((shoulder[0] + wrist[0]) / 2.0, (shoulder[1] + wrist[1]) / 2.0)
    half_base = math.dist(shoulder, wrist) / 2.0
    half_base = min(half_base, arm_length)  # clamp so sqrt stays real
    height = math.sqrt(max(0.0, arm_length * arm_length - half_base * half_base))
    dx, dy = wrist[0] - shoulder[0], wrist[1] - shoulder[1]
    length = math.hypot(dx, dy) or 1e-6
    # Unit perpendicular; bend_sign selects the floor-ward elbow.
    perp = (-dy / length, dx / length)
    return (mid[0] + bend_sign * height * perp[0], mid[1] + bend_sign * height * perp[1])


def generate_pushup_keypoints(
    phase: float,
    *,
    hips_offset: float = 0.0,
    arm_asymmetry: float = 0.0,
    frame_shape: Tuple[int, int] = (720, 1280),
    person_scale: float = 0.62,
    seed: Optional[int] = None,
    jitter: float = 0.0015,
) -> Dict[str, Tuple[float, float]]:
    """Build a geometrically consistent 17-keypoint side-view push-up pose.

    The elbow angle is the driver: shoulder height above the floor follows from
    the two-link arm geometry (``|shoulder-wrist| = 2L sin(elbow/2)``), so the
    joints the counter reads really do traverse UP -> DOWN -> UP.

    Args:
        phase: Movement phase, ``0.0`` = arms locked out (UP), ``1.0`` = chest at
            the floor (DOWN).
        hips_offset: Signed hip deflection off the plank. Positive lifts the hips
            (pike), negative lets them sag - exercises the posture analyzer.
        arm_asymmetry: Extra elbow flexion applied to the right arm, in degrees.
        frame_shape: ``(height, width)`` of the frame the pose is rendered into.
        person_scale: Person length as a fraction of the frame width.
        seed: Optional RNG seed for reproducible jitter.
        jitter: Sub-pixel noise added to every joint (fraction of frame width).

    Returns:
        Mapping of canonical keypoint name -> ``(x, y)`` pixel coordinates.
    """
    rng = np.random.default_rng(seed)
    height, width = float(frame_shape[0]), float(frame_shape[1])
    phase = float(min(1.0, max(0.0, phase)))

    floor_y = height * 0.80
    center_x = width * 0.50
    plank = width * float(person_scale) * 0.82
    arm_length = plank * 0.30

    # Hands planted on the floor, feet planted on the floor, head to the left.
    wrist_base = (center_x - plank * 0.42, floor_y)
    ankle_base = (center_x + plank * 0.46, floor_y)

    # Elbow angle drives everything: shoulder height = 2L sin(elbow/2).
    points: Dict[str, Tuple[float, float]] = {}
    shoulder_ref: Optional[Tuple[float, float]] = None
    hip_ref: Optional[Tuple[float, float]] = None
    for side, lateral, extra_deg in (("left", -1.0, 0.0), ("right", 1.0, float(arm_asymmetry))):
        elbow_deg = _UP_ELBOW_DEG + (_DOWN_ELBOW_DEG - _UP_ELBOW_DEG) * phase + extra_deg
        elbow_deg = max(20.0, min(178.0, elbow_deg))
        shoulder_height = 2.0 * arm_length * math.sin(math.radians(elbow_deg) / 2.0)
        spread = lateral * plank * _SIDE_SPREAD
        shoulder = (wrist_base[0] + plank * 0.03 + spread * 0.2, floor_y - shoulder_height)
        wrist = (wrist_base[0] + spread, wrist_base[1])
        # Elbow bows toward the floor (down) and slightly forward.
        elbow = _solve_elbow(shoulder, wrist, arm_length, bend_sign=1.0 if lateral < 0 else 1.0)
        points[f"{side}_shoulder"] = shoulder
        points[f"{side}_elbow"] = elbow
        points[f"{side}_wrist"] = wrist
        if side == "left":
            shoulder_ref = shoulder

    # Plank from shoulder to ankle; hips/knees interpolated along it.
    shoulder_mid = shoulder_ref or (wrist_base[0], floor_y - 2 * arm_length)
    hip_normal = (0.0, -1.0)  # +hips_offset lifts the hips off the plank
    for side, lateral in (("left", -1.0), ("right", 1.0)):
        spread = lateral * plank * _SIDE_SPREAD * 0.5
        ankle = (ankle_base[0] + spread, ankle_base[1])
        hip = _lerp(shoulder_mid, ankle, _PLANK["hip"])
        hip = (hip[0], hip[1] + float(hips_offset) * plank * hip_normal[1])
        knee = _lerp(shoulder_mid, ankle, _PLANK["knee"])
        knee = (knee[0], knee[1] + float(hips_offset) * plank * 0.35 * hip_normal[1])
        points[f"{side}_hip"] = hip
        points[f"{side}_knee"] = knee
        points[f"{side}_ankle"] = ankle
        if side == "left":
            hip_ref = hip

    # Head: nose beyond the shoulder, eyes/ears behind it along the plank axis.
    plank_dir = _unit_dir(shoulder_mid, hip_ref or ankle_base)
    head_dir = (-plank_dir[0], -plank_dir[1])
    head_len = plank * _HEAD_LEN
    nose = (shoulder_mid[0] + head_dir[0] * head_len, shoulder_mid[1] + head_dir[1] * head_len - plank * 0.03)
    points["nose"] = nose
    for name, frac, lateral in (
        ("left_eye", 0.62, -1.0),
        ("right_eye", 0.62, 1.0),
        ("left_ear", 0.40, -1.0),
        ("right_ear", 0.40, 1.0),
    ):
        offset = lateral * plank * 0.035
        points[name] = (
            shoulder_mid[0] + head_dir[0] * head_len * frac + offset * 0.3,
            shoulder_mid[1] + head_dir[1] * head_len * frac - plank * 0.045 + offset * 0.15,
        )

    if jitter > 0:
        noise = rng.normal(0.0, jitter * width, size=(len(KEYPOINT_NAMES), 2))
        for index, name in enumerate(KEYPOINT_NAMES):
            x, y = points[name]
            points[name] = (x + float(noise[index, 0]), y + float(noise[index, 1]))
    return points


def _unit_dir(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    """Unit vector pointing from ``a`` to ``b``."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy) or 1e-6
    return (dx / length, dy / length)


class SyntheticPushUpAdapter(BasePoseAdapter):
    """Generates a deterministic push-up pose stream.

    Args:
        cycle_seconds: Duration of one full UP -> DOWN -> UP repetition.
        fps: Assumed frame rate of the stream.
        noise_frames: Frame index at which the pose is dropped once, to exercise
            the "person leaves the frame" code path.
        hips_offset: Constant hip deflection (posture fault injection).
        arm_asymmetry: Constant left/right elbow mismatch in degrees.
    """

    name = "demo"

    def __init__(
        self,
        *,
        cycle_seconds: float = 3.0,
        fps: float = 25.0,
        noise_frames: Optional[int] = None,
        hips_offset: float = 0.0,
        arm_asymmetry: float = 0.0,
        seed: int = 42,
        model_name: str = "synthetic-pushup",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, **kwargs)
        self.cycle_seconds = float(cycle_seconds)
        self.fps = float(fps)
        self.noise_frames = noise_frames
        self.hips_offset = float(hips_offset)
        self.arm_asymmetry = float(arm_asymmetry)
        self.seed = int(seed)

    def load(self) -> None:
        """Nothing to download - mark the backend ready."""
        self._loaded = True
        self.status = self._make_status(device="cpu")
        logger.info("Synthetic push-up adapter ready (cycle=%.2fs)", self.cycle_seconds)

    def phase_for_frame(self, frame_id: int) -> float:
        """Movement phase in ``[0, 1]`` for a frame index (smooth cosine cycle)."""
        frames_per_cycle = max(2.0, self.cycle_seconds * self.fps)
        cycle_position = (frame_id % frames_per_cycle) / frames_per_cycle
        return 0.5 - 0.5 * math.cos(2.0 * math.pi * cycle_position)

    def infer(self, frame_bgr: np.ndarray) -> List[PersonPose]:
        """Return one synthetic person for the current frame."""
        frame_id = max(0, self._frame_id)
        if self.noise_frames is not None and frame_id == self.noise_frames:
            return []

        height, width = frame_bgr.shape[:2]
        phase = self.phase_for_frame(frame_id)
        raw = generate_pushup_keypoints(
            phase,
            hips_offset=self.hips_offset,
            arm_asymmetry=self.arm_asymmetry,
            frame_shape=(height, width),
            seed=self.seed + frame_id,
        )

        keypoints = {
            name: Keypoint(x=float(point[0]), y=float(point[1]), confidence=0.94, valid=True)
            for name, point in raw.items()
        }
        xs = [kp.x for kp in keypoints.values()]
        ys = [kp.y for kp in keypoints.values()]
        pad_x = width * 0.05
        pad_y = height * 0.05
        bbox = (min(xs) - pad_x, min(ys) - pad_y, max(xs) + pad_x, max(ys) + pad_y)
        return [
            PersonPose(
                keypoints=keypoints,
                detection_confidence=0.97,
                bbox=bbox,
                frame_id=frame_id,
            )
        ]


__all__ = ["SyntheticPushUpAdapter", "generate_pushup_keypoints"]

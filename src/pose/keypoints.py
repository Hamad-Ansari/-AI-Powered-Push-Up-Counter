"""Canonical COCO-17 keypoint schema and skeleton topology.

Every pose backend in this project is normalised onto *these* names before any
analysis happens, so no model-specific index ever leaks into the analysis,
counter or visualization layers.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Tuple


class KeypointName(str, Enum):
    """The 17 COCO keypoints, in model index order."""

    NOSE = "nose"
    LEFT_EYE = "left_eye"
    RIGHT_EYE = "right_eye"
    LEFT_EAR = "left_ear"
    RIGHT_EAR = "right_ear"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"


#: Canonical index order (identical to the COCO order used by RF-DETR Keypoint).
KEYPOINT_NAMES: Tuple[str, ...] = tuple(member.value for member in KeypointName)

#: Name -> index, for adapters that receive named keypoints.
KEYPOINT_INDEX: Dict[str, int] = {name: index for index, name in enumerate(KEYPOINT_NAMES)}

NUM_KEYPOINTS: int = len(KEYPOINT_NAMES)

#: Joints that matter for push-up analysis (drawn larger and used for feedback).
CRITICAL_KEYPOINTS: Tuple[str, ...] = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_ankle",
    "right_ankle",
)

#: Minimum keypoints that must be valid for the frame to be analysable.
MIN_VALID_FOR_ANALYSIS: Tuple[str, ...] = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)


class Limb(str, Enum):
    """Skeleton edge groups, used to colour the overlay by body region."""

    ARM = "arm"
    TORSO = "torso"
    LEG = "leg"
    HEAD = "head"


#: Skeleton topology as ``(from_name, to_name, limb)`` triples.
SKELETON_CONNECTIONS: Tuple[Tuple[str, str, Limb], ...] = (
    # Head
    ("nose", "left_eye", Limb.HEAD),
    ("nose", "right_eye", Limb.HEAD),
    ("left_eye", "left_ear", Limb.HEAD),
    ("right_eye", "right_ear", Limb.HEAD),
    # Torso
    ("left_shoulder", "right_shoulder", Limb.TORSO),
    ("left_shoulder", "left_hip", Limb.TORSO),
    ("right_shoulder", "right_hip", Limb.TORSO),
    ("left_hip", "right_hip", Limb.TORSO),
    # Arms
    ("left_shoulder", "left_elbow", Limb.ARM),
    ("left_elbow", "left_wrist", Limb.ARM),
    ("right_shoulder", "right_elbow", Limb.ARM),
    ("right_elbow", "right_wrist", Limb.ARM),
    # Legs
    ("left_hip", "left_knee", Limb.LEG),
    ("left_knee", "left_ankle", Limb.LEG),
    ("right_hip", "right_knee", Limb.LEG),
    ("right_knee", "right_ankle", Limb.LEG),
)

#: Per-limb base colours (BGR) used by the skeleton renderer.
LIMB_COLORS: Dict[Limb, Tuple[int, int, int]] = {
    Limb.ARM: (80, 220, 255),      # amber-ish
    Limb.TORSO: (255, 180, 90),    # cyan-ish
    Limb.LEG: (150, 255, 150),     # green-ish
    Limb.HEAD: (230, 230, 230),    # near white
}

#: Mirror pairs, used by symmetry analysis and horizontal-flip augmentation.
FLIP_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("left_eye", "right_eye"),
    ("left_ear", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
)


def skeleton_pairs() -> List[Tuple[int, int, Limb]]:
    """Skeleton connections expressed as index triples (fast path for drawing)."""
    return [(KEYPOINT_INDEX[a], KEYPOINT_INDEX[b], limb) for a, b, limb in SKELETON_CONNECTIONS]


def validate_schema() -> None:
    """Sanity-check the schema (called by the test-suite)."""
    assert NUM_KEYPOINTS == 17, "COCO schema must have exactly 17 keypoints"
    assert len(set(KEYPOINT_NAMES)) == NUM_KEYPOINTS, "Duplicate keypoint names"
    for a, b, _ in SKELETON_CONNECTIONS:
        assert a in KEYPOINT_INDEX and b in KEYPOINT_INDEX, f"Unknown joint in edge {a}-{b}"

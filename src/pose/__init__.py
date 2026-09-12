"""Pose detection layer: backends, keypoint schema and the detector facade."""

from src.pose.base import BasePoseAdapter, PoseBackendError, adapter_from_config
from src.pose.keypoints import (
    CRITICAL_KEYPOINTS,
    KEYPOINT_INDEX,
    KEYPOINT_NAMES,
    LIMB_COLORS,
    NUM_KEYPOINTS,
    SKELETON_CONNECTIONS,
    KeypointName,
    Limb,
    skeleton_pairs,
    validate_schema,
)
from src.pose.person import DetectorStatus, Keypoint, PersonPose, PoseFrame, empty_pose
from src.pose.pose_detector import PoseDetector, primary_points
from src.pose.subject_tracker import SubjectTracker

__all__ = [
    "BasePoseAdapter",
    "PoseBackendError",
    "adapter_from_config",
    "CRITICAL_KEYPOINTS",
    "KEYPOINT_INDEX",
    "KEYPOINT_NAMES",
    "LIMB_COLORS",
    "NUM_KEYPOINTS",
    "SKELETON_CONNECTIONS",
    "KeypointName",
    "Limb",
    "skeleton_pairs",
    "validate_schema",
    "DetectorStatus",
    "Keypoint",
    "PersonPose",
    "PoseFrame",
    "empty_pose",
    "PoseDetector",
    "primary_points",
    "SubjectTracker",
]

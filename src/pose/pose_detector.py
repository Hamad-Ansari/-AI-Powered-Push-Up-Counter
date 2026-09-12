"""High-level pose detection facade.

:class:`PoseDetector` is the *only* class the analysis pipeline talks to. It owns
the backend adapter, normalises keypoints, applies the confidence gate, selects
the primary subject and exposes a stable, model-agnostic API.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from config.settings import Settings
from src.pose.base import BasePoseAdapter, PoseBackendError, adapter_from_config
from src.pose.keypoints import KEYPOINT_NAMES, NUM_KEYPOINTS
from src.pose.person import DetectorStatus, Keypoint, PersonPose, PoseFrame
from src.pose.subject_tracker import SubjectTracker
from src.utils.logger import get_logger

logger = get_logger(__name__)

Point = Tuple[float, float]


class PoseDetector:
    """Pose detection facade with standardized output.

    Example:
        >>> detector = PoseDetector(settings)
        >>> detector.load_model()
        >>> pose = detector.detect(frame_bgr)
        >>> pose.point("left_elbow")
        (412.5, 388.0)
    """

    def __init__(
        self,
        settings: Settings,
        adapter: Optional[BasePoseAdapter] = None,
        *,
        model_name: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
    ) -> None:
        """Create the facade.

        Args:
            settings: Application settings (model + person-selection sections).
            adapter: Optional pre-built adapter (tests / custom backends).
            model_name: Optional override of ``settings.model.name``.
            confidence_threshold: Optional override of the keypoint gate.
        """
        self.settings = settings
        self.model_name = model_name or settings.model.name
        self.confidence_threshold = (
            float(confidence_threshold)
            if confidence_threshold is not None
            else float(settings.model.confidence_threshold)
        )
        self.detection_threshold = float(settings.model.detection_threshold)
        self.adapter: BasePoseAdapter = adapter or adapter_from_config(settings)
        self.tracker = SubjectTracker(settings.person_selection)
        self._last_error: str = ""
        self._frames_processed: int = 0
        self._frames_with_pose: int = 0

    # ------------------------------------------------------------------ model
    def load_model(self) -> None:
        """Load the underlying model weights.

        Raises:
            PoseBackendError: If the backend cannot be initialised.
        """
        self.adapter.load()
        logger.info("Pose backend ready: %s", self.adapter.describe())

    @property
    def status(self) -> DetectorStatus:
        """Backend health information for the UI."""
        return self.adapter.status

    @property
    def is_loaded(self) -> bool:
        """``True`` once the model is ready to infer."""
        return self.adapter.is_loaded()

    def release(self) -> None:
        """Release backend resources."""
        self.adapter.release()
        self.tracker.reset()

    # ----------------------------------------------------------------- detect
    def detect(self, frame_bgr: np.ndarray) -> Optional[PersonPose]:
        """Detect the primary subject in a BGR frame.

        Args:
            frame_bgr: ``HxWx3`` BGR image.

        Returns:
            The tracked primary :class:`PersonPose`, or ``None`` when nobody is
            detectable in this frame.
        """
        return self.detect_all(frame_bgr).primary

    def detect_all(self, frame_bgr: np.ndarray) -> PoseFrame:
        """Detect every person in the frame and pick the primary subject.

        Handles the documented failure modes (corrupted frame, no person,
        multiple people) without raising.
        """
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.ndim != 3 or frame_bgr.size == 0:
            self._last_error = "Invalid frame received from the video source"
            logger.warning(self._last_error)
            return PoseFrame(frame_id=self._frames_processed)

        pose_frame = self.adapter.predict(frame_bgr)
        validated = [
            person for person in (self.validate_keypoints(person) for person in pose_frame.people)
            if person.has_minimal_pose()
        ]
        pose_frame.people = validated
        pose_frame.multiple_people = len(validated) > 1

        height, width = frame_bgr.shape[:2]
        pose_frame.primary = self.tracker.select(validated, (height, width))
        if pose_frame.primary is not None:
            pose_frame.primary.frame_id = pose_frame.frame_id

        self._frames_processed += 1
        if pose_frame.primary is not None:
            self._frames_with_pose += 1
            self._last_error = ""
        elif validated:
            self._last_error = "Person detected but key joints are not visible"
        else:
            self._last_error = "No person detected"
        return pose_frame

    # -------------------------------------------------------- keypoint utils
    def extract_keypoints(self, prediction: Any, frame_id: int = -1) -> List[PersonPose]:
        """Normalise a raw backend prediction into :class:`PersonPose` objects.

        Accepts an already-normalised pose, a sequence of poses, an
        ``(N, K, 2)``/``(N, K, 3)`` array, or a backend-specific object handled by
        the adapter.
        """
        if prediction is None:
            return []
        if isinstance(prediction, PersonPose):
            return [prediction]
        if isinstance(prediction, (list, tuple)) and prediction and isinstance(prediction[0], PersonPose):
            return list(prediction)
        if isinstance(prediction, np.ndarray) or (
            isinstance(prediction, (list, tuple)) and prediction and isinstance(prediction[0], (list, tuple, np.ndarray))
        ):
            return [self.keypoints_from_array(np.asarray(prediction, dtype=np.float32), frame_id=frame_id)]
        normalize = getattr(self.adapter, "normalize", None)
        if callable(normalize):
            return normalize(prediction, frame_id=frame_id)
        logger.warning("Unsupported prediction type: %s", type(prediction).__name__)
        return []

    def keypoints_from_array(self, array: np.ndarray, frame_id: int = -1, confidence: float = 1.0) -> PersonPose:
        """Build a :class:`PersonPose` from a ``(K, 2)`` or ``(K, 3)`` array."""
        array = np.asarray(array, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] < NUM_KEYPOINTS or array.shape[1] < 2:
            raise ValueError(f"Expected an array of shape (>={NUM_KEYPOINTS}, 2|3), got {array.shape}")
        keypoints: Dict[str, Keypoint] = {}
        for index, name in enumerate(KEYPOINT_NAMES):
            x_value, y_value = float(array[index, 0]), float(array[index, 1])
            score = float(array[index, 2]) if array.shape[1] >= 3 else confidence
            keypoints[name] = Keypoint(
                x=x_value,
                y=y_value,
                confidence=score,
                valid=bool(np.isfinite(x_value) and np.isfinite(y_value) and score > 0.0),
            )
        return PersonPose(keypoints=keypoints, detection_confidence=confidence, frame_id=frame_id)

    def validate_keypoints(
        self,
        pose: PersonPose,
        threshold: Optional[float] = None,
    ) -> PersonPose:
        """Apply the confidence gate and drop non-finite joints.

        Args:
            pose: The pose to validate (mutated in place and returned).
            threshold: Override of the configured keypoint confidence gate.

        Returns:
            The same :class:`PersonPose` instance, with ``valid`` flags updated.
        """
        gate = self.confidence_threshold if threshold is None else float(threshold)
        for keypoint in pose.keypoints.values():
            if not (np.isfinite(keypoint.x) and np.isfinite(keypoint.y)):
                keypoint.valid = False
                continue
            if keypoint.confidence < gate:
                keypoint.valid = False
        return pose

    # --------------------------------------------------------------- analytics
    def detection_rate(self) -> float:
        """Fraction of processed frames that yielded an analysable pose."""
        if self._frames_processed == 0:
            return 0.0
        return self._frames_with_pose / self._frames_processed

    @property
    def last_error(self) -> str:
        """Most recent failure reason (empty string when healthy)."""
        return self._last_error

    def reset(self) -> None:
        """Reset per-session statistics and tracking state."""
        self.tracker.reset()
        self._frames_processed = 0
        self._frames_with_pose = 0
        self._last_error = ""


def primary_points(pose: Optional[PersonPose], names: Iterable[str]) -> Dict[str, Point]:
    """Collect usable points for ``names`` (missing joints are simply omitted)."""
    result: Dict[str, Point] = {}
    if pose is None:
        return result
    for name in names:
        point = pose.point(name)
        if point is not None:
            result[name] = point
    return result


__all__ = ["PoseDetector", "PoseBackendError", "primary_points"]

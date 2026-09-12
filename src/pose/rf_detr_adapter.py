"""RF-DETR Keypoint adapter.

Verified against ``rfdetr==1.10.1`` (Apache-2.0, Roboflow):

* ``from rfdetr import RFDETRKeypointPreview``
* ``model.predict(image_rgb, threshold=...)`` returns an ``sv.KeyPoints`` object
  (``supervision>=0.29``) with:

  ==============================  ===========  =================================
  field                           shape        meaning
  ==============================  ===========  =================================
  ``key_points.xy``               (N, 17, 2)   pixel coordinates
  ``key_points.keypoint_confidence`` (N, 17)   per-joint findability score
  ``key_points.detection_confidence`` (N,)     per-person score
  ``key_points.visible``          (N, 17)      bool mask (``conf > 0``)
  ``key_points.data["xyxy"]``     (N, 4)       bounding boxes
  ``key_points.data["class_name"]`` (N,)       resolved class names
  ==============================  ===========  =================================

The preview checkpoint predicts the 17 COCO keypoints in the canonical order
defined in :mod:`src.pose.keypoints`, so index ``i`` maps directly to
``KEYPOINT_NAMES[i]``.

The default checkpoint (``rf-detr-keypoint-preview-xlarge.pth``) is downloaded
and cached automatically by the ``rfdetr`` package on first use.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from src.pose.base import BasePoseAdapter, PoseBackendError
from src.pose.keypoints import KEYPOINT_INDEX, KEYPOINT_NAMES, NUM_KEYPOINTS
from src.pose.person import DetectorStatus, Keypoint, PersonPose
from src.utils.logger import get_logger

logger = get_logger(__name__)

_BACKGROUND_CLASSES = {"__background__", "background", ""}


class RFDETRPoseAdapter(BasePoseAdapter):
    """Adapter for :class:`rfdetr.RFDETRKeypointPreview`."""

    name = "rf-detr"

    def __init__(
        self,
        *,
        pretrain_weights: str = "",
        class_name: str = "RFDETRKeypointPreview",
        keypoint_confidence_threshold: float = 0.5,
        detection_threshold: float = 0.5,
        device: str = "auto",
        model_name: str = "rf-detr-keypoint-preview",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            keypoint_confidence_threshold=keypoint_confidence_threshold,
            detection_threshold=detection_threshold,
            device=device,
            model_name=model_name,
            **kwargs,
        )
        self.pretrain_weights = str(pretrain_weights or "")
        self.class_name = class_name
        self._model: Any = None

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        """Import ``rfdetr`` and instantiate the keypoint model (idempotent).

        Raises:
            PoseBackendError: If ``rfdetr`` is missing, the class is unavailable
                or the weights cannot be loaded.
        """
        if self._loaded and self._model is not None:
            return

        try:
            import rfdetr  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on the host env
            message = (
                "The 'rfdetr' package is not installed. Install the pose stack with "
                "`pip install \"rfdetr>=1.8.1,<2\"` (pulls torch + torchvision), or set "
                "model.engine: demo in config/config.yaml to run without it."
            )
            logger.error("%s (%s)", message, exc)
            self.status = DetectorStatus(backend=self.name, loaded=False, device="cpu", model_name=self.model_name, error=message)
            raise PoseBackendError(message) from exc

        model_cls = getattr(rfdetr, self.class_name, None)
        if model_cls is None:
            available = [n for n in dir(rfdetr) if n.startswith("RFDETR")]
            message = (
                f"rfdetr {getattr(rfdetr, '__version__', '?')} does not expose '{self.class_name}'. "
                f"Available classes: {available}. The keypoint preview model requires rfdetr>=1.8.1."
            )
            logger.error(message)
            self.status = DetectorStatus(backend=self.name, loaded=False, error=message)
            raise PoseBackendError(message)

        device = self._resolve_device(self.device)
        kwargs: Dict[str, Any] = {}
        if self.pretrain_weights:
            kwargs["pretrain_weights"] = self.pretrain_weights
        kwargs["device"] = device

        try:
            try:
                self._model = model_cls(**kwargs)
            except TypeError:
                # Older/newer builds that do not accept `device=`.
                kwargs.pop("device", None)
                self._model = model_cls(**kwargs)
                device = "auto"
        except Exception as exc:  # noqa: BLE001 - wrap any weight/download failure
            message = f"Failed to load {self.class_name}: {type(exc).__name__}: {exc}"
            logger.error(message)
            self.status = DetectorStatus(backend=self.name, loaded=False, device=device, model_name=self.model_name, error=message)
            raise PoseBackendError(message) from exc

        self._loaded = True
        self.status = self._make_status(device=device)
        self.status.keypoints_supported = NUM_KEYPOINTS
        logger.info("Loaded %s on %s", self.class_name, device)

    # ----------------------------------------------------------------- infer
    def infer(self, frame_bgr: np.ndarray) -> List[PersonPose]:
        """Run RF-DETR on one BGR frame and return normalised :class:`PersonPose` objects."""
        if self._model is None:
            raise PoseBackendError("RF-DETR model is not loaded. Call load() first.")

        import cv2  # local import keeps module import light

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        prediction = self._model.predict(frame_rgb, threshold=self.detection_threshold)
        return self.normalize(prediction, frame_id=self._frame_id)

    # ------------------------------------------------------------- normalize
    def normalize(self, prediction: Any, frame_id: int = -1) -> List[PersonPose]:
        """Convert an ``sv.KeyPoints`` (or compatible) object into poses.

        Args:
            prediction: The object returned by ``model.predict``.
            frame_id: Frame index to stamp on the resulting poses.

        Returns:
            One :class:`PersonPose` per detected person (background classes dropped).
        """
        if prediction is None:
            return []
        if isinstance(prediction, (list, tuple)):
            prediction = prediction[0] if prediction else None
        if prediction is None:
            return []

        xy = getattr(prediction, "xy", None)
        if xy is None:
            return []
        xy = np.asarray(xy, dtype=np.float32)
        if xy.ndim != 3:
            logger.warning("Unexpected keypoint tensor shape %s; expected (N, K, 2)", xy.shape)
            return []

        confidence = getattr(prediction, "keypoint_confidence", None)
        confidence = (
            np.ones(xy.shape[:2], dtype=np.float32) if confidence is None else np.asarray(confidence, dtype=np.float32)
        )
        detection_confidence = getattr(prediction, "detection_confidence", None)
        detection_confidence = (
            np.ones(xy.shape[0], dtype=np.float32)
            if detection_confidence is None
            else np.asarray(detection_confidence, dtype=np.float32)
        )
        visible = getattr(prediction, "visible", None)
        visible = None if visible is None else np.asarray(visible, dtype=bool)

        data: Dict[str, Any] = dict(getattr(prediction, "data", {}) or {})
        boxes = data.get("xyxy")
        boxes = None if boxes is None else np.asarray(boxes, dtype=np.float32)
        class_names = data.get("class_name")
        class_names = list(class_names) if class_names is not None else None

        people: List[PersonPose] = []
        for index in range(xy.shape[0]):
            if class_names is not None and index < len(class_names):
                if str(class_names[index]).strip().lower() in _BACKGROUND_CLASSES:
                    continue

            person_confidence = float(detection_confidence[index]) if index < len(detection_confidence) else 1.0
            keypoints: Dict[str, Keypoint] = {}
            for kp_index in range(min(xy.shape[1], NUM_KEYPOINTS)):
                name = KEYPOINT_NAMES[kp_index]
                x_value = float(xy[index, kp_index, 0])
                y_value = float(xy[index, kp_index, 1])
                score = float(confidence[index, kp_index]) if kp_index < confidence.shape[1] else 1.0
                is_visible = True if visible is None else bool(visible[index, kp_index])
                keypoints[name] = Keypoint(
                    x=x_value,
                    y=y_value,
                    confidence=score,
                    valid=bool(is_visible and np.isfinite(x_value) and np.isfinite(y_value)),
                )

            bbox = tuple(float(v) for v in boxes[index]) if boxes is not None and index < len(boxes) else None
            people.append(
                PersonPose(
                    keypoints=keypoints,
                    detection_confidence=person_confidence,
                    bbox=bbox,  # type: ignore[arg-type]
                    frame_id=frame_id,
                )
            )
        return people

    def release(self) -> None:
        """Drop the model reference so GPU memory can be reclaimed."""
        self._model = None
        self._loaded = False
        self.status.loaded = False


def keypoint_names_from_prediction(prediction: Any) -> Optional[List[str]]:
    """Return resolved keypoint names when a backend exposes them (else COCO)."""
    data = getattr(prediction, "data", None)
    if isinstance(data, dict) and "keypoint_names" in data:
        return [str(name) for name in data["keypoint_names"]]
    return list(KEYPOINT_NAMES)


def keypoint_index(name: str) -> int:
    """Index of a canonical keypoint name (re-exported convenience)."""
    return KEYPOINT_INDEX[name]

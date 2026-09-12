"""Pose backend contract (adapter pattern).

The rest of the application only ever talks to :class:`BasePoseAdapter`, so
swapping RF-DETR for another estimator never touches analysis or UI code.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np

from src.pose.person import DetectorStatus, PersonPose, PoseFrame
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PoseBackendError(RuntimeError):
    """Raised when a pose backend cannot be constructed or run."""


class BasePoseAdapter(ABC):
    """Adapter that converts a concrete model's output into :class:`PersonPose`.

    Subclasses only implement :meth:`load` and :meth:`infer`; keypoint
    normalisation, confidence gating and person bookkeeping are shared.
    """

    name: str = "base"

    def __init__(
        self,
        *,
        keypoint_confidence_threshold: float = 0.5,
        detection_threshold: float = 0.5,
        device: str = "auto",
        model_name: str = "",
        **kwargs: Any,
    ) -> None:
        self.keypoint_confidence_threshold = float(keypoint_confidence_threshold)
        self.detection_threshold = float(detection_threshold)
        self.device = device
        self.model_name = model_name
        self.options = kwargs
        self.status = DetectorStatus(backend=self.name, model_name=model_name)
        self._loaded = False
        self._frame_id = 0

    # ------------------------------------------------------------------ API
    @abstractmethod
    def load(self) -> None:
        """Load model weights. Must be idempotent."""

    @abstractmethod
    def infer(self, frame_bgr: np.ndarray) -> List[PersonPose]:
        """Run inference on one BGR frame and return normalised poses."""

    def is_loaded(self) -> bool:
        """``True`` once :meth:`load` has succeeded."""
        return self._loaded

    def ensure_loaded(self) -> None:
        """Load the model on first use (lazy initialisation)."""
        if not self._loaded:
            self.load()

    def predict(self, frame_bgr: np.ndarray) -> PoseFrame:
        """Detect all people in ``frame_bgr`` and wrap them in a :class:`PoseFrame`.

        Errors are captured into :attr:`status` instead of propagating so a single
        corrupted frame cannot kill a live session.
        """
        if frame_bgr is None or not isinstance(frame_bgr, np.ndarray) or frame_bgr.size == 0:
            return PoseFrame(frame_id=self._frame_id)

        started = time.perf_counter()
        try:
            self.ensure_loaded()
            people = self.infer(frame_bgr)
        except PoseBackendError:
            raise
        except Exception as exc:  # noqa: BLE001 - defensive: keep the session alive
            logger.exception("Pose inference failed on frame %s", self._frame_id)
            self.status.error = f"{type(exc).__name__}: {exc}"
            people = []

        inference_ms = (time.perf_counter() - started) * 1000.0
        pose_frame = PoseFrame(
            people=people,
            primary=None,
            frame_id=self._frame_id,
            timestamp=time.time(),
            inference_ms=inference_ms,
            multiple_people=len(people) > 1,
        )
        self._frame_id += 1
        return pose_frame

    def release(self) -> None:
        """Release backend resources (optional)."""
        self._loaded = False

    def describe(self) -> str:
        """Human readable backend description for the UI."""
        return self.status.describe()

    # ------------------------------------------------------------- utilities
    def _resolve_device(self, requested: str = "auto") -> str:
        """Resolve ``"auto"`` to ``"cuda"``/``"cpu"`` when torch is available."""
        if requested != "auto":
            return requested
        try:  # pragma: no cover - depends on the host machine
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"

    def _make_status(self, device: str, loaded: bool = True) -> DetectorStatus:
        return DetectorStatus(
            backend=self.name,
            loaded=loaded,
            device=device,
            model_name=self.model_name or self.name,
        )


def adapter_from_config(settings: Any) -> BasePoseAdapter:
    """Build the adapter selected in ``config.yaml`` / the sidebar.

    Args:
        settings: A :class:`src.config.settings.Settings` instance.

    Returns:
        A ready-to-load :class:`BasePoseAdapter` (not yet loaded).

    Raises:
        ValueError: If ``settings.model.engine`` is unknown.
    """
    engine = (settings.model.engine or "rf-detr").lower()
    common = dict(
        keypoint_confidence_threshold=settings.model.confidence_threshold,
        detection_threshold=settings.model.detection_threshold,
        device=settings.model.device,
        model_name=settings.model.name,
    )
    if engine in {"rf-detr", "rfdetr", "rf_detr"}:
        from src.pose.rf_detr_adapter import RFDETRPoseAdapter

        return RFDETRPoseAdapter(pretrain_weights=settings.model.pretrain_weights, **common)
    if engine in {"demo", "synthetic"}:
        from src.pose.demo_adapter import SyntheticPushUpAdapter

        return SyntheticPushUpAdapter(**common)
    raise ValueError(f"Unknown pose engine: {settings.model.engine!r} (expected 'rf-detr' or 'demo')")


__all__ = ["BasePoseAdapter", "PoseBackendError", "adapter_from_config"]

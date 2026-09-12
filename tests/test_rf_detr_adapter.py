"""Tests for the RF-DETR keypoint adapter's output normalisation.

These use a mock that mirrors the exact ``sv.KeyPoints`` fields verified against
``supervision==0.30.2`` (``xy`` (N,K,2), ``keypoint_confidence`` (N,K),
``detection_confidence`` (N,), ``visible`` (N,K), ``data["xyxy"|"class_name"]``),
so the real-model code path is covered without installing torch/weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import load_settings  # noqa: E402
from src.pose.keypoints import NUM_KEYPOINTS  # noqa: E402
from src.pose.pose_detector import PoseDetector  # noqa: E402
from src.pose.rf_detr_adapter import RFDETRPoseAdapter  # noqa: E402


class MockKeyPoints:
    """Stand-in for ``sv.KeyPoints`` with the verified field layout."""

    def __init__(self, xy, kc, dc, vis, data):
        self.xy = xy
        self.keypoint_confidence = kc
        self.detection_confidence = dc
        self.visible = vis
        self.data = data


def _mock(n: int = 1, *, low_conf_index: int | None = None, class_names=None) -> MockKeyPoints:
    xy = np.zeros((n, NUM_KEYPOINTS, 2), dtype=np.float32)
    for person in range(n):
        xy[person, :, 0] = np.arange(NUM_KEYPOINTS) * 10.0 + person * 100.0
        xy[person, :, 1] = np.arange(NUM_KEYPOINTS) * 5.0 + person * 50.0
    kc = np.full((n, NUM_KEYPOINTS), 0.9, dtype=np.float32)
    if low_conf_index is not None:
        kc[0, low_conf_index] = 0.2
    dc = np.linspace(0.97, 0.55, n).astype(np.float32)
    vis = kc > 0
    data = {"xyxy": np.tile(np.array([0, 0, 160, 80], dtype=np.float32), (n, 1))}
    if class_names is not None:
        data["class_name"] = np.array(class_names)
    return MockKeyPoints(xy, kc, dc, vis, data)


def test_normalize_maps_coco_indices_to_names():
    adapter = RFDETRPoseAdapter()
    people = adapter.normalize(_mock(1), frame_id=7)
    assert len(people) == 1
    person = people[0]
    # COCO index 5 == left_shoulder -> x = 5*10 = 50, y = 5*5 = 25
    assert person.point("left_shoulder") == pytest.approx((50.0, 25.0))
    assert person.frame_id == 7
    assert person.valid_count == NUM_KEYPOINTS
    assert person.has_minimal_pose()


def test_normalize_filters_background_class():
    adapter = RFDETRPoseAdapter()
    people = adapter.normalize(_mock(2, class_names=["person", "__background__"]))
    assert len(people) == 1  # background dropped


def test_normalize_handles_none_and_empty():
    adapter = RFDETRPoseAdapter()
    assert adapter.normalize(None) == []
    empty = MockKeyPoints(
        np.zeros((0, NUM_KEYPOINTS, 2), np.float32),
        np.zeros((0, NUM_KEYPOINTS), np.float32),
        np.zeros((0,), np.float32),
        None,
        {},
    )
    assert adapter.normalize(empty) == []


def test_normalize_handles_list_wrapper():
    """predict() may return a single object; a list is unwrapped defensively."""
    adapter = RFDETRPoseAdapter()
    people = adapter.normalize([_mock(1)])
    assert len(people) == 1


def test_confidence_gate_drops_low_score_joints():
    settings = load_settings()
    settings.model.confidence_threshold = 0.5
    adapter = RFDETRPoseAdapter()
    detector = PoseDetector(settings, adapter=adapter)

    person = adapter.normalize(_mock(1, low_conf_index=3))[0]
    assert person.get("left_ear").valid  # visible before the gate
    detector.validate_keypoints(person, threshold=0.5)
    assert not person.get("left_ear").valid  # 0.2 < 0.5 -> dropped
    assert person.valid_count == NUM_KEYPOINTS - 1


def test_load_without_rfdetr_raises_backend_error():
    """When the rfdetr package is absent, load() raises a clear PoseBackendError."""
    from src.pose.base import PoseBackendError

    adapter = RFDETRPoseAdapter()
    # Force the import to fail deterministically regardless of the host env.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "rfdetr":
            raise ImportError("no rfdetr here")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = fake_import
    try:
        with pytest.raises(PoseBackendError):
            adapter.load()
    finally:
        builtins.__import__ = real_import
    assert "rfdetr" in adapter.status.error

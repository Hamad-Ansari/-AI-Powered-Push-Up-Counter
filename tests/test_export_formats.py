"""Tests for multi-format video export and fault-injected demo sources."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import load_settings  # noqa: E402
from src.analysis.angle_calculator import compute_angles  # noqa: E402
from src.pose.demo_adapter import SyntheticPushUpAdapter  # noqa: E402
from src.video.sources import DemoSource  # noqa: E402
from src.video.video_exporter import VideoExporter  # noqa: E402


@pytest.fixture()
def settings(tmp_path):
    s = load_settings()
    s.project_root = tmp_path
    s.output.videos_dir = str(tmp_path / "videos")
    s.ensure_directories()
    return s


def _frames(n: int = 12, size=(320, 180)):
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8) for _ in range(n)]


# --------------------------------------------------------------------------- #
# Formats
# --------------------------------------------------------------------------- #
def test_gif_export_writes_valid_gif(settings):
    path = settings.videos_dir / "out.gif"
    exporter = VideoExporter(path, (320, 180), fps=10.0)
    assert exporter.open()
    for frame in _frames():
        exporter.write(frame)
    result = exporter.close()
    assert result is not None and result.exists()
    assert result.read_bytes()[:6] in {b"GIF87a", b"GIF89a"}


def test_mp4_and_avi_export(settings):
    for suffix in ("mp4", "avi"):
        path = settings.videos_dir / f"out.{suffix}"
        exporter = VideoExporter(path, (320, 180), fps=10.0, fourcc="mp4v", fallback_fourccs=["XVID", "MJPG"])
        assert exporter.open(), f"could not open {suffix}"
        for frame in _frames():
            exporter.write(frame)
        result = exporter.close()
        assert result is not None and result.stat().st_size > 0


def test_empty_export_is_removed(settings):
    path = settings.videos_dir / "empty.gif"
    exporter = VideoExporter(path, (320, 180), fps=10.0)
    exporter.open()
    assert exporter.close() is None
    assert not path.exists()


def test_unknown_container_falls_back_to_mp4(settings):
    path = settings.videos_dir / "out.weird"
    exporter = VideoExporter(path, (320, 180), fps=10.0)
    exporter.open()
    assert exporter.path.suffix == ".mp4"
    exporter.close()


# --------------------------------------------------------------------------- #
# Fault-injected demo source
# --------------------------------------------------------------------------- #
def _scan_angles(source, frames=40):
    """Advance source + adapter in lock-step for `frames` steps, collecting angles."""
    source.open()
    adapter = SyntheticPushUpAdapter(
        cycle_seconds=source.cycle_seconds,
        fps=source.fps,
        hips_offset=source.hips_offset,
        arm_asymmetry=source.arm_asymmetry,
    )
    adapter.load()
    results = []
    for _ in range(frames):
        ok, frame = source.read()
        if not ok:
            break
        pose_frame = adapter.predict(frame)  # predict() advances the frame counter
        if pose_frame.people:
            results.append(compute_angles(pose_frame.people[0]))
    source.release()
    return results


def _max_deviation(angles):
    import math

    return max(a.alignment_deviation for a in angles if math.isfinite(a.alignment_deviation))


def _max_elbow_diff(angles):
    import math

    return max(a.elbow_diff for a in angles if math.isfinite(a.elbow_diff))


def test_demo_source_hips_high_deviates_alignment():
    source = DemoSource(cycle_seconds=3.0, fps=25.0, size=(640, 360), hips_offset=0.12)
    assert _max_deviation(_scan_angles(source)) > 12.0  # clearly off-straight


def test_demo_source_good_keeps_alignment():
    source = DemoSource(cycle_seconds=3.0, fps=25.0, size=(640, 360))
    assert _max_deviation(_scan_angles(source)) < 12.0


def test_demo_source_uneven_arms():
    source = DemoSource(cycle_seconds=3.0, fps=25.0, size=(640, 360), arm_asymmetry=30.0)
    assert _max_elbow_diff(_scan_angles(source)) > 15.0


def test_demo_source_tempo_configurable():
    fast = DemoSource(cycle_seconds=1.5, fps=25.0)
    slow = DemoSource(cycle_seconds=4.0, fps=25.0)
    # Half-cycle phase should differ between tempos at the same frame index.
    assert fast.phase_for_frame(20) != slow.phase_for_frame(20)

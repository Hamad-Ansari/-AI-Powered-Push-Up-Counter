"""Integration tests: the full frame pipeline and the session controller.

These exercise the same code paths the Streamlit UI drives, using the
dependency-free demo adapter so they run anywhere (no torch, no weights).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import load_settings  # noqa: E402
from src.pose.demo_adapter import SyntheticPushUpAdapter  # noqa: E402
from src.pose.pose_detector import PoseDetector  # noqa: E402
from src.video.session_controller import SessionController, build_source  # noqa: E402
from src.video.sources import DemoSource, VideoSourceError, VideoFileSource  # noqa: E402
from src.video.video_processor import FrameAnalyzer, FrameRenderer  # noqa: E402


@pytest.fixture()
def demo_settings(tmp_path):
    """Settings pointed at a temp output dir, using the demo engine."""
    settings = load_settings()
    settings.model.engine = "demo"
    settings.project_root = tmp_path
    settings.output.screenshots_dir = str(tmp_path / "shots")
    settings.output.videos_dir = str(tmp_path / "videos")
    settings.output.reports_dir = str(tmp_path / "reports")
    settings.ensure_directories()
    return settings


def _analyzer(settings):
    adapter = SyntheticPushUpAdapter(cycle_seconds=3.0, fps=25.0)
    return FrameAnalyzer(settings, pose_detector=PoseDetector(settings, adapter=adapter))


def test_pipeline_counts_reps_over_demo_stream(demo_settings):
    """Five demo cycles produce five valid reps with a rendered frame."""
    analyzer = _analyzer(demo_settings)
    analyzer.load_model()
    renderer = FrameRenderer(demo_settings, analyzer)

    source = DemoSource(cycle_seconds=3.0, fps=25.0, size=(640, 360))
    source.open()
    last = None
    for index in range(375):  # 15 s
        ok, frame = source.read()
        assert ok
        result = analyzer.process_frame(frame, timestamp=index / 25.0)
        annotated = renderer.render(frame, result)
        assert annotated.shape == frame.shape
        last = result
    source.release()

    summary = analyzer.session.summary()
    assert summary.total_reps == 5
    assert summary.valid_reps == 5
    assert summary.invalid_reps == 0
    assert last.visible_keypoints == 17
    assert summary.average_form_score > 0


def test_pipeline_handles_lost_tracking(demo_settings):
    """A dropped pose frame is tolerated and does not crash the pipeline."""
    adapter = SyntheticPushUpAdapter(cycle_seconds=3.0, fps=25.0, noise_frames=5)
    analyzer = FrameAnalyzer(demo_settings, pose_detector=PoseDetector(demo_settings, adapter=adapter))
    analyzer.load_model()

    source = DemoSource(cycle_seconds=3.0, fps=25.0, size=(640, 360))
    source.open()
    saw_missing = False
    for index in range(60):
        ok, frame = source.read()
        result = analyzer.process_frame(frame, timestamp=index / 25.0)
        if not result.has_pose:
            saw_missing = True
    source.release()
    assert saw_missing  # frame 5 was dropped on purpose


def test_invalid_reps_are_rejected(demo_settings):
    """A shallow, fast jiggle is counted as invalid rather than as a rep."""
    demo_settings.pushup.min_rep_duration = 0.05
    demo_settings.validation.min_depth_angle = 80.0  # require a very deep rep
    analyzer = _analyzer(demo_settings)
    analyzer.load_model()

    source = DemoSource(cycle_seconds=1.0, fps=25.0, size=(640, 360))
    source.open()
    for index in range(150):
        ok, frame = source.read()
        analyzer.process_frame(frame, timestamp=index / 25.0)
    source.release()

    summary = analyzer.session.summary()
    # The 1 s cycle reaches ~88 deg which fails the 80 deg depth gate.
    assert summary.valid_reps == 0
    assert summary.invalid_reps >= 1


def test_reports_and_dataframe(demo_settings):
    """Reports are written in all three formats and the dataframe is populated."""
    from src.analytics.reporter import generate_reports

    analyzer = _analyzer(demo_settings)
    analyzer.load_model()
    source = DemoSource(cycle_seconds=3.0, fps=25.0, size=(480, 320))
    source.open()
    for index in range(150):
        ok, frame = source.read()
        analyzer.process_frame(frame, timestamp=index / 25.0)
    source.release()

    df = analyzer.session.to_dataframe()
    assert not df.empty
    assert {"left_elbow", "form_score", "rep_count", "state"}.issubset(df.columns)

    paths = generate_reports(analyzer.session, demo_settings.reports_dir, source_name="test")
    written = paths.existing()
    assert len(written) == 3
    assert all(path.exists() and path.stat().st_size > 0 for path in written)


def test_session_controller_lifecycle(demo_settings):
    """The threaded controller starts, produces frames, exports and stops."""
    factory = build_source(demo_settings, "demo")
    controller = SessionController(demo_settings, factory, auto_export=True)
    assert controller.start()
    assert controller.running

    deadline = time.time() + 6.0
    frame = None
    while time.time() < deadline:
        frame, result = controller.latest()
        if frame is not None and result is not None and result.rep_count >= 1:
            break
        time.sleep(0.1)

    frame, result = controller.latest()
    assert frame is not None and isinstance(frame, np.ndarray)
    assert result is not None

    controller.pause()
    assert controller.paused
    controller.step_frame()
    controller.resume()
    assert not controller.paused

    export_path = controller.stop_export()
    assert export_path is not None and export_path.exists()

    paths = controller.generate_reports(("json",))
    assert paths.json_path is not None and paths.json_path.exists()

    controller.stop()
    assert not controller.running


def test_video_file_source_rejects_missing_file(demo_settings):
    """A non-existent path raises a friendly error rather than a raw OpenCV failure."""
    source = VideoFileSource(demo_settings.project_root / "does_not_exist.mp4")
    with pytest.raises(VideoSourceError):
        source.open()


def test_build_source_modes(demo_settings):
    """The source factory returns the right type for each mode."""
    assert isinstance(build_source(demo_settings, "demo")(), DemoSource)
    from src.video.sources import WebcamSource

    assert isinstance(build_source(demo_settings, "webcam")(), WebcamSource)
    with pytest.raises(ValueError):
        build_source(demo_settings, "video")  # no path
    with pytest.raises(ValueError):
        build_source(demo_settings, "nonsense")

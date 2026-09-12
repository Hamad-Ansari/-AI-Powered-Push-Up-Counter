"""Threaded live-session controller for the Streamlit dashboard.

Runs the capture -> analyse -> render -> export loop on a worker thread and
exposes a small, thread-safe control surface (start/stop/pause/step/screenshot/
export) plus a lock-guarded snapshot of the latest annotated frame. This keeps
``app.py`` free of threading concerns.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

from src.analytics.reporter import ReportPaths, generate_reports
from config.settings import Settings
from src.utils.logger import get_logger
from src.video.sources import DemoSource, FrameSource, VideoFileSource, VideoSourceError, WebcamSource
from src.utils.helpers import timestamp_slug
from src.video.video_exporter import VideoExporter, save_screenshot
from src.video.video_processor import FrameAnalyzer, FrameRenderer, FrameResult
from src.visualization.drawing import resize_to_width

logger = get_logger(__name__)


class SessionController:
    """Owns one live processing session and its worker thread.

    Example:
        >>> controller = SessionController(settings, source_factory)
        >>> controller.start()
        >>> frame, result = controller.latest()
        >>> controller.stop()
    """

    def __init__(
        self,
        settings: Settings,
        source_factory: Callable[[], FrameSource],
        *,
        analyzer: Optional[FrameAnalyzer] = None,
        auto_export: bool = False,
        export_format: Optional[str] = None,
    ) -> None:
        self.settings = settings
        self.source_factory = source_factory
        self.analyzer = analyzer or FrameAnalyzer(settings)
        self.renderer = FrameRenderer(settings, self.analyzer)
        self.auto_export = auto_export
        self.export_format = (export_format or settings.video.export_format or "mp4").lstrip(".").lower()

        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._paused = threading.Event()
        self._step = threading.Event()
        self._lock = threading.RLock()

        self._source: Optional[FrameSource] = None
        self._exporter: Optional[VideoExporter] = None
        self._export_path: Optional[Path] = None

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_result: Optional[FrameResult] = None
        self._error: str = ""
        self._frame_index = 0
        self._screenshot_request = threading.Event()

    # ------------------------------------------------------------------ state
    @property
    def running(self) -> bool:
        """``True`` while the worker thread is alive."""
        return self._running.is_set()

    @property
    def paused(self) -> bool:
        """``True`` when the session is paused."""
        return self._paused.is_set()

    @property
    def error(self) -> str:
        """Most recent session error (empty when healthy)."""
        with self._lock:
            return self._error

    @property
    def source_name(self) -> str:
        """Name of the active frame source."""
        return getattr(self._source, "name", "unknown")

    # ---------------------------------------------------------------- control
    def start(self) -> bool:
        """Open the source and start the worker thread.

        Returns:
            ``True`` when the session started successfully.
        """
        if self.running:
            return True
        try:
            self.analyzer.load_model()
            self._source = self.source_factory()
            self._source.open()
        except VideoSourceError as exc:
            with self._lock:
                self._error = str(exc)
            logger.error("Session start failed: %s", exc)
            return False
        except Exception as exc:  # noqa: BLE001 - surface model load failures
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
            logger.exception("Session start failed")
            return False

        self._frame_index = 0
        if self.auto_export:
            self.start_export()

        self._paused.clear()
        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="pushup-session", daemon=True)
        self._thread.start()
        logger.info("Session started on %s", self.source_name)
        return True

    def stop(self) -> None:
        """Stop the worker thread and release the source/exporter."""
        self._running.clear()
        self._step.set()  # unblock a paused loop
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.stop_export()
        if self._source is not None:
            self._source.release()
            self._source = None
        logger.info("Session stopped")

    def pause(self) -> None:
        """Freeze processing on the current frame."""
        self._paused.set()

    def resume(self) -> None:
        """Resume processing."""
        self._paused.clear()

    def toggle_pause(self) -> bool:
        """Flip the pause state; returns the new ``paused`` value."""
        if self._paused.is_set():
            self.resume()
        else:
            self.pause()
        return self.paused

    def step_frame(self) -> None:
        """Process exactly one additional frame while paused."""
        self._paused.set()
        self._step.set()

    def request_screenshot(self) -> None:
        """Ask the worker to save the next annotated frame as a screenshot."""
        self._screenshot_request.set()

    # ----------------------------------------------------------------- export
    def _build_export_path(self) -> Path:
        """Default export filename using the selected container format."""
        name = self.settings.output.video_name_template.format(timestamp=timestamp_slug())
        stem = Path(name).stem
        return self.settings.videos_dir / f"{stem}.{self.export_format}"

    def start_export(self) -> bool:
        """Begin writing the processed video to ``outputs/videos``."""
        with self._lock:
            if self._exporter is not None:
                return True
            source = self._source
            width = getattr(source, "width", 0) or 1280
            height = getattr(source, "height", 0) or 720
            fps = getattr(source, "fps", 0.0) or self.settings.video.output_fps
            path = self._build_export_path()
            exporter = VideoExporter(path, (width, height), fps, self.settings.video.fourcc, self.settings.video.fallback_fourccs)
            if not exporter.open():
                self._error = "No usable video codec for export"
                return False
            self._exporter = exporter
            self._export_path = path
            logger.info("Export started: %s", path.name)
            return True

    def stop_export(self) -> Optional[Path]:
        """Finalise and return the exported video path (if any)."""
        with self._lock:
            if self._exporter is None:
                return None
            path = self._exporter.close()
            self._exporter = None
            logger.info("Export stopped: %s", path.name if path else "none")
            return path

    @property
    def exporting(self) -> bool:
        """``True`` while a processed video is being written."""
        with self._lock:
            return self._exporter is not None

    @property
    def export_path(self) -> Optional[Path]:
        """Path of the current/most recent export."""
        with self._lock:
            return self._export_path

    # ------------------------------------------------------------------- loop
    def _loop(self) -> None:
        assert self._source is not None
        fps = getattr(self._source, "fps", 0.0) or self.settings.video.output_fps
        frame_period = 1.0 / fps if fps > 0 else 0.0
        # A webcam must run in real time. Demo/video previews are paced to the
        # source frame rate too (when enabled) so playback feels natural instead
        # of "as fast as the CPU can decode"; analysis timestamps always follow
        # the source timeline for consistent rep timing.
        is_live = isinstance(self._source, WebcamSource)
        pace = is_live or self.settings.video.realtime_pacing

        while self._running.is_set():
            loop_start = time.perf_counter()
            if self._paused.is_set() and not self._step.is_set():
                time.sleep(0.02)
                continue
            self._step.clear()

            try:
                success, frame = self._source.read()
            except Exception as exc:  # noqa: BLE001 - keep the session alive
                with self._lock:
                    self._error = f"Read error: {exc}"
                break
            if not success or frame is None:
                if not is_live:
                    logger.info("End of video reached at frame %d", self._frame_index)
                self._running.clear()
                break

            stamp = self._frame_index / fps if fps > 0 else None
            try:
                result = self.analyzer.process_frame(frame, timestamp=stamp)
                annotated = self.renderer.render(frame, result)
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._error = f"{type(exc).__name__}: {exc}"
                logger.exception("Frame processing failed")
                break

            with self._lock:
                if self._exporter is not None:
                    self._exporter.write(annotated)
                # The UI receives a downscaled copy so the Streamlit preview stays
                # snappy; the exporter keeps the full-resolution frame.
                self._latest_frame = resize_to_width(annotated, self.settings.video.display_width)
                self._latest_result = result

            if self._screenshot_request.is_set():
                self._screenshot_request.clear()
                try:
                    save_screenshot(annotated, self.settings.screenshots_dir, image_format=self.settings.output.screenshot_format)
                except (OSError, ValueError) as exc:
                    logger.warning("Screenshot failed: %s", exc)

            self._frame_index += 1
            if pace and frame_period > 0:
                elapsed = time.perf_counter() - loop_start
                if elapsed < frame_period:
                    time.sleep(frame_period - elapsed)

    # ---------------------------------------------------------------- snapshot
    def latest(self) -> Tuple[Optional[np.ndarray], Optional[FrameResult]]:
        """Return the most recent ``(annotated_frame, result)`` pair."""
        with self._lock:
            return self._latest_frame, self._latest_result

    def summary(self):
        """Current session summary statistics."""
        return self.analyzer.session.summary()

    def generate_reports(self, formats: Tuple[str, ...] = ("json", "csv", "md")) -> ReportPaths:
        """Write the session reports to ``outputs/reports``."""
        return generate_reports(
            self.analyzer.session,
            self.settings.reports_dir,
            formats=formats,
            source_name=self.source_name,
            pose_backend=self.analyzer.pose_detector.adapter.describe(),
        )


def build_source(
    settings: Settings,
    mode: str,
    *,
    video_path: Optional[str | Path] = None,
    camera_index: int = 0,
    demo_cycle: float = 3.0,
    hips_offset: float = 0.0,
    arm_asymmetry: float = 0.0,
) -> Callable[[], FrameSource]:
    """Return a factory for the frame source selected in the sidebar.

    Args:
        settings: Application settings.
        mode: ``"webcam"``, ``"video"`` or ``"demo"``.
        video_path: Path to the uploaded/selected video (``mode="video"``).
        camera_index: Webcam index (``mode="webcam"``).
        demo_cycle: Seconds per push-up repetition in demo mode.
        hips_offset: Demo hip deflection for bad-form previews.
        arm_asymmetry: Demo left/right elbow mismatch (degrees).

    Returns:
        A zero-argument callable producing a fresh :class:`FrameSource`.

    Raises:
        ValueError: If ``mode`` is unknown or ``mode="video"`` has no path.
    """
    mode = mode.lower()
    if mode == "webcam":
        return lambda: WebcamSource(camera_index, fallback_fps=settings.video.output_fps)
    if mode == "demo":
        return lambda: DemoSource(
            cycle_seconds=demo_cycle,
            fps=settings.video.output_fps,
            size=(1280, 720),
            hips_offset=hips_offset,
            arm_asymmetry=arm_asymmetry,
        )
    if mode == "video":
        if video_path is None:
            raise ValueError("A video path is required for mode='video'")
        return lambda: VideoFileSource(video_path, loop=settings.video.loop_webcam_demo, fallback_fps=settings.video.output_fps)
    raise ValueError(f"Unknown source mode: {mode!r}")


__all__ = ["SessionController", "build_source"]

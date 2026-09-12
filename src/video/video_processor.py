"""The end-to-end frame pipeline.

Order of operations (exactly as documented in the README)::

    Frame -> validation -> pose detection -> keypoint extraction -> confidence
    gate -> angle calculation -> smoothing -> movement analysis -> posture
    analysis -> push-up state machine -> rep validation -> form scoring ->
    skeleton drawing -> HUD overlay -> analytics -> display / save
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from src.analytics.session_tracker import FrameRecord, RepRecord, SessionTracker
from src.analysis.angle_calculator import AngleSet, compute_angles
from src.analysis.form_analyzer import FormAnalyzer
from src.analysis.form_scorer import FormGrade, FormResult
from src.analysis.movement_analyzer import ExerciseState, MovementState
from src.analysis.posture_analyzer import PostureResult
from config.settings import Settings
from src.counter.pushup_counter import CounterUpdate, PushUpCounter, RepValidationResult
from src.pose.keypoints import NUM_KEYPOINTS
from src.pose.person import PersonPose, PoseFrame
from src.pose.pose_detector import PoseDetector
from src.utils.helpers import FPSMeter, format_duration, timestamp_slug
from src.utils.logger import get_logger
from src.video.sources import FrameSource, VideoSourceError, open_source
from src.video.video_exporter import VideoExporter, save_screenshot
from src.visualization.colors import Palette
from src.visualization.overlay import HudData, OverlayRenderer
from src.visualization.skeleton_drawer import SkeletonDrawer

logger = get_logger(__name__)


@dataclass(slots=True)
class FrameResult:
    """Complete analysis output for one frame."""

    frame_id: int = 0
    timestamp: float = 0.0
    frame: Optional[np.ndarray] = None
    pose_frame: Optional[PoseFrame] = None
    pose: Optional[PersonPose] = None
    angles: AngleSet = field(default_factory=AngleSet)
    movement: Optional[MovementState] = None
    posture: Optional[PostureResult] = None
    form: Optional[FormResult] = None
    counter: Optional[CounterUpdate] = None
    rep_completed: bool = False
    last_validation: Optional[RepValidationResult] = None
    fps: float = 0.0
    inference_ms: float = 0.0
    visible_keypoints: int = 0
    multiple_people: bool = False
    has_pose: bool = False
    detection_confidence: float = 0.0
    message: str = ""

    @property
    def state(self) -> ExerciseState:
        """Confirmed exercise state for this frame."""
        return self.counter.state if self.counter is not None else ExerciseState.UNKNOWN

    @property
    def rep_count(self) -> int:
        """Valid repetitions counted so far."""
        return self.counter.rep_count if self.counter is not None else 0

    @property
    def form_score(self) -> float:
        """Current form score (``0`` when unavailable)."""
        return self.form.form_score if self.form is not None else 0.0

    @property
    def feedback(self) -> List[str]:
        """Feedback lines to display."""
        return list(self.form.feedback) if self.form is not None else []


class FrameAnalyzer:
    """Runs the detection -> analysis -> scoring chain on a single frame."""

    def __init__(
        self,
        settings: Settings,
        *,
        pose_detector: Optional[PoseDetector] = None,
        counter: Optional[PushUpCounter] = None,
        form_analyzer: Optional[FormAnalyzer] = None,
        session: Optional[SessionTracker] = None,
    ) -> None:
        self.settings = settings
        self.pose_detector = pose_detector or PoseDetector(settings)
        self.form_analyzer = form_analyzer or FormAnalyzer(settings.form, settings.posture, settings.movement)
        self.counter = counter or PushUpCounter(
            copy.replace(settings.pushup),
            copy.replace(settings.validation),
            smoothing_config=settings.smoothing,
            movement_config=settings.movement,
            form_analyzer=self.form_analyzer,
        )
        self.session = session or SessionTracker(
            history_stride=settings.analytics.history_stride,
            max_history_points=settings.analytics.max_history_points,
        )
        self.fps_meter = FPSMeter()
        self._frame_id = 0
        self._previous_rep_count = 0
        self._last_form_score = float("nan")

    # ------------------------------------------------------------------ model
    def load_model(self) -> None:
        """Load the pose model (call once before processing)."""
        self.pose_detector.load_model()

    # ---------------------------------------------------------------- process
    def process_frame(self, frame_bgr: np.ndarray, timestamp: Optional[float] = None) -> FrameResult:
        """Analyse one frame.

        Args:
            frame_bgr: Raw BGR frame from any :class:`FrameSource`.
            timestamp: Frame timestamp in seconds (defaults to the wall clock).

        Returns:
            A :class:`FrameResult` with pose, angles, state, form and counters.
        """
        stamp = time.time() if timestamp is None else float(timestamp)
        frame_id = self._frame_id
        self._frame_id += 1

        if not _is_valid_frame(frame_bgr):
            logger.warning("Dropping invalid frame %s", frame_id)
            return FrameResult(frame_id=frame_id, timestamp=stamp, frame=frame_bgr, message="Corrupted frame skipped")

        pose_frame = self.pose_detector.detect_all(frame_bgr)
        pose = pose_frame.primary

        angles = compute_angles(pose)
        raw_elbow = angles.primary_elbow()
        update = self.counter.update(raw_elbow, angles, angles=angles, pose=pose, timestamp=stamp)
        self._last_form_score = update.form.form_score if update.form is not None else self._last_form_score

        result = FrameResult(
            frame_id=frame_id,
            timestamp=stamp,
            frame=frame_bgr,
            pose_frame=pose_frame,
            pose=pose,
            angles=angles,
            movement=update.movement,
            posture=update.posture,
            form=update.form,
            counter=update,
            rep_completed=update.rep_completed,
            last_validation=update.last_validation,
            fps=self.fps_meter.tick(),
            inference_ms=pose_frame.inference_ms,
            visible_keypoints=pose.valid_count if pose is not None else 0,
            multiple_people=pose_frame.multiple_people,
            has_pose=pose is not None,
            detection_confidence=update.detection_confidence,
            message=update.message,
        )

        self._record_frame(result)
        if update.rep_completed:
            self._record_rep(update, stamp)
        self._previous_rep_count = update.rep_count
        result.fps = self.fps_meter.fps
        return result

    # ---------------------------------------------------------------- analytics
    def _record_frame(self, result: FrameResult) -> None:
        angles = result.angles
        form = result.form
        update = result.counter
        self.session.record_frame(
            FrameRecord(
                frame_id=result.frame_id,
                timestamp=result.timestamp,
                left_elbow=angles.left_elbow,
                right_elbow=angles.right_elbow,
                elbow_mean=angles.elbow_mean,
                elbow_diff=angles.elbow_diff,
                body_alignment=angles.body_alignment,
                alignment_deviation=angles.alignment_deviation,
                form_score=form.form_score if form is not None else float("nan"),
                form_status=form.form_status.value if form is not None else "",
                depth_score=form.depth_score if form is not None else float("nan"),
                alignment_score=form.body_alignment_score if form is not None else float("nan"),
                symmetry_score=form.symmetry_score if form is not None else float("nan"),
                control_score=form.movement_score if form is not None else float("nan"),
                rep_count=update.rep_count if update is not None else 0,
                valid_reps=update.valid_reps if update is not None else 0,
                invalid_reps=update.invalid_reps if update is not None else 0,
                state=result.state.value,
                has_pose=result.has_pose,
                detection_confidence=result.detection_confidence,
                fps=result.fps,
                inference_ms=result.inference_ms,
            )
        )

    def _record_rep(self, update: CounterUpdate, stamp: float) -> None:
        validation = update.last_validation
        history = self.counter.history[-1] if self.counter.history else {}
        cycle_payload = history.get("cycle", {}) if isinstance(history, dict) else {}
        form_score = update.form.form_score if update.form is not None else float("nan")
        self.session.record_rep(
            RepRecord(
                timestamp=stamp,
                valid=bool(validation.valid) if validation is not None else False,
                reason=validation.reason if validation is not None else "",
                code=validation.code if validation is not None else "",
                duration_s=float(cycle_payload.get("duration_s", 0.0) or 0.0),
                min_elbow_angle=float(cycle_payload.get("min_elbow_angle", float("nan")) or float("nan")),
                max_elbow_angle=float(cycle_payload.get("max_elbow_angle", float("nan")) or float("nan")),
                elbow_travel=float(cycle_payload.get("elbow_travel", float("nan")) or float("nan")),
                max_alignment_deviation=float(cycle_payload.get("max_alignment_deviation", float("nan")) or float("nan")),
                max_arm_diff=float(cycle_payload.get("max_arm_diff", float("nan")) or float("nan")),
                tracking_ratio=float(cycle_payload.get("tracking_ratio", 1.0) or 1.0),
                form_score=form_score,
            )
        )

    # ------------------------------------------------------------------ utils
    def reset(self, *, full: bool = True) -> None:
        """Reset counters, analytics and the detector session state."""
        self.counter.reset(full=full)
        self.session.reset()
        self.pose_detector.reset()
        self.fps_meter.reset()
        self._frame_id = 0
        self._previous_rep_count = 0

    def build_hud_data(self, result: FrameResult) -> HudData:
        """Convert a :class:`FrameResult` into overlay data."""
        validation = result.last_validation
        return HudData(
            rep_count=result.rep_count,
            valid_reps=result.counter.valid_reps if result.counter else 0,
            invalid_reps=result.counter.invalid_reps if result.counter else 0,
            state=result.state,
            raw_state=result.movement.raw_position if result.movement else ExerciseState.UNKNOWN,
            detection_confidence=result.detection_confidence,
            visible_keypoints=result.visible_keypoints,
            total_keypoints=NUM_KEYPOINTS,
            fps=result.fps,
            inference_ms=result.inference_ms,
            form_score=result.form_score,
            form_grade=result.form.form_status if result.form is not None else FormGrade.POOR,
            elbow_angle=result.angles.primary_elbow(),
            body_alignment=result.angles.body_alignment,
            arm_diff=result.angles.elbow_diff,
            feedback=result.feedback,
            last_rep_valid=None if validation is None else validation.valid,
            last_rep_reason=validation.reason if validation is not None else "",
            has_pose=result.has_pose,
            multiple_people=result.multiple_people,
            message=result.message,
        )


class FrameRenderer:
    """Draws the skeleton and the HUD onto a frame."""

    def __init__(self, settings: Settings, analyzer: Optional[FrameAnalyzer] = None) -> None:
        self.settings = settings
        self.analyzer = analyzer
        self.palette = Palette(settings.visualization)
        self.skeleton = SkeletonDrawer(
            settings.visualization,
            self.palette,
            alignment_warning=settings.posture.alignment_tolerance,
            alignment_critical=settings.posture.alignment_max_deviation,
        )
        self.overlay = OverlayRenderer(settings.visualization, self.palette)

    def render(self, frame_bgr: np.ndarray, result: FrameResult, *, copy: bool = True) -> np.ndarray:
        """Annotate a frame with skeleton + HUD.

        Args:
            frame_bgr: The raw frame.
            result: Analysis result for that frame.
            copy: Work on a copy (keep the original untouched).

        Returns:
            The annotated BGR frame.
        """
        canvas = frame_bgr.copy() if copy else frame_bgr
        self.skeleton.draw(
            canvas,
            result.pose,
            result.angles,
            draw_angles=self.settings.visualization.hud.show_angle_labels,
        )
        if self.analyzer is not None:
            hud = self.analyzer.build_hud_data(result)
        else:
            hud = HudData(state=result.state, rep_count=result.rep_count, has_pose=result.has_pose)
        self.overlay.draw(canvas, hud)
        return canvas


def _is_valid_frame(frame: Optional[np.ndarray]) -> bool:
    return frame is not None and isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.size > 0


@dataclass(slots=True)
class ProcessingReport:
    """Outcome of a batch run over a video source."""

    frames_processed: int = 0
    output_video: Optional[Path] = None
    output_fps: float = 0.0
    duration_seconds: float = 0.0
    duration_label: str = "00:00"
    rep_count: int = 0
    valid_reps: int = 0
    invalid_reps: int = 0
    average_form_score: float = 0.0
    average_fps: float = 0.0
    error: str = ""


def _default_export_path_for(settings: Settings, export_format: Optional[str]) -> Path:
    """Build the default export filename using the requested container format."""
    fmt = (export_format or settings.video.export_format or "mp4").lstrip(".").lower()
    name = settings.output.video_name_template.format(timestamp=timestamp_slug())
    return settings.videos_dir / f"{Path(name).stem}.{fmt}"


def process_source(
    source: FrameSource,
    settings: Settings,
    *,
    analyzer: Optional[FrameAnalyzer] = None,
    renderer: Optional[FrameRenderer] = None,
    export_video: bool = True,
    output_path: Optional[str | Path] = None,
    max_frames: Optional[int] = None,
    export_format: Optional[str] = None,
    on_frame: Optional[Callable[[FrameResult, np.ndarray], None]] = None,
) -> Tuple[ProcessingReport, FrameAnalyzer]:
    """Run the full pipeline over any :class:`FrameSource`.

    Args:
        source: Video file, webcam or demo source.
        settings: Application settings.
        analyzer: Optional pre-built analyzer (reuse across runs).
        renderer: Optional pre-built renderer.
        export_video: Write the annotated video to ``outputs/videos``.
        output_path: Optional explicit export path.
        max_frames: Stop after this many frames (used by tests and previews).
        on_frame: Optional callback ``(result, annotated_frame)`` per frame.

    Returns:
        ``(report, analyzer)`` so the caller can reuse the session data.

    Raises:
        VideoSourceError: When the source cannot be opened.
    """
    analyzer = analyzer or FrameAnalyzer(settings)
    renderer = renderer or FrameRenderer(settings, analyzer)
    analyzer.load_model()
    open_source(source)

    fps = source.fps or settings.video.output_fps
    export_fps = fps if fps > 0 else settings.video.output_fps
    exporter: Optional[VideoExporter] = None
    if export_video and settings.video.save_processed_video:
        target = Path(output_path) if output_path else _default_export_path_for(settings, export_format)
        exporter = VideoExporter(
            target,
            (source.width or 1280, source.height or 720),
            export_fps,
            settings.video.fourcc,
            settings.video.fallback_fourccs,
        )
        if not exporter.open():
            exporter = None

    report = ProcessingReport(output_fps=export_fps)
    started = time.perf_counter()
    frame_index = 0
    try:
        while True:
            success, frame = source.read()
            if not success or frame is None:
                break
            # Drive analysis timing from the video timeline (frame / fps), not the
            # wall clock: a recorded clip must be analysed at its own cadence even
            # when we decode it faster or slower than real time.
            stamp = frame_index / export_fps if export_fps > 0 else None
            result = analyzer.process_frame(frame, timestamp=stamp)
            annotated = renderer.render(frame, result)
            if exporter is not None:
                exporter.write(annotated)
            if on_frame is not None:
                on_frame(result, annotated)
            report.frames_processed += 1
            frame_index += 1
            if max_frames is not None and report.frames_processed >= max_frames:
                break
    except VideoSourceError:
        raise
    except Exception as exc:  # noqa: BLE001 - report, never crash the caller
        logger.exception("Processing loop failed")
        report.error = f"{type(exc).__name__}: {exc}"
    finally:
        source.release()
        if exporter is not None:
            report.output_video = exporter.close()

    summary = analyzer.session.summary()
    report.duration_seconds = time.perf_counter() - started
    report.duration_label = format_duration(report.duration_seconds)
    report.rep_count = summary.total_reps
    report.valid_reps = summary.valid_reps
    report.invalid_reps = summary.invalid_reps
    report.average_form_score = summary.average_form_score
    report.average_fps = analyzer.fps_meter.average_fps
    logger.info(
        "Processed %d frames: %d reps (%d valid), avg form %.1f, %.1f fps",
        report.frames_processed,
        report.rep_count,
        report.valid_reps,
        report.average_form_score,
        report.average_fps,
    )
    return report, analyzer


def process_video_file(
    path: str | Path,
    settings: Settings,
    **kwargs,
) -> Tuple[ProcessingReport, FrameAnalyzer]:
    """Convenience wrapper: analyse a video file end to end."""
    from src.video.sources import VideoFileSource

    source = VideoFileSource(path, fallback_fps=settings.video.output_fps)
    return process_source(source, settings, **kwargs)


def capture_screenshot(frame_bgr: np.ndarray, settings: Settings, prefix: str = "screenshot") -> Path:
    """Save an annotated frame into ``outputs/screenshots``."""
    return save_screenshot(frame_bgr, settings.screenshots_dir, prefix=prefix, image_format=settings.output.screenshot_format)


__all__ = [
    "FrameAnalyzer",
    "FrameRenderer",
    "FrameResult",
    "ProcessingReport",
    "process_source",
    "process_video_file",
    "capture_screenshot",
]

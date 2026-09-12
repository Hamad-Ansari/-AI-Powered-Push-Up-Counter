"""Workout session tracking: per-frame history, rep log and summary statistics."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.helpers import ensure_dir, format_duration, moving_statistics, timestamp_slug
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class FrameRecord:
    """Per-frame telemetry stored for the analytics charts."""

    frame_id: int = 0
    timestamp: float = 0.0
    left_elbow: float = float("nan")
    right_elbow: float = float("nan")
    elbow_mean: float = float("nan")
    elbow_diff: float = float("nan")
    body_alignment: float = float("nan")
    alignment_deviation: float = float("nan")
    form_score: float = float("nan")
    form_status: str = ""
    depth_score: float = float("nan")
    alignment_score: float = float("nan")
    symmetry_score: float = float("nan")
    control_score: float = float("nan")
    rep_count: int = 0
    valid_reps: int = 0
    invalid_reps: int = 0
    state: str = "UNKNOWN"
    has_pose: bool = False
    detection_confidence: float = 0.0
    fps: float = 0.0
    inference_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly snapshot."""
        payload = asdict(self)
        return {key: (None if isinstance(value, float) and math.isnan(value) else value) for key, value in payload.items()}


@dataclass(slots=True)
class RepRecord:
    """One completed repetition attempt (valid or invalid)."""

    index: int = 0
    timestamp: float = 0.0
    valid: bool = False
    reason: str = ""
    code: str = ""
    duration_s: float = 0.0
    min_elbow_angle: float = float("nan")
    max_elbow_angle: float = float("nan")
    elbow_travel: float = float("nan")
    max_alignment_deviation: float = float("nan")
    max_arm_diff: float = float("nan")
    tracking_ratio: float = 1.0
    form_score: float = float("nan")

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly snapshot."""
        payload = asdict(self)
        return {key: (None if isinstance(value, float) and math.isnan(value) else value) for key, value in payload.items()}


@dataclass(slots=True)
class SessionSummary:
    """Aggregate statistics for the whole workout."""

    total_reps: int = 0
    valid_reps: int = 0
    invalid_reps: int = 0
    accuracy: float = 0.0
    average_form_score: float = 0.0
    best_form_score: float = 0.0
    lowest_form_score: float = 0.0
    average_left_elbow: float = float("nan")
    average_right_elbow: float = float("nan")
    average_elbow_angle: float = float("nan")
    minimum_elbow_angle: float = float("nan")
    average_body_alignment: float = float("nan")
    average_arm_diff: float = float("nan")
    duration_seconds: float = 0.0
    duration_label: str = "00:00"
    reps_per_minute: float = 0.0
    average_rep_duration: float = 0.0
    frames_processed: int = 0
    frames_with_pose: int = 0
    detection_rate: float = 0.0
    average_fps: float = 0.0
    average_inference_ms: float = 0.0
    time_up: float = 0.0
    time_down: float = 0.0
    time_moving: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly snapshot used by the reports."""
        payload = asdict(self)
        return {key: (None if isinstance(value, float) and math.isnan(value) else value) for key, value in payload.items()}

    def headline_rows(self) -> List[tuple[str, str]]:
        """Ordered ``(label, value)`` rows for the dashboard summary card."""
        return [
            ("Total Push-Ups", str(self.total_reps)),
            ("Valid Reps", str(self.valid_reps)),
            ("Invalid Reps", str(self.invalid_reps)),
            ("Rep Accuracy", f"{self.accuracy:.0f}%"),
            ("Average Form Score", f"{self.average_form_score:.1f}"),
            ("Best Form Score", f"{self.best_form_score:.1f}"),
            ("Lowest Form Score", f"{self.lowest_form_score:.1f}"),
            ("Average Elbow Angle", _deg(self.average_elbow_angle)),
            ("Average Body Alignment", _deg(self.average_body_alignment)),
            ("Duration", self.duration_label),
            ("Reps / Minute", f"{self.reps_per_minute:.1f}"),
            ("Average Rep Duration", f"{self.average_rep_duration:.2f}s"),
            ("Frames Processed", str(self.frames_processed)),
            ("Pose Detection Rate", f"{self.detection_rate * 100:.0f}%"),
            ("Average FPS", f"{self.average_fps:.1f}"),
        ]


def _deg(value: float) -> str:
    return f"{value:.1f}°" if math.isfinite(value) else "--"


class SessionTracker:
    """Accumulates frame telemetry and repetition records for one session.

    Example:
        >>> tracker = SessionTracker()
        >>> tracker.record_frame(FrameRecord(frame_id=0, rep_count=1, form_score=91.0, has_pose=True))
        >>> tracker.summary().total_reps
        1
    """

    def __init__(self, *, history_stride: int = 2, max_history_points: int = 1500) -> None:
        self.history_stride = max(1, int(history_stride))
        self.max_history_points = max(50, int(max_history_points))
        self.frames: List[FrameRecord] = []
        self.reps: List[RepRecord] = []
        self._frame_counter = 0
        self._started_at: Optional[float] = None
        self._last_timestamp: float = 0.0
        self._fps_sum = 0.0
        self._fps_count = 0
        self._inference_sum = 0.0
        self._inference_count = 0
        self._state_time: Dict[str, float] = {"UP": 0.0, "DOWN": 0.0, "MOVING": 0.0, "UNKNOWN": 0.0}
        self._last_state: str = "UNKNOWN"
        self._last_state_stamp: float = 0.0

    # ------------------------------------------------------------------ record
    def record_frame(self, record: FrameRecord) -> None:
        """Store one frame of telemetry (down-sampled by ``history_stride``)."""
        self._frame_counter += 1
        stamp = record.timestamp if math.isfinite(record.timestamp) else None
        if self._started_at is None:
            self._started_at = time.time() if stamp is None else stamp
        if stamp is not None:
            self._last_timestamp = stamp

        if math.isfinite(record.fps) and record.fps > 0:
            self._fps_sum += record.fps
            self._fps_count += 1
        if math.isfinite(record.inference_ms) and record.inference_ms > 0:
            self._inference_sum += record.inference_ms
            self._inference_count += 1

        self._accumulate_state_time(record)

        keep = (record.frame_id % self.history_stride == 0) or record.rep_count > (self.frames[-1].rep_count if self.frames else 0)
        if keep:
            self.frames.append(record)
            if len(self.frames) > self.max_history_points:
                self.frames = self.frames[-self.max_history_points :]

    def record_rep(self, record: RepRecord) -> None:
        """Store one completed repetition attempt."""
        record.index = len(self.reps) + 1
        self.reps.append(record)

    def _accumulate_state_time(self, record: FrameRecord) -> None:
        stamp = record.timestamp
        if self._last_state_stamp and stamp > self._last_state_stamp:
            delta = min(1.0, stamp - self._last_state_stamp)
            self._state_time[self._last_state] = self._state_time.get(self._last_state, 0.0) + delta
        self._last_state = record.state
        self._last_state_stamp = stamp

    # ----------------------------------------------------------------- summary
    def summary(self) -> SessionSummary:
        """Compute the aggregate workout statistics."""
        form_scores = [frame.form_score for frame in self.frames if math.isfinite(frame.form_score)]
        left_elbows = [frame.left_elbow for frame in self.frames if math.isfinite(frame.left_elbow)]
        right_elbows = [frame.right_elbow for frame in self.frames if math.isfinite(frame.right_elbow)]
        elbow_means = [frame.elbow_mean for frame in self.frames if math.isfinite(frame.elbow_mean)]
        alignments = [frame.body_alignment for frame in self.frames if math.isfinite(frame.body_alignment)]
        arm_diffs = [frame.elbow_diff for frame in self.frames if math.isfinite(frame.elbow_diff)]

        total_reps = self.reps[-1].index if self.reps else 0
        valid_reps = sum(1 for rep in self.reps if rep.valid)
        invalid_reps = sum(1 for rep in self.reps if not rep.valid)
        durations = [rep.duration_s for rep in self.reps if rep.valid and rep.duration_s > 0]

        started = self._started_at if self._started_at is not None else self._last_timestamp
        duration = max(0.0, self._last_timestamp - started)
        frames_with_pose = sum(1 for frame in self.frames if frame.has_pose)

        summary = SessionSummary(
            total_reps=total_reps,
            valid_reps=valid_reps,
            invalid_reps=invalid_reps,
            accuracy=(valid_reps / total_reps * 100.0) if total_reps else 0.0,
            average_form_score=float(sum(form_scores) / len(form_scores)) if form_scores else 0.0,
            best_form_score=max(form_scores) if form_scores else 0.0,
            lowest_form_score=min(form_scores) if form_scores else 0.0,
            average_left_elbow=_mean(left_elbows),
            average_right_elbow=_mean(right_elbows),
            average_elbow_angle=_mean(elbow_means),
            minimum_elbow_angle=min(elbow_means) if elbow_means else float("nan"),
            average_body_alignment=_mean(alignments),
            average_arm_diff=_mean(arm_diffs),
            duration_seconds=duration,
            duration_label=format_duration(duration),
            reps_per_minute=(valid_reps * 60.0 / duration) if duration > 0 else 0.0,
            average_rep_duration=float(sum(durations) / len(durations)) if durations else 0.0,
            frames_processed=self._frame_counter,
            frames_with_pose=frames_with_pose,
            detection_rate=(frames_with_pose / len(self.frames)) if self.frames else 0.0,
            average_fps=(self._fps_sum / self._fps_count) if self._fps_count else 0.0,
            average_inference_ms=(self._inference_sum / self._inference_count) if self._inference_count else 0.0,
            time_up=self._state_time.get("UP", 0.0),
            time_down=self._state_time.get("DOWN", 0.0),
            time_moving=self._state_time.get("MOVING", 0.0),
        )
        return summary

    # ------------------------------------------------------------------ export
    def to_dataframe(self):
        """Frame history as a :class:`pandas.DataFrame` (for Plotly and reports)."""
        import pandas as pd

        if not self.frames:
            return pd.DataFrame(
                columns=[
                    "frame_id",
                    "timestamp",
                    "left_elbow",
                    "right_elbow",
                    "elbow_mean",
                    "elbow_diff",
                    "body_alignment",
                    "alignment_deviation",
                    "form_score",
                    "form_status",
                    "rep_count",
                    "state",
                    "has_pose",
                    "fps",
                ]
            )
        return pd.DataFrame([record.to_dict() for record in self.frames])

    def reps_dataframe(self):
        """Repetition log as a :class:`pandas.DataFrame`."""
        import pandas as pd

        if not self.reps:
            return pd.DataFrame(columns=["index", "timestamp", "valid", "reason", "duration_s", "min_elbow_angle", "form_score"])
        return pd.DataFrame([record.to_dict() for record in self.reps])

    def to_dict(self) -> Dict[str, Any]:
        """Full session payload (summary + history + reps)."""
        return {
            "summary": self.summary().to_dict(),
            "reps": [record.to_dict() for record in self.reps],
            "frames": [record.to_dict() for record in self.frames],
        }

    def reset(self) -> None:
        """Start a new session."""
        self.frames.clear()
        self.reps.clear()
        self._frame_counter = 0
        self._started_at = None
        self._last_timestamp = 0.0
        self._fps_sum = 0.0
        self._fps_count = 0
        self._inference_sum = 0.0
        self._inference_count = 0
        self._state_time = {"UP": 0.0, "DOWN": 0.0, "MOVING": 0.0, "UNKNOWN": 0.0}
        self._last_state = "UNKNOWN"
        self._last_state_stamp = 0.0


def _mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def export_session_json(tracker: SessionTracker, reports_dir: str | Path, prefix: str = "workout_session") -> Path:
    """Write the full session payload to ``outputs/reports`` as JSON."""
    directory = ensure_dir(reports_dir)
    path = directory / f"{prefix}_{timestamp_slug()}.json"
    path.write_text(json.dumps(tracker.to_dict(), indent=2), encoding="utf-8")
    logger.info("Session JSON written: %s", path.name)
    return path


__all__ = [
    "SessionTracker",
    "SessionSummary",
    "FrameRecord",
    "RepRecord",
    "export_session_json",
    "moving_statistics",
]

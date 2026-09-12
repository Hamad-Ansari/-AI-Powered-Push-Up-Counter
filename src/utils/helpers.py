"""Small, dependency-light helpers shared across the application."""

from __future__ import annotations

import math
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, Optional, Sequence, Tuple

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_VIDEO_EXTENSIONS: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg")
Point2D = Tuple[float, float]


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def midpoint(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> Optional[Point2D]:
    """Return the midpoint of two ``[x, y]`` points, or ``None`` if either is missing."""
    if a is None or b is None:
        return None
    return (float(a[0]) + float(b[0])) / 2.0, (float(a[1]) + float(b[1])) / 2.0


def is_valid_point(point: Optional[Sequence[float]], frame_shape: Optional[Sequence[int]] = None) -> bool:
    """Check that a point exists, is finite and (optionally) inside the frame."""
    if point is None:
        return False
    try:
        x, y = float(point[0]), float(point[1])
    except (TypeError, ValueError, IndexError):
        return False
    if not (math.isfinite(x) and math.isfinite(y)):
        return False
    if frame_shape is not None:
        height, width = int(frame_shape[0]), int(frame_shape[1])
        margin = max(width, height) * 0.5  # allow generous off-screen extrapolation
        if not (-margin <= x <= width + margin and -margin <= y <= height + margin):
            return False
    return True


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp ``value`` into ``[low, high]``."""
    return max(low, min(high, value))


def linear_score(value: float, best: float, worst: float) -> float:
    """Map ``value`` linearly to ``[0, 1]`` where ``best`` -> 1 and ``worst`` -> 0.

    Works in both directions (``best`` may be larger or smaller than ``worst``).
    """
    if abs(worst - best) < 1e-9:
        return 1.0
    return clamp((worst - value) / (worst - best), 0.0, 1.0)


def box_iou(box_a: Optional[Sequence[float]], box_b: Optional[Sequence[float]]) -> float:
    """Intersection-over-union of two ``[x1, y1, x2, y2]`` boxes."""
    if box_a is None or box_b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = (float(v) for v in box_a)
    bx1, by1, bx2, by2 = (float(v) for v in box_b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 1e-9 else 0.0


def box_area(box: Optional[Sequence[float]]) -> float:
    """Area of an ``[x1, y1, x2, y2]`` box."""
    if box is None:
        return 0.0
    x1, y1, x2, y2 = (float(v) for v in box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


# --------------------------------------------------------------------------- #
# Files / timestamps
# --------------------------------------------------------------------------- #
def timestamp_slug(moment: Optional[datetime] = None, fmt: str = "%Y%m%d_%H%M%S") -> str:
    """Filesystem-safe timestamp, e.g. ``20260910_183045``."""
    return (moment or datetime.now()).strftime(fmt)


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if required and return it as :class:`Path`."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def is_supported_video(filename: str | Path) -> bool:
    """Return ``True`` when the extension is a container OpenCV can usually read."""
    return Path(filename).suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS


def format_duration(seconds: float) -> str:
    """Format seconds as ``MM:SS`` (or ``H:MM:SS`` for long sessions)."""
    seconds = max(0.0, float(seconds))
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def safe_float(value: object, default: float = float("nan")) -> float:
    """Best-effort float conversion that never raises."""
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
class FPSMeter:
    """Rolling FPS / frame-time monitor.

    Example:
        >>> meter = FPSMeter(window_size=3)
        >>> for _ in range(3):
        ...     _ = meter.tick()
        >>> meter.fps > 0
        True
    """

    __slots__ = ("_deltas", "_last", "_total_frames", "_total_time", "_window")

    def __init__(self, window_size: int = 30) -> None:
        self._window = max(2, int(window_size))
        self._deltas: Deque[float] = deque(maxlen=self._window)
        self._last: Optional[float] = None
        self._total_frames: int = 0
        self._total_time: float = 0.0

    def tick(self, now: Optional[float] = None) -> float:
        """Register a processed frame and return the instantaneous FPS."""
        now = time.perf_counter() if now is None else now
        if self._last is not None:
            delta = now - self._last
            if delta > 0:
                self._deltas.append(delta)
                self._total_time += delta
                self._total_frames += 1
        self._last = now
        return self.fps

    @property
    def fps(self) -> float:
        """Smoothed frames-per-second over the rolling window."""
        if not self._deltas:
            return 0.0
        return len(self._deltas) / float(sum(self._deltas))

    @property
    def frame_ms(self) -> float:
        """Mean milliseconds per frame over the rolling window."""
        if not self._deltas:
            return 0.0
        return 1000.0 * (sum(self._deltas) / len(self._deltas))

    @property
    def average_fps(self) -> float:
        """Average FPS across the whole session."""
        if self._total_time <= 0 or self._total_frames == 0:
            return 0.0
        return self._total_frames / self._total_time

    def reset(self) -> None:
        """Clear all accumulated timing statistics."""
        self._deltas.clear()
        self._last = None
        self._total_frames = 0
        self._total_time = 0.0


def moving_statistics(values: Sequence[float]) -> dict[str, float]:
    """Return ``min``/``max``/``mean`` of a sequence (NaN-safe)."""
    if not values:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan")}
    array = np.asarray([v for v in values if math.isfinite(float(v))], dtype=float)
    if array.size == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan")}
    return {"min": float(array.min()), "max": float(array.max()), "mean": float(array.mean())}

"""Angle smoothing.

Pose estimators produce jittery joint positions, which shows up as a noisy elbow
angle. Classification therefore always runs on a *smoothed* signal. Two filters
are supported and selectable from ``config.yaml``:

* ``moving_average`` - mean of the last ``window_size`` samples.
* ``ema`` - exponential moving average with factor ``ema_alpha``.

Both filters are NaN-aware: a frame without a pose never poisons the history.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, List, Optional

from config.settings import SmoothingConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AngleSmoother(ABC):
    """Base class for 1-D signal smoothers."""

    name: str = "base"

    @abstractmethod
    def update(self, value: Optional[float]) -> Optional[float]:
        """Push a new raw sample and return the smoothed value.

        Args:
            value: Raw sample, or ``None`` when the frame had no usable pose.

        Returns:
            The smoothed value, or ``None`` when nothing has been seen yet.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear the filter state."""

    @property
    @abstractmethod
    def current(self) -> Optional[float]:
        """Last smoothed value (``None`` before the first sample)."""


class MovingAverageSmoother(AngleSmoother):
    """Simple moving average over the last ``window_size`` samples."""

    name = "moving_average"

    def __init__(self, window_size: int = 5) -> None:
        self.window_size = max(1, int(window_size))
        self._history: Deque[float] = deque(maxlen=self.window_size)
        self._count = 0

    def update(self, value: Optional[float]) -> Optional[float]:
        """Append ``value`` (when usable) and return the window mean."""
        if value is None:
            return self.current
        try:
            sample = float(value)
        except (TypeError, ValueError):
            return self.current
        if sample != sample:  # NaN check without importing math
            return self.current
        self._history.append(sample)
        self._count += 1
        return self.current

    @property
    def current(self) -> Optional[float]:
        if not self._history:
            return None
        return sum(self._history) / len(self._history)

    def history(self) -> List[float]:
        """Snapshot of the current window (oldest first)."""
        return list(self._history)

    def reset(self) -> None:
        """Empty the window."""
        self._history.clear()
        self._count = 0


class ExponentialMovingAverageSmoother(AngleSmoother):
    """Exponential moving average: ``s_t = a * x_t + (1 - a) * s_{t-1}``."""

    name = "ema"

    def __init__(self, alpha: float = 0.3) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"EMA alpha must be in (0, 1], got {alpha}")
        self.alpha = float(alpha)
        self._value: Optional[float] = None
        self._count = 0

    def update(self, value: Optional[float]) -> Optional[float]:
        """Fold ``value`` into the running estimate and return it."""
        if value is None:
            return self._value
        try:
            sample = float(value)
        except (TypeError, ValueError):
            return self._value
        if sample != sample:
            return self._value
        self._count += 1
        if self._value is None:
            self._value = sample
        else:
            self._value = self.alpha * sample + (1.0 - self.alpha) * self._value
        return self._value

    @property
    def current(self) -> Optional[float]:
        return self._value

    def reset(self) -> None:
        """Forget the running estimate."""
        self._value = None
        self._count = 0


class PassthroughSmoother(AngleSmoother):
    """No-op smoother used when smoothing is disabled."""

    name = "none"

    def __init__(self) -> None:
        self._value: Optional[float] = None
        self._count = 0

    def update(self, value: Optional[float]) -> Optional[float]:
        """Return the raw value unchanged (``None`` stays ``None``)."""
        if value is None or value != value:
            self._value = None
            return None
        self._value = float(value)
        self._count += 1
        return self._value

    @property
    def current(self) -> Optional[float]:
        return self._value

    def reset(self) -> None:
        """Clear the cached value."""
        self._value = None
        self._count = 0


def create_smoother(config: SmoothingConfig) -> AngleSmoother:
    """Build the smoother selected in the configuration.

    Args:
        config: Smoothing section of the settings.

    Returns:
        A ready-to-use :class:`AngleSmoother`.
    """
    if not config.enabled:
        return PassthroughSmoother()
    method = (config.method or "moving_average").lower()
    if method in {"moving_average", "ma", "window"}:
        return MovingAverageSmoother(config.window_size)
    if method in {"ema", "exponential"}:
        return ExponentialMovingAverageSmoother(config.ema_alpha)
    logger.warning("Unknown smoothing method %r - falling back to moving average", config.method)
    return MovingAverageSmoother(config.window_size)


def smooth_series(values: List[Optional[float]], config: SmoothingConfig) -> List[Optional[float]]:
    """Smooth a whole series offline (used by analytics and tests)."""
    smoother = create_smoother(config)
    return [smoother.update(value) for value in values]


__all__ = [
    "AngleSmoother",
    "MovingAverageSmoother",
    "ExponentialMovingAverageSmoother",
    "PassthroughSmoother",
    "create_smoother",
    "smooth_series",
]

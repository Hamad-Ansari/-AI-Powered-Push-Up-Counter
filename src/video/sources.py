"""Video sources: uploaded files, webcams and the bundled demo stream.

Every source implements the same tiny protocol so the processing pipeline never
cares where the frames come from::

    source.open() -> bool
    source.read() -> tuple[bool, np.ndarray | None]
    source.release() -> None
"""

from __future__ import annotations

import math
import platform
from pathlib import Path
from typing import Optional, Protocol, Tuple, runtime_checkable

import cv2
import numpy as np

from src.pose.demo_adapter import generate_pushup_keypoints
from src.pose.keypoints import KEYPOINT_NAMES, skeleton_pairs
from src.utils.helpers import is_supported_video
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VideoSourceError(RuntimeError):
    """Raised when a video source cannot be opened or read."""


@runtime_checkable
class FrameSource(Protocol):
    """Minimal contract implemented by every frame source."""

    fps: float
    width: int
    height: int
    frame_count: int
    name: str

    def open(self) -> bool:
        """Open the source; ``False`` when it is unavailable."""

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next frame as ``(success, frame_bgr)``."""

    def release(self) -> None:
        """Release underlying resources."""


def preferred_backend() -> int:
    """Best OpenCV capture backend for the current OS."""
    system = platform.system()
    if system == "Windows":
        return cv2.CAP_DSHOW
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    if system == "Linux":
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


class VideoFileSource:
    """Reads a video file (MP4/AVI/MOV/...) with optional looping."""

    def __init__(self, path: str | Path, *, loop: bool = False, fallback_fps: float = 30.0) -> None:
        self.path = Path(path)
        self.loop = loop
        self.fallback_fps = float(fallback_fps)
        self.name = self.path.name
        self._capture: Optional[cv2.VideoCapture] = None
        self.fps: float = fallback_fps
        self.width: int = 0
        self.height: int = 0
        self.frame_count: int = 0
        self.frames_read: int = 0

    def open(self) -> bool:
        """Open the file and read its metadata.

        Raises:
            VideoSourceError: If the file is missing, unsupported or unreadable.
        """
        if not self.path.exists():
            raise VideoSourceError(f"Video file not found: {self.path}")
        if not is_supported_video(self.path):
            logger.warning("Unsupported video extension: %s (attempting to open anyway)", self.path.suffix)

        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise VideoSourceError(
                f"OpenCV could not decode '{self.path.name}'. The container may be unsupported or the file corrupted."
            )

        self._capture = capture
        fps = capture.get(cv2.CAP_PROP_FPS)
        self.fps = float(fps) if fps and math.isfinite(fps) and fps > 0 else self.fallback_fps
        self.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frames_read = 0
        logger.info(
            "Opened %s (%dx%d @ %.2f fps, %d frames)",
            self.path.name,
            self.width,
            self.height,
            self.fps,
            self.frame_count,
        )
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next frame; rewinds when looping is enabled."""
        if self._capture is None:
            return False, None
        success, frame = self._capture.read()
        if not success or frame is None:
            if self.loop:
                self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = self._capture.read()
                if not success or frame is None:
                    return False, None
                self.frames_read = 0
            else:
                return False, None
        if frame.ndim != 3 or frame.shape[2] != 3:
            logger.warning("Skipping malformed frame at index %s", self.frames_read)
            return self.read()
        self.frames_read += 1
        return True, frame

    def release(self) -> None:
        """Close the file handle."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    @property
    def is_open(self) -> bool:
        """``True`` while the capture is usable."""
        return self._capture is not None and self._capture.isOpened()


class WebcamSource:
    """Reads from a local webcam index."""

    def __init__(
        self,
        index: int = 0,
        *,
        width: int = 1280,
        height: int = 720,
        backend: Optional[int] = None,
        fallback_fps: float = 30.0,
    ) -> None:
        self.index = int(index)
        self.requested_size = (int(width), int(height))
        self.backend = preferred_backend() if backend is None else backend
        self.fallback_fps = fallback_fps
        self.name = f"webcam-{self.index}"
        self._capture: Optional[cv2.VideoCapture] = None
        self.fps: float = fallback_fps
        self.width: int = width
        self.height: int = height
        self.frame_count: int = 0
        self.frames_read: int = 0

    def open(self) -> bool:
        """Open the camera, retrying with the generic backend if needed.

        Raises:
            VideoSourceError: When no camera can be opened at this index.
        """
        for backend in (self.backend, cv2.CAP_ANY):
            capture = cv2.VideoCapture(self.index, backend)
            if capture.isOpened():
                self._capture = capture
                break
            capture.release()
        if self._capture is None:
            raise VideoSourceError(
                f"Camera {self.index} is unavailable. Check that no other application is using it "
                "and that the browser/OS granted camera permission."
            )

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_size[0])
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_size[1])
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.requested_size[0]
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.requested_size[1]
        reported_fps = self._capture.get(cv2.CAP_PROP_FPS)
        self.fps = float(reported_fps) if reported_fps and reported_fps > 0 else self.fallback_fps
        self.frame_count = 0
        self.frames_read = 0
        logger.info("Opened webcam %s (%dx%d @ %.2f fps)", self.index, self.width, self.height, self.fps)
        return True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Grab the next camera frame."""
        if self._capture is None:
            return False, None
        success, frame = self._capture.read()
        if not success or frame is None:
            return False, None
        self.frames_read += 1
        return True, frame

    def release(self) -> None:
        """Release the camera so other applications can use it."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    @property
    def is_open(self) -> bool:
        """``True`` while the camera handle is valid."""
        return self._capture is not None and self._capture.isOpened()


class DemoSource:
    """Generates a synthetic side-view push-up video in memory.

    Used by the in-app *Demo* mode and by the test-suite: it needs no camera, no
    file and no model weights, and it produces real BGR frames with a stick
    figure that the overlay can be verified against.
    """

    def __init__(
        self,
        *,
        cycle_seconds: float = 3.0,
        fps: float = 25.0,
        size: Tuple[int, int] = (1280, 720),
        draw_figure: bool = True,
        max_frames: Optional[int] = None,
        hips_offset: float = 0.0,
        arm_asymmetry: float = 0.0,
    ) -> None:
        self.cycle_seconds = float(cycle_seconds)
        self.fps = float(fps)
        self.width, self.height = int(size[0]), int(size[1])
        self.draw_figure = draw_figure
        self.max_frames = max_frames
        self.hips_offset = float(hips_offset)
        self.arm_asymmetry = float(arm_asymmetry)
        self.name = "demo-pushup"
        self.frame_count = int(max_frames) if max_frames else 0
        self.frames_read: int = 0
        self._opened = False

    def open(self) -> bool:
        """Mark the source ready (nothing to allocate)."""
        self._opened = True
        self.frames_read = 0
        return True

    def phase_for_frame(self, frame_id: int) -> float:
        """Movement phase in ``[0, 1]`` for a frame index."""
        frames_per_cycle = max(2.0, self.cycle_seconds * self.fps)
        position = (frame_id % frames_per_cycle) / frames_per_cycle
        return 0.5 - 0.5 * math.cos(2.0 * math.pi * position)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Render the next demo frame."""
        if not self._opened:
            return False, None
        if self.max_frames is not None and self.frames_read >= self.max_frames:
            return False, None
        frame = self._render(self.frames_read)
        self.frames_read += 1
        return True, frame

    def release(self) -> None:
        """Nothing to release."""
        self._opened = False

    # ------------------------------------------------------------------ render
    def _render(self, frame_id: int) -> np.ndarray:
        phase = self.phase_for_frame(frame_id)
        points = generate_pushup_keypoints(
            phase,
            hips_offset=self.hips_offset,
            arm_asymmetry=self.arm_asymmetry,
            frame_shape=(self.height, self.width),
            person_scale=0.62,
            seed=1000 + frame_id,
            jitter=0.0,
        )

        frame = np.full((self.height, self.width, 3), 26, dtype=np.uint8)
        floor_y = int(self.height * 0.78) + int(self.height * 0.03)
        cv2.rectangle(frame, (0, floor_y), (self.width, self.height), (38, 38, 38), thickness=cv2.FILLED)
        cv2.line(frame, (0, floor_y), (self.width, floor_y), (70, 70, 70), 2, cv2.LINE_AA)

        if self.draw_figure:
            pixels = {name: (int(point[0]), int(point[1])) for name, point in points.items()}
            for index_a, index_b, _limb in skeleton_pairs():
                name_a, name_b = KEYPOINT_NAMES[index_a], KEYPOINT_NAMES[index_b]
                if name_a in pixels and name_b in pixels:
                    cv2.line(frame, pixels[name_a], pixels[name_b], (150, 150, 150), 4, cv2.LINE_AA)
            for point in pixels.values():
                cv2.circle(frame, point, 5, (200, 200, 200), -1, cv2.LINE_AA)

        cv2.putText(
            frame,
            "DEMO STREAM",
            (20, self.height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (110, 110, 110),
            1,
            cv2.LINE_AA,
        )
        return frame

    @property
    def is_open(self) -> bool:
        """``True`` once :meth:`open` has been called."""
        return self._opened


def open_source(source: FrameSource) -> FrameSource:
    """Open a source and translate failures into :class:`VideoSourceError`."""
    try:
        opened = source.open()
    except VideoSourceError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface a friendly message
        raise VideoSourceError(f"Could not open {getattr(source, 'name', 'source')}: {exc}") from exc
    if not opened:
        raise VideoSourceError(f"Could not open {getattr(source, 'name', 'source')}")
    return source


__all__ = [
    "FrameSource",
    "VideoFileSource",
    "WebcamSource",
    "DemoSource",
    "VideoSourceError",
    "open_source",
    "preferred_backend",
]

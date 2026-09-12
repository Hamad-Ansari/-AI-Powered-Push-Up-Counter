"""Processed-video export using ``cv2.VideoWriter`` with codec fallbacks."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.utils.helpers import ensure_dir, timestamp_slug
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VideoExporter:
    """Writes annotated frames to an MP4 file.

    Windows builds of OpenCV do not always expose H.264, so the exporter tries the
    configured codec first and then the fallbacks from ``video.fallback_fourccs``.

    Example:
        >>> exporter = VideoExporter(path, (1280, 720), fps=30.0)
        >>> exporter.open()
        True
        >>> exporter.write(annotated_frame)
        >>> exporter.close()
    """

    #: Reasonable default codec per container (H.264 is often missing on Windows).
    _CONTAINER_FOURCC = {".mp4": "mp4v", ".avi": "XVID", ".mkv": "mp4v", ".mov": "mp4v"}

    def __init__(
        self,
        path: str | Path,
        frame_size: Tuple[int, int],
        fps: float = 30.0,
        fourcc: str = "mp4v",
        fallback_fourccs: Optional[List[str]] = None,
        gif_width: int = 480,
    ) -> None:
        """Create the exporter.

        Args:
            path: Output file path (parent directories are created). ``.gif``
                produces an animated GIF; other supported containers use OpenCV.
            frame_size: ``(width, height)`` of the written video.
            fps: Output frame rate.
            fourcc: Preferred four-character codec.
            fallback_fourccs: Codecs to try when the preferred one is unavailable.
            gif_width: Width (px) the GIF is downscaled to, to keep files small.
        """
        self.path = Path(path)
        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self.fps = max(1.0, float(fps))
        self.fourcc = fourcc
        self.fallback_fourccs = list(fallback_fourccs or [])
        self.gif_width = int(gif_width)
        self._writer: Optional[cv2.VideoWriter] = None
        self._gif_frames: List[np.ndarray] = []
        self._is_gif = False
        self._frames_written = 0
        self._codec_used: str = ""

    def open(self) -> bool:
        """Create the underlying ``VideoWriter``.

        Returns:
            ``True`` when a usable writer was created.
        """
        ensure_dir(self.path.parent)
        suffix = self.path.suffix.lower()
        if suffix == ".gif":
            self._is_gif = True
            self._gif_frames = []
            self._codec_used = "gif"
            self._frames_written = 0
            logger.info("GIF export started: %s (%dx%d @ %.1f fps)", self.path.name, *self.frame_size, self.fps)
            return True
        if suffix not in self._CONTAINER_FOURCC:
            self.path = self.path.with_suffix(".mp4")
            suffix = ".mp4"

        candidates = [self.fourcc, *self.fallback_fourccs, self._CONTAINER_FOURCC[suffix]]
        for codec in candidates:
            fourcc_code = cv2.VideoWriter_fourcc(*codec[:4])
            writer = cv2.VideoWriter(str(self.path), fourcc_code, self.fps, self.frame_size)
            if writer.isOpened():
                self._writer = writer
                self._codec_used = codec
                self._frames_written = 0
                logger.info("Video export started: %s (%s, %dx%d @ %.1f fps)", self.path.name, codec, *self.frame_size, self.fps)
                return True
            writer.release()
            logger.warning("Codec %s unavailable for %s", codec, self.path.name)

        logger.error("No usable video codec found - export disabled")
        self._writer = None
        return False

    def write(self, frame_bgr: np.ndarray) -> bool:
        """Append one annotated frame.

        Args:
            frame_bgr: BGR frame; resized when it does not match the target size.

        Returns:
            ``True`` when the frame was written.
        """
        if frame_bgr is None:
            return False
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            logger.warning("Refusing to write malformed frame (shape=%s)", getattr(frame_bgr, "shape", None))
            return False
        height, width = frame_bgr.shape[:2]
        if (width, height) != self.frame_size:
            frame_bgr = cv2.resize(frame_bgr, self.frame_size, interpolation=cv2.INTER_AREA)

        if self._is_gif:
            scaled = frame_bgr
            if scaled.shape[1] > self.gif_width:
                scale = self.gif_width / float(scaled.shape[1])
                scaled = cv2.resize(scaled, (self.gif_width, int(scaled.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            self._gif_frames.append(cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB))
            self._frames_written += 1
            return True

        if self._writer is None:
            return False
        self._writer.write(frame_bgr)
        self._frames_written += 1
        return True

    def close(self) -> Optional[Path]:
        """Finalise the file.

        Returns:
            The output path when at least one frame was written, else ``None``.
        """
        if self._is_gif:
            result = self._finish_gif()
            self._is_gif = False
            return result

        if self._writer is None:
            return None
        self._writer.release()
        self._writer = None
        if self._frames_written == 0:
            logger.warning("No frames were written - removing empty export %s", self.path.name)
            self.path.unlink(missing_ok=True)
            return None
        logger.info("Video export finished: %s (%d frames)", self.path.name, self._frames_written)
        return self.path

    def _finish_gif(self) -> Optional[Path]:
        """Write the buffered frames out as an animated GIF (Pillow)."""
        if not self._gif_frames:
            logger.warning("No frames were written - removing empty export %s", self.path.name)
            self.path.unlink(missing_ok=True)
            return None
        try:
            from PIL import Image
        except ImportError:  # pragma: no cover - Pillow ships with Streamlit
            logger.error("Pillow is required for GIF export (pip install Pillow)")
            return None

        images = [Image.fromarray(frame) for frame in self._gif_frames]
        duration_ms = max(1, int(round(1000.0 / self.fps)))
        images[0].save(
            self.path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )
        self._gif_frames = []
        logger.info("GIF export finished: %s (%d frames)", self.path.name, self._frames_written)
        return self.path

    @property
    def frames_written(self) -> int:
        """How many frames were written so far."""
        return self._frames_written

    @property
    def codec_used(self) -> str:
        """Codec that actually opened successfully."""
        return self._codec_used

    @property
    def is_open(self) -> bool:
        """``True`` while the writer is active."""
        return self._writer is not None or self._is_gif

    def __enter__(self) -> VideoExporter:
        self.open()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def default_export_path(videos_dir: str | Path, template: str = "pushup_analysis_{timestamp}.mp4") -> Path:
    """Build the default export filename inside ``videos_dir``."""
    directory = ensure_dir(videos_dir)
    return directory / template.format(timestamp=timestamp_slug())


def save_screenshot(
    frame_bgr: np.ndarray,
    screenshots_dir: str | Path,
    *,
    prefix: str = "screenshot",
    image_format: str = "jpg",
) -> Path:
    """Save one annotated frame as a screenshot.

    Args:
        frame_bgr: The processed frame to save.
        screenshots_dir: Destination directory (created when missing).
        prefix: Filename prefix.
        image_format: ``jpg`` or ``png``.

    Returns:
        Path of the written image.

    Raises:
        ValueError: If the frame is empty.
    """
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        raise ValueError("Cannot save an empty frame")
    directory = ensure_dir(screenshots_dir)
    extension = image_format.lower().lstrip(".")
    if extension not in {"jpg", "jpeg", "png"}:
        extension = "jpg"
    path = directory / f"{prefix}_{timestamp_slug()}.{extension}"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), 95] if extension in {"jpg", "jpeg"} else []
    if not cv2.imwrite(str(path), frame_bgr, params):
        raise OSError(f"OpenCV failed to write screenshot to {path}")
    logger.info("Screenshot saved: %s", path.name)
    return path


__all__ = ["VideoExporter", "default_export_path", "save_screenshot"]

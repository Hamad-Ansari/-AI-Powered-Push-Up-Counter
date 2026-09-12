"""Generate the bundled demo clips under ``assets/demo/``.

The clips are synthetic side-view push-ups rendered with OpenCV, so the project
ships runnable samples that need no camera and no model weights. Several variants
are produced so every posture rule can be demonstrated::

    python -m scripts.make_demo_video

Produces:
    demo_pushup.mp4        good form, 3.0 s per rep
    demo_fast.mp4          faster, more real-life tempo (1.8 s per rep)
    demo_hips_high.mp4     pike fault (hips raised)
    demo_hips_low.mp4      sag fault (hips dropped)
    demo_uneven_arms.mp4   asymmetric arm movement
    demo_pushup.gif        animated-GIF preview of the good-form clip
"""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger, setup_logging  # noqa: E402
from src.video.sources import DemoSource  # noqa: E402
from src.video.video_exporter import VideoExporter  # noqa: E402

setup_logging()
logger = get_logger(__name__)

CLIPS = [
    # (filename, seconds, kwargs for DemoSource)
    ("demo_pushup.mp4", 18.0, {"cycle_seconds": 3.0}),
    ("demo_fast.mp4", 12.0, {"cycle_seconds": 1.8}),
    ("demo_hips_high.mp4", 12.0, {"cycle_seconds": 3.0, "hips_offset": 0.10}),
    ("demo_hips_low.mp4", 12.0, {"cycle_seconds": 3.0, "hips_offset": -0.10}),
    ("demo_uneven_arms.mp4", 12.0, {"cycle_seconds": 3.0, "arm_asymmetry": 25.0}),
]


def build_demo_video(
    output: Path,
    *,
    seconds: float = 18.0,
    fps: float = 25.0,
    size: tuple[int, int] = (1280, 720),
    cycle_seconds: float = 3.0,
    hips_offset: float = 0.0,
    arm_asymmetry: float = 0.0,
) -> Path:
    """Render ``seconds`` of demo push-up and write it to ``output``.

    The container is chosen by the file extension (``.mp4``/``.avi`` use OpenCV,
    ``.gif`` produces an animated GIF).
    """
    source = DemoSource(
        cycle_seconds=cycle_seconds,
        fps=fps,
        size=size,
        draw_figure=True,
        hips_offset=hips_offset,
        arm_asymmetry=arm_asymmetry,
    )
    source.open()
    exporter = VideoExporter(output, size, fps, fourcc="mp4v", fallback_fourccs=["avc1", "XVID", "MJPG"])
    if not exporter.open():
        raise RuntimeError("No usable video codec available to write the demo clip")

    total = int(seconds * fps)
    for _ in range(total):
        ok, frame = source.read()
        if not ok:
            break
        exporter.write(frame)
    exporter.close()
    source.release()
    logger.info("Wrote %d frames -> %s", exporter.frames_written, output)
    return output


def build_all(output_dir: Path) -> list[Path]:
    """Render every bundled clip plus a GIF preview."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, seconds, kwargs in CLIPS:
        written.append(build_demo_video(output_dir / filename, seconds=seconds, **kwargs))
    # GIF preview of the good-form clip (short + downscaled to stay small).
    written.append(
        build_demo_video(
            output_dir / "demo_pushup.gif",
            seconds=6.0,
            fps=15.0,
            size=(640, 360),
            cycle_seconds=3.0,
        )
    )
    return written


if __name__ == "__main__":
    target_dir = Path(__file__).resolve().parent.parent / "assets" / "demo"
    for path in build_all(target_dir):
        print(f"  {path.name}  ({path.stat().st_size / 1e6:.2f} MB)")

"""Video layer: sources, the analysis pipeline and the video exporter."""

from src.video.sources import (
    DemoSource,
    FrameSource,
    VideoFileSource,
    VideoSourceError,
    WebcamSource,
    open_source,
    preferred_backend,
)
from src.video.video_exporter import VideoExporter, default_export_path, save_screenshot
from src.video.video_processor import (
    FrameAnalyzer,
    FrameRenderer,
    FrameResult,
    ProcessingReport,
    capture_screenshot,
    process_source,
    process_video_file,
)

__all__ = [
    "DemoSource",
    "FrameSource",
    "VideoFileSource",
    "VideoSourceError",
    "WebcamSource",
    "open_source",
    "preferred_backend",
    "VideoExporter",
    "default_export_path",
    "save_screenshot",
    "FrameAnalyzer",
    "FrameRenderer",
    "FrameResult",
    "ProcessingReport",
    "capture_screenshot",
    "process_source",
    "process_video_file",
]

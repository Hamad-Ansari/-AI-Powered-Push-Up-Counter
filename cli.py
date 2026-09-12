"""Command-line runner for headless batch analysis.

Processes a video file (or the bundled demo) without the Streamlit UI, writes the
annotated video + reports, and prints the workout summary. Useful for CI, quick
checks and server-side batch jobs::

    python cli.py --source assets/demo/demo_pushup.mp4
    python cli.py --demo --engine demo --max-frames 250
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config.settings import load_settings
from src.analytics.reporter import generate_reports
from src.utils.logger import get_logger, setup_logging
from src.video.sources import DemoSource, VideoFileSource
from src.video.video_processor import process_source

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Define the CLI surface."""
    parser = argparse.ArgumentParser(description="AI-Powered Push-Up Counter — batch analysis")
    parser.add_argument("--source", type=str, default=None, help="Path to a video file (mp4/avi/mov/...)")
    parser.add_argument("--demo", action="store_true", help="Use the built-in synthetic demo stream")
    parser.add_argument("--engine", type=str, default=None, choices=["rf-detr", "demo"], help="Override the pose engine")
    parser.add_argument("--cycle", type=float, default=3.0, help="Demo seconds per repetition")
    parser.add_argument("--hips-offset", type=float, default=0.0, help="Demo hip deflection (+ pike, - sag)")
    parser.add_argument("--arm-asymmetry", type=float, default=0.0, help="Demo left/right elbow mismatch (deg)")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    parser.add_argument("--format", type=str, default=None, choices=["mp4", "avi", "gif"], help="Processed-video container")
    parser.add_argument("--no-export", action="store_true", help="Do not write the processed video")
    parser.add_argument("--config", type=str, default=None, help="Path to an alternative config.yaml")
    parser.add_argument("--log-level", type=str, default=None, help="Override the log level")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    settings = load_settings(args.config)
    setup_logging(args.log_level or settings.runtime.log_level, settings.log_path)
    settings.ensure_directories()

    if args.engine:
        settings.model.engine = args.engine
    if args.format:
        settings.video.export_format = args.format

    analyzer = None
    if args.demo or not args.source:
        if not args.demo and not args.source:
            logger.info("No --source given; falling back to the demo stream.")
        source = DemoSource(
            cycle_seconds=args.cycle,
            fps=settings.video.output_fps,
            size=(1280, 720),
            hips_offset=args.hips_offset,
            arm_asymmetry=args.arm_asymmetry,
        )
        source_name = "demo-stream"
        # Match the analysed pose to the rendered demo (incl. form faults).
        from src.pose.demo_adapter import SyntheticPushUpAdapter
        from src.pose.pose_detector import PoseDetector
        from src.video.video_processor import FrameAnalyzer

        adapter = SyntheticPushUpAdapter(
            cycle_seconds=args.cycle,
            fps=settings.video.output_fps,
            hips_offset=args.hips_offset,
            arm_asymmetry=args.arm_asymmetry,
        )
        analyzer = FrameAnalyzer(settings, pose_detector=PoseDetector(settings, adapter=adapter))
    else:
        path = Path(args.source)
        if not path.exists():
            logger.error("Source video not found: %s", path)
            return 2
        source = VideoFileSource(path, fallback_fps=settings.video.output_fps)
        source_name = path.name

    report, analyzer = process_source(
        source,
        settings,
        analyzer=analyzer,
        export_video=not args.no_export,
        max_frames=args.max_frames,
    )

    if report.error:
        logger.error("Processing error: %s", report.error)

    summary = analyzer.session.summary()
    print("\n" + "=" * 46)
    print("  WORKOUT SUMMARY")
    print("=" * 46)
    for label, value in summary.headline_rows():
        print(f"  {label:<22} {value}")
    print("=" * 46)
    if report.output_video:
        print(f"  Processed video: {report.output_video}")

    paths = generate_reports(
        analyzer.session,
        settings.reports_dir,
        source_name=source_name,
        pose_backend=analyzer.pose_detector.adapter.describe(),
    )
    for path in paths.existing():
        print(f"  Report:          {path}")
    return 0 if not report.error else 1


if __name__ == "__main__":
    sys.exit(main())

"""Analytics layer: session tracking, Plotly charts and report generation."""

from src.analytics.charts import (
    all_charts,
    arm_symmetry_chart,
    elbow_angle_chart,
    elbow_comparison_chart,
    form_score_chart,
    phase_timeline_chart,
    reps_over_time_chart,
)
from src.analytics.reporter import ReportPaths, generate_reports, render_markdown
from src.analytics.session_tracker import (
    FrameRecord,
    RepRecord,
    SessionSummary,
    SessionTracker,
    export_session_json,
)

__all__ = [
    "all_charts",
    "arm_symmetry_chart",
    "elbow_angle_chart",
    "elbow_comparison_chart",
    "form_score_chart",
    "phase_timeline_chart",
    "reps_over_time_chart",
    "ReportPaths",
    "generate_reports",
    "render_markdown",
    "FrameRecord",
    "RepRecord",
    "SessionSummary",
    "SessionTracker",
    "export_session_json",
]

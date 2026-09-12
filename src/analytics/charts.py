"""Interactive Plotly charts for the workout analytics dashboard."""

from __future__ import annotations

from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go

from src.analytics.session_tracker import SessionTracker
from config.settings import AnalyticsConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

TEMPLATE = "plotly_dark"
COLORS = {
    "left": "#22d3ee",
    "right": "#f59e0b",
    "form": "#22c55e",
    "reps": "#3b82f6",
    "up": "#22c55e",
    "down": "#3b82f6",
    "moving": "#facc15",
    "invalid": "#ef4444",
    "grid": "#1f2a44",
}


def _base_figure(title: str, height: int) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        template=TEMPLATE,
        title=dict(text=title, font=dict(size=15)),
        height=height,
        margin=dict(l=45, r=20, t=48, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11,18,32,0.65)",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
        hovermode="x unified",
    )
    figure.update_xaxes(gridcolor=COLORS["grid"], zeroline=False)
    figure.update_yaxes(gridcolor=COLORS["grid"], zeroline=False)
    return figure


def elbow_angle_chart(tracker: SessionTracker, config: Optional[AnalyticsConfig] = None) -> go.Figure:
    """Left / right / mean elbow angle over time."""
    config = config or AnalyticsConfig()
    frame = tracker.to_dataframe()
    figure = _base_figure("Elbow Angle Over Time", config.chart_height)
    if frame.empty:
        return figure

    x = frame["timestamp"]
    figure.add_trace(go.Scatter(x=x, y=frame["left_elbow"], name="Left elbow", mode="lines", line=dict(color=COLORS["left"], width=2)))
    figure.add_trace(go.Scatter(x=x, y=frame["right_elbow"], name="Right elbow", mode="lines", line=dict(color=COLORS["right"], width=2)))
    figure.add_trace(
        go.Scatter(x=x, y=frame["elbow_mean"], name="Mean", mode="lines", line=dict(color="#e2e8f0", width=1, dash="dot"))
    )
    figure.update_yaxes(title="degrees", range=[40, 190])
    return figure


def form_score_chart(tracker: SessionTracker, config: Optional[AnalyticsConfig] = None) -> go.Figure:
    """Form score over time with the classification bands."""
    config = config or AnalyticsConfig()
    frame = tracker.to_dataframe()
    figure = _base_figure("Form Score Over Time", config.chart_height)
    if frame.empty:
        return figure

    figure.add_trace(
        go.Scatter(
            x=frame["timestamp"],
            y=frame["form_score"],
            name="Form score",
            mode="lines+markers",
            line=dict(color=COLORS["form"], width=2),
            marker=dict(size=4),
        )
    )
    for value, label, color in ((90, "Excellent", COLORS["up"]), (75, "Good", COLORS["reps"]), (60, "Needs improvement", COLORS["moving"])):
        figure.add_hline(y=value, line=dict(color=color, width=1, dash="dash"), annotation_text=label, annotation_position="right")
    figure.update_yaxes(title="score", range=[0, 100])
    return figure


def reps_over_time_chart(tracker: SessionTracker, config: Optional[AnalyticsConfig] = None) -> go.Figure:
    """Cumulative valid and invalid repetitions."""
    config = config or AnalyticsConfig()
    frame = tracker.to_dataframe()
    figure = _base_figure("Repetitions Over Time", config.chart_height)
    if frame.empty:
        return figure

    figure.add_trace(
        go.Scatter(
            x=frame["timestamp"],
            y=frame["valid_reps"],
            name="Valid reps",
            mode="lines",
            line=dict(color=COLORS["reps"], width=2, shape="hv"),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.15)",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["timestamp"],
            y=frame["invalid_reps"],
            name="Invalid reps",
            mode="lines",
            line=dict(color=COLORS["invalid"], width=2, shape="hv", dash="dot"),
        )
    )
    figure.update_yaxes(title="reps", rangemode="tozero")
    return figure


def phase_timeline_chart(tracker: SessionTracker, config: Optional[AnalyticsConfig] = None) -> go.Figure:
    """Coloured UP / MOVING / DOWN timeline (Gantt-style bars)."""
    config = config or AnalyticsConfig()
    frame = tracker.to_dataframe()
    figure = _base_figure("Push-Up Phase Timeline", max(180, int(config.chart_height * 0.6)))
    if frame.empty:
        return figure

    spans = _state_spans(frame)
    for state, color in (("UP", COLORS["up"]), ("MOVING", COLORS["moving"]), ("DOWN", COLORS["down"])):
        entries = [(start, end) for label, start, end in spans if label == state]
        if not entries:
            continue
        figure.add_trace(
            go.Bar(
                x=[end - start for start, end in entries],
                base=[start for start, _ in entries],
                y=[state] * len(entries),
                orientation="h",
                name=state,
                marker=dict(color=color),
                hovertemplate="%{base:.2f}s → %{x:.2f}s<extra>" + state + "</extra>",
            )
        )
    figure.update_layout(barmode="overlay", showlegend=False, height=max(180, int(config.chart_height * 0.6)))
    figure.update_xaxes(title="seconds")
    return figure


def elbow_comparison_chart(tracker: SessionTracker, config: Optional[AnalyticsConfig] = None) -> go.Figure:
    """Left vs right elbow: scatter plus the per-frame difference."""
    config = config or AnalyticsConfig()
    frame = tracker.to_dataframe()
    figure = _base_figure("Left vs Right Elbow", config.chart_height)
    if frame.empty:
        return figure

    valid = frame.dropna(subset=["left_elbow", "right_elbow"])
    if valid.empty:
        return figure
    figure.add_trace(
        go.Scatter(
            x=valid["left_elbow"],
            y=valid["right_elbow"],
            mode="markers",
            name="L vs R",
            marker=dict(size=5, color=COLORS["left"], opacity=0.65),
        )
    )
    low = float(min(valid["left_elbow"].min(), valid["right_elbow"].min())) - 5
    high = float(max(valid["left_elbow"].max(), valid["right_elbow"].max())) + 5
    figure.add_trace(
        go.Scatter(x=[low, high], y=[low, high], mode="lines", name="Perfect symmetry", line=dict(color="#94a3b8", dash="dash", width=1))
    )
    figure.update_xaxes(title="left elbow (°)", range=[low, high])
    figure.update_yaxes(title="right elbow (°)", range=[low, high])
    return figure


def arm_symmetry_chart(tracker: SessionTracker, config: Optional[AnalyticsConfig] = None) -> go.Figure:
    """Left/right elbow difference over time (symmetry tracking)."""
    config = config or AnalyticsConfig()
    frame = tracker.to_dataframe()
    figure = _base_figure("Arm Symmetry (|L - R|)", config.chart_height)
    if frame.empty:
        return figure
    figure.add_trace(
        go.Scatter(
            x=frame["timestamp"],
            y=frame["elbow_diff"],
            name="|L - R|",
            mode="lines",
            line=dict(color=COLORS["right"], width=2),
            fill="tozeroy",
            fillcolor="rgba(245,158,11,0.12)",
        )
    )
    figure.update_yaxes(title="degrees", rangemode="tozero")
    return figure


def all_charts(tracker: SessionTracker, config: Optional[AnalyticsConfig] = None) -> dict:
    """Every dashboard chart keyed by name."""
    return {
        "elbow_angle": elbow_angle_chart(tracker, config),
        "form_score": form_score_chart(tracker, config),
        "reps": reps_over_time_chart(tracker, config),
        "timeline": phase_timeline_chart(tracker, config),
        "comparison": elbow_comparison_chart(tracker, config),
        "symmetry": arm_symmetry_chart(tracker, config),
    }


def _state_spans(frame: pd.DataFrame) -> List[tuple]:
    """Collapse the per-frame state column into ``(state, start, end)`` spans."""
    spans: List[tuple] = []
    if frame.empty:
        return spans
    rows = frame[["timestamp", "state"]].to_dict("records")
    current_state = rows[0]["state"]
    start = float(rows[0]["timestamp"])
    for previous, current in zip(rows, rows[1:], strict=False):
        if current["state"] != previous["state"]:
            spans.append((previous["state"], start, float(current["timestamp"])))
            current_state = current["state"]
            start = float(current["timestamp"])
    spans.append((current_state, start, float(rows[-1]["timestamp"])))
    return spans


__all__ = [
    "elbow_angle_chart",
    "form_score_chart",
    "reps_over_time_chart",
    "phase_timeline_chart",
    "elbow_comparison_chart",
    "arm_symmetry_chart",
    "all_charts",
]

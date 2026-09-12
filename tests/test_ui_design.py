"""Tests for the web UI design system (src.ui): tokens + pure HTML/SVG builders."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.form_scorer import FormGrade  # noqa: E402
from src.analysis.movement_analyzer import ExerciseState  # noqa: E402
from src.ui import components as ui  # noqa: E402
from src.ui import theme  # noqa: E402


def test_css_is_rendered_from_palette():
    sheet = theme.css()
    assert theme.PALETTE["accent"] in sheet
    assert theme.PALETTE["bg"] in sheet
    assert ".pu-kpi" in sheet and ".pu-card" in sheet and ".pu-hero" in sheet


def test_grade_color_mapping():
    assert theme.grade_color(FormGrade.EXCELLENT) == theme.PALETTE["good"]
    assert theme.grade_color(FormGrade.POOR) == theme.PALETTE["bad"]


def test_state_meta_fallback():
    label, color, _ = theme.state_meta(ExerciseState.UNKNOWN)
    assert label == "WAITING" and color == theme.PALETTE["muted"]


def test_kpi_card_contains_value_and_accent():
    html = ui.kpi("🔁", "Total Reps", "12", "10 valid", "#A3E635")
    assert "12" in html and "Total Reps" in html and "--acc:#A3E635" in html


def test_kpi_row_wraps_cards():
    html = ui.kpi_row([ui.kpi("a", "L", "1"), ui.kpi("b", "M", "2")])
    assert html.startswith("<div class='pu-kpis'>") and html.count("pu-kpi\"") >= 2


def test_score_ring_geometry():
    size, stroke = 132, 10
    circ = 2 * math.pi * (size - stroke) / 2
    half = ui.score_ring(50.0, "#ffffff", size=size)
    assert f"{circ / 2:.1f}" in half
    full = ui.score_ring(100.0, "#ffffff", size=size)
    assert f"{circ:.1f}" in full
    # NaN and out-of-range scores are clamped, not rendered raw.
    assert 'stroke-dasharray="0.0' in ui.score_ring(float("nan"), "#ffffff", size=size)
    assert f"{circ:.1f}" in ui.score_ring(150.0, "#ffffff", size=size)


def test_meter_clamps_fraction():
    assert "width:100.0%" in ui.meter("X", "9°", 3.0, "#fff")
    assert "width:0.0%" in ui.meter("X", "—", float("nan"), "#fff")


def test_html_escaping():
    html = ui.feedback_line("<script>alert('x')</script>")
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_hero_shows_status():
    html = ui.hero("LIVE", theme.PALETTE["good"])
    assert "LIVE" in html and theme.PALETTE["good"] in html


def test_sidebar_brand_has_svg_logo():
    html = ui.sidebar_brand()
    assert "<svg" in html and theme.BRAND_NAME in html

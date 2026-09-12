"""Design system for the Streamlit dashboard ("PULSE" dark-athletic theme).

Every colour, radius and shadow used by the web UI lives here — components in
:mod:`src.ui.components` and the global stylesheet both read from
:data:`PALETTE`, so re-skinning the app is a one-file change.

The overlay (OpenCV) colours stay in ``config/settings.yaml``; this module is
the single source of truth for the *web* surface.
"""

from __future__ import annotations

import streamlit as st

from src.analysis.form_scorer import FormGrade
from src.analysis.movement_analyzer import ExerciseState

# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #
PALETTE = {
    # Surfaces
    "bg": "#070B14",
    "surface": "#0D1526",
    "surface_2": "#111C33",
    "border": "#1E2A44",
    # Type
    "text": "#E6EDF7",
    "muted": "#8B98AC",
    # Brand / data accents
    "accent": "#A3E635",   # electric lime  — energy, primary actions
    "accent_2": "#22D3EE", # cyan           — telemetry / data
    # Semantic states
    "good": "#34D399",
    "warn": "#FBBF24",
    "bad": "#F87171",
    "info": "#60A5FA",
}

RADIUS = "16px"
RADIUS_SM = "10px"
SHADOW = "0 10px 30px rgba(2, 6, 18, 0.55)"

BRAND_NAME = "PULSE·REP"
BRAND_TAG = "AI Push-Up Coach"
BRAND_SUB = "Real-time pose detection & exercise-form analysis"

GRADE_COLORS = {
    FormGrade.EXCELLENT: PALETTE["good"],
    FormGrade.GOOD: PALETTE["info"],
    FormGrade.NEEDS_IMPROVEMENT: PALETTE["warn"],
    FormGrade.POOR: PALETTE["bad"],
}

STATE_META = {
    ExerciseState.UP: ("UP", PALETTE["good"], "▲"),
    ExerciseState.DOWN: ("DOWN", PALETTE["accent_2"], "▼"),
    ExerciseState.MOVING: ("MOVING", PALETTE["warn"], "●"),
    ExerciseState.UNKNOWN: ("WAITING", PALETTE["muted"], "○"),
}


def grade_color(grade: FormGrade) -> str:
    """Hex accent for a form grade."""
    return GRADE_COLORS.get(grade, PALETTE["muted"])


def state_meta(state: ExerciseState) -> tuple[str, str, str]:
    """(label, hex colour, glyph) for a movement state."""
    return STATE_META.get(state, STATE_META[ExerciseState.UNKNOWN])


# --------------------------------------------------------------------------- #
# Global stylesheet
# --------------------------------------------------------------------------- #
def css() -> str:
    """Return the full dashboard stylesheet, rendered from :data:`PALETTE`."""
    p = PALETTE
    return f"""
<style>
/* ============ base / background ============ */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
    background: {p['bg']};
}}
[data-testid="stAppViewContainer"] > section.main > div.block-container {{
    padding-top: 1.1rem;
    padding-bottom: 4rem;
    max-width: 1280px;
}}
body, p, span, label, .stMarkdown {{ color: {p['text']}; }}

/* hide the stock Streamlit chrome for a product feel */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}

/* ============ sidebar ============ */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {p['surface']} 0%, {p['bg']} 100%);
    border-right: 1px solid {p['border']};
}}
[data-testid="stSidebar"] .stMarkdown p {{ color: {p['muted']}; }}

/* ============ cards ============ */
.pu-card {{
    background: linear-gradient(165deg, {p['surface_2']} 0%, {p['surface']} 100%);
    border: 1px solid {p['border']};
    border-radius: {RADIUS};
    box-shadow: {SHADOW};
    padding: 16px 18px;
}}

/* ============ hero ============ */
.pu-hero {{ display: flex; align-items: center; justify-content: space-between;
    gap: 16px; margin: 2px 0 14px; }}
.pu-hero h1 {{
    margin: 0; font-size: 1.9rem; font-weight: 800; letter-spacing: -0.6px;
    background: linear-gradient(92deg, {p['text']} 0%, {p['accent']} 55%, {p['accent_2']} 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}}
.pu-hero .tag {{ color: {p['muted']}; font-size: 0.86rem; margin-top: 2px; }}

/* ============ pills / badges ============ */
.pu-pill {{
    display: inline-flex; align-items: center; gap: 7px;
    padding: 5px 12px; border-radius: 999px; font-size: 0.78rem; font-weight: 700;
    letter-spacing: 0.6px; border: 1px solid;
    background: color-mix(in srgb, currentColor 12%, transparent);
}}
.pu-pill .dot {{ width: 8px; height: 8px; border-radius: 50%; background: currentColor;
    box-shadow: 0 0 10px currentColor; }}

/* ============ KPI cards ============ */
.pu-kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 4px 0 14px; }}
.pu-kpi {{
    background: linear-gradient(165deg, {p['surface_2']} 0%, {p['surface']} 100%);
    border: 1px solid {p['border']}; border-radius: {RADIUS};
    padding: 12px 14px; display: flex; gap: 12px; align-items: center;
}}
.pu-kpi .ico {{
    width: 40px; height: 40px; border-radius: 12px; flex: 0 0 auto;
    display: flex; align-items: center; justify-content: center; font-size: 1.15rem;
    background: color-mix(in srgb, var(--acc) 16%, transparent);
    border: 1px solid color-mix(in srgb, var(--acc) 35%, transparent);
}}
.pu-kpi .lab {{ color: {p['muted']}; font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.8px; text-transform: uppercase; }}
.pu-kpi .val {{ font-size: 1.45rem; font-weight: 800; line-height: 1.15;
    font-variant-numeric: tabular-nums; }}
.pu-kpi .sub {{ color: {p['muted']}; font-size: 0.75rem; }}

/* ============ section headers ============ */
.pu-section {{ display: flex; align-items: baseline; gap: 10px; margin: 18px 0 10px; }}
.pu-section .bar {{ width: 4px; height: 18px; border-radius: 2px; align-self: center;
    background: linear-gradient(180deg, {p['accent']}, {p['accent_2']}); }}
.pu-section .ttl {{ font-size: 1.02rem; font-weight: 700; }}
.pu-section .hint {{ color: {p['muted']}; font-size: 0.78rem; }}

/* ============ meters ============ */
.pu-meter {{ margin: 8px 0; }}
.pu-meter .top {{ display: flex; justify-content: space-between; font-size: 0.78rem;
    color: {p['muted']}; margin-bottom: 4px; }}
.pu-meter .top b {{ color: {p['text']}; font-variant-numeric: tabular-nums; }}
.pu-meter .track {{ height: 7px; border-radius: 999px; background: {p['surface_2']};
    border: 1px solid {p['border']}; overflow: hidden; }}
.pu-meter .fill {{ height: 100%; border-radius: 999px; transition: width 120ms linear; }}

/* ============ feedback ============ */
.pu-feedback {{
    border-radius: {RADIUS_SM}; padding: 10px 14px; margin: 6px 0;
    border: 1px solid {p['border']}; border-left: 4px solid var(--acc, {p['accent_2']});
    background: {p['surface']}; font-size: 0.88rem;
}}
.pu-feedback.head {{ font-size: 0.95rem; }}

/* ============ buttons (transport bar) ============ */
div.stButton > button, [data-testid^="stBaseButton"] {{
    border-radius: 12px; border: 1px solid {p['border']};
    background: {p['surface_2']}; color: {p['text']}; font-weight: 600;
}}
div.stButton > button:hover, [data-testid^="stBaseButton"]:hover {{
    border-color: {p['accent']}; color: {p['accent']};
}}
div.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {{
    background: linear-gradient(94deg, {p['accent']} 0%, #65a30d 100%);
    border: none; color: #0b1220; font-weight: 800;
}}
div.stButton > button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover {{
    color: #0b1220; filter: brightness(1.08);
}}

/* ============ tabs / expanders ============ */
[data-testid="stTabs"] [data-testid="stTabsHeader"] {{ gap: 6px; }}
[data-testid="stTab"] {{ border-radius: 10px 10px 0 0; font-weight: 600; }}
[data-testid="stExpander"] {{
    background: {p['surface']}; border: 1px solid {p['border']}; border-radius: {RADIUS_SM};
}}

/* ============ footer ============ */
.pu-foot {{ margin-top: 26px; padding-top: 12px; border-top: 1px solid {p['border']};
    color: {p['muted']}; font-size: 0.75rem; display: flex; gap: 14px; }}

/* tidy scrollbars (WebKit) */
::-webkit-scrollbar {{ width: 9px; height: 9px; }}
::-webkit-scrollbar-thumb {{ background: {p['border']}; border-radius: 999px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
</style>
"""


def inject() -> None:
    """Push the global stylesheet into the page."""
    st.markdown(css(), unsafe_allow_html=True)


__all__ = [
    "PALETTE",
    "RADIUS",
    "RADIUS_SM",
    "SHADOW",
    "BRAND_NAME",
    "BRAND_TAG",
    "BRAND_SUB",
    "css",
    "inject",
    "grade_color",
    "state_meta",
]

"""Reusable UI components for the Streamlit dashboard.

Every function returns an HTML/SVG string; the app renders them with
``st.markdown(..., unsafe_allow_html=True)``.  Keeping them as pure string
builders makes them trivially unit-testable and preview-safe (no external
assets — everything is inline CSS/SVG).
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

from src.ui import theme

_P = theme.PALETTE


def _esc(text: object) -> str:
    """Minimal HTML escaping for interpolated values."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# --------------------------------------------------------------------------- #
# Brand + hero
# --------------------------------------------------------------------------- #
def sidebar_brand() -> str:
    """Logo block shown at the top of the sidebar."""
    acc, acc2 = _P["accent"], _P["accent_2"]
    return f"""
<div style="display:flex;align-items:center;gap:12px;margin:4px 0 10px;">
  <svg width="44" height="44" viewBox="0 0 44 44" aria-hidden="true">
    <defs>
      <linearGradient id="lg" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stop-color="{acc}"/><stop offset="1" stop-color="{acc2}"/>
      </linearGradient>
    </defs>
    <rect x="1.5" y="1.5" width="41" height="41" rx="12"
          fill="none" stroke="url(#lg)" stroke-width="2"/>
    <polyline points="8,24 15,24 19,14 25,31 29,20 36,20"
          fill="none" stroke="url(#lg)" stroke-width="2.6"
          stroke-linecap="round" stroke-linejoin="round"/>
  </svg>
  <div>
    <div style="font-weight:800;font-size:1.12rem;letter-spacing:0.4px;">{theme.BRAND_NAME}</div>
    <div style="color:{_P['muted']};font-size:0.74rem;letter-spacing:0.5px;">{_esc(theme.BRAND_TAG)}</div>
  </div>
</div>
"""


def hero(status: str, status_color: str) -> str:
    """Page header: wordmark left, live-status pill right."""
    return f"""
<div class="pu-hero">
  <div>
    <h1>{theme.BRAND_NAME} <span style="font-weight:600;font-size:1rem;color:{_P['muted']};">· {theme.BRAND_TAG}</span></h1>
    <div class="tag">{_esc(theme.BRAND_SUB)}</div>
  </div>
  <span class="pu-pill" style="color:{status_color};border-color:{status_color};">
    <span class="dot"></span>{_esc(status)}
  </span>
</div>
"""


# --------------------------------------------------------------------------- #
# KPI row
# --------------------------------------------------------------------------- #
def kpi(icon: str, label: str, value: str, sub: str = "", accent: str = _P["accent"]) -> str:
    """One KPI card."""
    return f"""
<div class="pu-kpi" style="--acc:{accent};">
  <div class="ico">{icon}</div>
  <div>
    <div class="lab">{_esc(label)}</div>
    <div class="val">{_esc(value)}</div>
    <div class="sub">{_esc(sub)}</div>
  </div>
</div>
"""


def kpi_row(cards: Iterable[str]) -> str:
    """Wrap KPI cards in a responsive grid."""
    return f"<div class='pu-kpis'>{''.join(cards)}</div>"


# --------------------------------------------------------------------------- #
# Data visual bits
# --------------------------------------------------------------------------- #
def score_ring(score: float, color: str, size: int = 132, label: str = "FORM") -> str:
    """SVG donut gauge for the 0-100 form score."""
    score = max(0.0, min(100.0, float(score) if math.isfinite(score) else 0.0))
    stroke = 10
    radius = (size - stroke) / 2
    circ = 2 * math.pi * radius
    dash = circ * score / 100.0
    center = size / 2
    return f"""
<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" style="display:block;margin:0 auto;">
  <circle cx="{center}" cy="{center}" r="{radius}" fill="none"
          stroke="{_P['surface_2']}" stroke-width="{stroke}"/>
  <circle cx="{center}" cy="{center}" r="{radius}" fill="none"
          stroke="{color}" stroke-width="{stroke}" stroke-linecap="round"
          stroke-dasharray="{dash:.1f} {circ - dash:.1f}"
          transform="rotate(-90 {center} {center})"/>
  <text x="{center}" y="{center - 2}" text-anchor="middle"
        style="font-size:{size * 0.24}px;font-weight:800;fill:{_P['text']};font-variant-numeric:tabular-nums;">
    {score:.0f}</text>
  <text x="{center}" y="{center + size * 0.16}" text-anchor="middle"
        style="font-size:{size * 0.085}px;letter-spacing:2px;fill:{_P['muted']};">{label}</text>
</svg>
"""


def meter(label: str, value_text: str, fraction: float, color: str) -> str:
    """Horizontal bar meter (angles, symmetry, ...)."""
    fraction = max(0.0, min(1.0, fraction if math.isfinite(fraction) else 0.0))
    return f"""
<div class="pu-meter">
  <div class="top"><span>{_esc(label)}</span><b>{_esc(value_text)}</b></div>
  <div class="track"><div class="fill" style="width:{fraction * 100:.1f}%;background:{color};"></div></div>
</div>
"""


def section(icon: str, title: str, hint: str = "") -> str:
    """Section header with the gradient accent bar."""
    hint_html = f"<span class='hint'>{_esc(hint)}</span>" if hint else ""
    return f"""
<div class="pu-section"><span class="bar"></span>
  <span class="ttl">{icon} {_esc(title)}</span>{hint_html}</div>
"""


# --------------------------------------------------------------------------- #
# Feedback + pills
# --------------------------------------------------------------------------- #
def feedback_head(emoji: str, headline: str, score: float, accent: str) -> str:
    return f"""
<div class="pu-feedback head" style="--acc:{accent};">
  <b>{emoji} {_esc(headline)}</b> &nbsp;·&nbsp; form score <b>{score:.0f}/100</b>
</div>
"""


def feedback_line(text: str, accent: str = _P["accent_2"]) -> str:
    return f"<div class='pu-feedback' style='--acc:{accent};'>{_esc(text)}</div>"


def pill(text: str, color: str) -> str:
    return f"<span class='pu-pill' style='color:{color};border-color:{color};'>{_esc(text)}</span>"


def footer() -> str:
    return f"""
<div class="pu-foot">
  <span>{theme.BRAND_NAME}</span><span>RF-DETR pose · angle FSM · form scoring</span>
  <span style="margin-left:auto;">outputs/ · config.yaml</span>
</div>
"""


def video_frame_placeholder() -> str:
    """Shown in the video slot before the first frame arrives."""
    return f"""
<div class="pu-card" style="min-height:340px;display:flex;flex-direction:column;
     align-items:center;justify-content:center;gap:8px;color:{_P['muted']};">
  <div style="font-size:2.2rem;">🎬</div>
  <div style="font-weight:700;color:{_P['text']};">No live frame yet</div>
  <div style="font-size:0.82rem;">Press ▶️ Start to begin the session</div>
</div>
"""


def clamp01(value: Optional[float]) -> float:
    """Utility for tests / callers."""
    if value is None or not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


__all__ = [
    "sidebar_brand",
    "hero",
    "kpi",
    "kpi_row",
    "score_ring",
    "meter",
    "section",
    "feedback_head",
    "feedback_line",
    "pill",
    "footer",
    "video_frame_placeholder",
    "clamp01",
]

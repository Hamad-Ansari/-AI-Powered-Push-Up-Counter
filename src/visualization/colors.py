"""Central colour palette.

No BGR literal appears anywhere else in the codebase: every module asks the
:class:`~src.config.settings.VisualizationConfig` (backed by ``config.yaml``) for
a named colour. Semantic names only - ``good``, ``warning``, ``bad``,
``tracking``, ``accent`` - so a re-theme is a one-file change.
"""

from __future__ import annotations

from typing import Tuple

from src.analysis.form_scorer import FormGrade
from src.analysis.posture_analyzer import Severity
from config.settings import VisualizationConfig

Color = Tuple[int, int, int]

#: Fallback palette (BGR) used when a key is missing from the configuration.
DEFAULT_PALETTE: dict[str, Color] = {
    "good": (94, 197, 34),
    "warning": (21, 204, 250),
    "bad": (68, 68, 239),
    "tracking": (246, 130, 59),
    "accent": (238, 211, 34),
    "neutral": (184, 163, 148),
    "panel": (32, 18, 11),
    "panel_border": (68, 42, 31),
    "text": (240, 232, 226),
    "text_dim": (184, 163, 148),
    "skeleton": (238, 211, 34),
    "skeleton_invalid": (139, 116, 100),
    "skeleton_highlight": (11, 158, 245),
}


class Palette:
    """Read-only view over the configured colours with safe fallbacks."""

    def __init__(self, config: VisualizationConfig) -> None:
        self.config = config

    def get(self, name: str) -> Color:
        """Return the BGR colour registered under ``name``."""
        colors = self.config.colors
        if name in colors:
            return tuple(int(v) for v in colors[name])  # type: ignore[return-value]
        return DEFAULT_PALETTE.get(name, (255, 255, 255))

    # ------------------------------------------------------------- semantic API
    @property
    def good(self) -> Color:
        """Green - good posture / valid movement."""
        return self.get("good")

    @property
    def warning(self) -> Color:
        """Yellow - moving / warning."""
        return self.get("warning")

    @property
    def bad(self) -> Color:
        """Red - bad form / invalid rep."""
        return self.get("bad")

    @property
    def tracking(self) -> Color:
        """Blue - pose detection / tracking."""
        return self.get("tracking")

    @property
    def accent(self) -> Color:
        """Cyan - brand accent."""
        return self.get("accent")

    @property
    def neutral(self) -> Color:
        """Grey - idle / unknown."""
        return self.get("neutral")

    @property
    def panel(self) -> Color:
        """HUD panel background."""
        return self.get("panel")

    @property
    def panel_border(self) -> Color:
        """HUD panel border."""
        return self.get("panel_border")

    @property
    def text(self) -> Color:
        """Primary text colour."""
        return self.get("text")

    @property
    def text_dim(self) -> Color:
        """Secondary text colour."""
        return self.get("text_dim")

    # --------------------------------------------------------------- mappings
    def severity(self, severity: Severity) -> Color:
        """Colour for a posture severity."""
        return self.get(severity.color_key)

    def grade(self, grade: FormGrade) -> Color:
        """Colour for a form grade."""
        return self.get(grade.color_key)

    def state(self, status_key: str) -> Color:
        """Colour for an exercise state (``ExerciseState.status_key``)."""
        return self.get(status_key)

    def score(self, value: float) -> Color:
        """Colour for a 0-100 score: green -> yellow -> red."""
        if value >= 90.0:
            return self.good
        if value >= 75.0:
            return self.tracking
        if value >= 60.0:
            return self.warning
        return self.bad


def with_alpha(base: Color, alpha: float) -> Color:
    """Blend ``base`` towards black by ``alpha`` (used for translucent panels)."""
    factor = max(0.0, min(1.0, alpha))
    return (int(base[0] * factor), int(base[1] * factor), int(base[2] * factor))


def mix(a: Color, b: Color, ratio: float) -> Color:
    """Linear blend between two colours (``ratio`` = weight of ``b``)."""
    ratio = max(0.0, min(1.0, ratio))
    return (
        int(a[0] * (1 - ratio) + b[0] * ratio),
        int(a[1] * (1 - ratio) + b[1] * ratio),
        int(a[2] * (1 - ratio) + b[2] * ratio),
    )


__all__ = ["Palette", "DEFAULT_PALETTE", "Color", "with_alpha", "mix"]

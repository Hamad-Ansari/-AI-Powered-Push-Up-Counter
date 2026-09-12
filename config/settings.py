"""Typed configuration layer for the AI-Powered Push-Up Counter.

The configuration system has three layers, applied in increasing precedence:

1. ``dataclass`` field defaults (this module) - always valid, always complete.
2. ``config/config.yaml`` - the human-editable master file.
3. Environment variables / ``.env`` - deployment overrides
   (e.g. ``PUSHUP_CONFIDENCE_THRESHOLD=0.65``).

Nothing else in the codebase reads YAML or environment variables directly: every
module receives a :class:`Settings` instance (or one of its sections).
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

try:  # python-dotenv is optional: configuration also works without it.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - exercised only without the dependency
    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        """No-op fallback used when ``python-dotenv`` is not installed."""
        return False


PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config" / "config.yaml"
ENV_PREFIX: str = "PUSHUP_"

# Environment override map: "ENV_SUFFIX" -> "dotted.path.in.Settings".
ENV_OVERRIDES: Dict[str, str] = {
    "ENGINE": "model.engine",
    "MODEL_NAME": "model.name",
    "PRETRAIN_WEIGHTS": "model.pretrain_weights",
    "DEVICE": "model.device",
    "CONFIDENCE_THRESHOLD": "model.confidence_threshold",
    "DETECTION_THRESHOLD": "model.detection_threshold",
    "UP_THRESHOLD": "pushup.up_threshold",
    "DOWN_THRESHOLD": "pushup.down_threshold",
    "STABLE_FRAMES": "pushup.stable_frames",
    "SMOOTHING_ENABLED": "smoothing.enabled",
    "SMOOTHING_WINDOW": "smoothing.window_size",
    "SMOOTHING_METHOD": "smoothing.method",
    "OUTPUT_FPS": "video.output_fps",
    "LOG_LEVEL": "runtime.log_level",
    "DEVELOPER_MODE": "runtime.developer_mode",
}


def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Convert ``#RRGGBB`` to an OpenCV BGR tuple.

    Args:
        hex_color: Hex colour, with or without the leading ``#``.

    Returns:
        ``(blue, green, red)`` integer tuple.

    Raises:
        ValueError: If the string is not a valid 6-digit hex colour.
    """
    raw = hex_color.lstrip("#")
    if len(raw) != 6:
        raise ValueError(f"Invalid hex colour: {hex_color!r}")
    r, g, b = int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)
    return (b, g, r)


@dataclass(slots=True)
class PoseModelConfig:
    """Pose-estimation backend configuration."""

    engine: str = "rf-detr"
    name: str = "rf-detr-keypoint-preview"
    class_name: str = "RFDETRKeypointPreview"
    pretrain_weights: str = ""
    confidence_threshold: float = 0.5
    detection_threshold: float = 0.5
    resolution: int = 576
    device: str = "auto"
    preview_window: bool = False


@dataclass(slots=True)
class PersonSelectionConfig:
    """Primary-subject selection / temporal tracking configuration."""

    weights: Dict[str, float] = field(
        default_factory=lambda: {"confidence": 0.5, "size": 0.35, "centrality": 0.15}
    )
    tracking_iou_threshold: float = 0.30
    tracking_margin: float = 0.10
    max_lost_frames: int = 15


@dataclass(slots=True)
class PushUpConfig:
    """Push-up state machine configuration."""

    up_threshold: float = 150.0
    down_threshold: float = 90.0
    stable_frames: int = 3
    min_rep_duration: float = 0.55
    max_cycle_duration: float = 12.0
    min_elbow_bend: float = 25.0
    require_down_before_up: bool = True


@dataclass(slots=True)
class ValidationConfig:
    """Rules used to accept or reject a repetition."""

    min_depth_angle: float = 100.0
    max_alignment_deviation: float = 35.0
    max_arm_asymmetry: float = 25.0
    min_tracking_ratio: float = 0.70
    max_cadence_rpm: float = 0.0


@dataclass(slots=True)
class SmoothingConfig:
    """Angle smoothing configuration."""

    enabled: bool = True
    method: str = "moving_average"
    window_size: int = 5
    ema_alpha: float = 0.3


@dataclass(slots=True)
class MovementConfig:
    """Movement-phase analysis configuration."""

    min_phase_duration: float = 0.15
    velocity_window: int = 5


@dataclass(slots=True)
class PostureConfig:
    """Posture analysis thresholds."""

    alignment_tolerance: float = 12.0
    alignment_max_deviation: float = 35.0
    depth_full_angle: float = 90.0
    depth_zero_angle: float = 160.0
    symmetry_threshold: float = 20.0
    hips_high_angle: float = 150.0
    hips_low_angle: float = 205.0
    min_keypoint_confidence: float = 0.5


@dataclass(slots=True)
class FormConfig:
    """Form-score weighting, classification bands and control scoring."""

    weights: Dict[str, float] = field(
        default_factory=lambda: {"depth": 30.0, "alignment": 30.0, "symmetry": 20.0, "control": 20.0}
    )
    bands: Dict[str, float] = field(
        default_factory=lambda: {"excellent": 90.0, "good": 75.0, "needs_improvement": 60.0}
    )
    control: Dict[str, float] = field(
        default_factory=lambda: {"ideal_cycle_duration": 2.5, "min_cycle_duration": 0.8, "jerk_free_velocity": 60.0}
    )


@dataclass(slots=True)
class SkeletonConfig:
    """Skeleton rendering configuration."""

    line_thickness: int = 3
    joint_radius: int = 5
    highlight_radius: int = 7
    draw_confidence_labels: bool = False


@dataclass(slots=True)
class HudConfig:
    """Heads-up-display configuration."""

    enabled: bool = True
    base_width: int = 1280
    opacity: float = 0.62
    font_scale: float = 0.6
    feedback_history: int = 3
    show_angle_labels: bool = True
    show_bottom_telemetry: bool = True


@dataclass(slots=True)
class VisualizationConfig:
    """All rendering configuration (skeleton + HUD + colours)."""

    skeleton: SkeletonConfig = field(default_factory=SkeletonConfig)
    hud: HudConfig = field(default_factory=HudConfig)
    colors: Dict[str, Tuple[int, int, int]] = field(
        default_factory=lambda: {
            "good": hex_to_bgr("#22c55e"),
            "warning": hex_to_bgr("#facc15"),
            "bad": hex_to_bgr("#ef4444"),
            "tracking": hex_to_bgr("#3b82f6"),
            "accent": hex_to_bgr("#22d3ee"),
            "neutral": hex_to_bgr("#94a3b8"),
            "panel": hex_to_bgr("#0b1220"),
            "panel_border": hex_to_bgr("#1f2a44"),
            "text": hex_to_bgr("#e2e8f0"),
            "text_dim": hex_to_bgr("#94a3b8"),
            "skeleton": hex_to_bgr("#22d3ee"),
            "skeleton_invalid": hex_to_bgr("#64748b"),
            "skeleton_highlight": hex_to_bgr("#f59e0b"),
        }
    )

    def color(self, name: str, fallback: Tuple[int, int, int] = (255, 255, 255)) -> Tuple[int, int, int]:
        """Return the BGR colour registered under ``name``."""
        return self.colors.get(name, fallback)


@dataclass(slots=True)
class VideoConfig:
    """Video input/output configuration."""

    output_fps: float = 30.0
    save_processed_video: bool = True
    save_original_video: bool = False
    fourcc: str = "mp4v"
    fallback_fourccs: List[str] = field(default_factory=lambda: ["avc1", "XVID", "MJPG"])
    display_width: int = 960
    loop_webcam_demo: bool = True
    # Play the Streamlit preview at the source's own frame rate so demo/video
    # sessions feel like real-time instead of "as fast as the CPU can decode".
    realtime_pacing: bool = True
    # Default processed-video container: "mp4" | "avi" | "gif".
    export_format: str = "mp4"


@dataclass(slots=True)
class AnalyticsConfig:
    """Analytics / charting configuration."""

    history_stride: int = 2
    max_history_points: int = 1500
    rolling_form_window: int = 30
    chart_height: int = 300


@dataclass(slots=True)
class OutputConfig:
    """Filesystem output configuration."""

    base_dir: str = "outputs"
    screenshots_dir: str = "outputs/screenshots"
    videos_dir: str = "outputs/videos"
    reports_dir: str = "outputs/reports"
    screenshot_format: str = "jpg"
    video_name_template: str = "pushup_analysis_{timestamp}.mp4"
    report_formats: List[str] = field(default_factory=lambda: ["json", "csv", "md"])


@dataclass(slots=True)
class RuntimeConfig:
    """Application runtime configuration."""

    log_level: str = "INFO"
    log_file: str = "outputs/app.log"
    seed: int = 42
    developer_mode: bool = False


@dataclass(slots=True)
class Settings:
    """Root configuration object shared by every module."""

    model: PoseModelConfig = field(default_factory=PoseModelConfig)
    person_selection: PersonSelectionConfig = field(default_factory=PersonSelectionConfig)
    pushup: PushUpConfig = field(default_factory=PushUpConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    movement: MovementConfig = field(default_factory=MovementConfig)
    posture: PostureConfig = field(default_factory=PostureConfig)
    form: FormConfig = field(default_factory=FormConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    project_root: Path = PROJECT_ROOT
    config_path: Path = DEFAULT_CONFIG_PATH

    # ------------------------------------------------------------------ paths
    @property
    def screenshots_dir(self) -> Path:
        """Absolute directory used for screenshots."""
        return self._resolve(self.output.screenshots_dir)

    @property
    def videos_dir(self) -> Path:
        """Absolute directory used for exported videos."""
        return self._resolve(self.output.videos_dir)

    @property
    def reports_dir(self) -> Path:
        """Absolute directory used for generated reports."""
        return self._resolve(self.output.reports_dir)

    @property
    def log_path(self) -> Path:
        """Absolute path of the application log file."""
        return self._resolve(self.runtime.log_file)

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.project_root / path

    def ensure_directories(self) -> None:
        """Create every output directory if it does not exist yet."""
        for path in (self.screenshots_dir, self.videos_dir, self.reports_dir, self.log_path.parent):
            path.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------- mutators
    def clone(self) -> Settings:
        """Return a deep copy so a live session can be tweaked safely."""
        return copy.deepcopy(self)

    def apply_overrides(self, overrides: Dict[str, Any]) -> Settings:
        """Apply dotted-path overrides (used by the Streamlit sidebar).

        Args:
            overrides: Mapping of ``"pushup.up_threshold"`` style keys to values.

        Returns:
            ``self``, to allow chaining.
        """
        for dotted, value in overrides.items():
            if value is None:
                continue
            set_nested(self, dotted, value)
        return self


def _coerce(target: Any, value: Any) -> Any:
    """Coerce ``value`` to the type of ``target`` (YAML/env strings -> typed)."""
    if target is None or isinstance(value, type(target)):
        return value
    if isinstance(target, bool):
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(target, int) and not isinstance(target, bool):
        return int(float(value))
    if isinstance(target, float):
        return float(value)
    if isinstance(target, str):
        return str(value)
    return value


def set_nested(root: Any, dotted: str, value: Any) -> None:
    """Set ``root.a.b.c = value`` given the dotted key ``"a.b.c"``.

    Hex colour strings are converted to BGR tuples when the target field holds a
    colour triple, which keeps ``config.yaml`` human friendly.

    Raises:
        KeyError: If any segment of the path does not exist.
    """
    parts = dotted.split(".")
    node = root
    for part in parts[:-1]:
        if is_dataclass(node):
            if not hasattr(node, part):
                raise KeyError(f"Unknown configuration path segment: {part} in {dotted}")
            node = getattr(node, part)
        elif isinstance(node, dict):
            if part not in node:
                raise KeyError(f"Unknown configuration path segment: {part} in {dotted}")
            node = node[part]
        else:
            raise KeyError(f"Cannot traverse into {type(node).__name__} for {dotted}")

    leaf = parts[-1]
    if isinstance(node, dict):
        current = node.get(leaf)
        if isinstance(current, tuple) and isinstance(value, str):
            node[leaf] = hex_to_bgr(value)
        else:
            node[leaf] = value
        return

    if not hasattr(node, leaf):
        raise KeyError(f"Unknown configuration key: {dotted}")
    current = getattr(node, leaf)
    if is_dataclass(current) and isinstance(value, dict):
        # Nested section (e.g. visualization.skeleton) -> build the dataclass
        # recursively so typed access keeps working after a YAML load.
        setattr(node, leaf, _build_dataclass(type(current), value))
        return
    if isinstance(current, dict):
        if not isinstance(value, dict):
            raise TypeError(f"{dotted} expects a mapping, received {type(value).__name__}")
        target_template = dict(current)
        for key, raw in value.items():
            existing = target_template.get(key)
            if isinstance(existing, tuple) and isinstance(raw, str):
                target_template[key] = hex_to_bgr(raw)
            else:
                target_template[key] = _coerce(existing, raw)
        setattr(node, leaf, target_template)
    else:
        setattr(node, leaf, _coerce(current, value))


def _build_dataclass(cls: Any, payload: Dict[str, Any]) -> Any:
    """Instantiate a (possibly nested) config dataclass from a YAML mapping.

    Unknown keys are ignored so an older/newer ``config.yaml`` cannot crash the
    application; nested mappings recurse into their own dataclass.
    """
    kwargs: Dict[str, Any] = {}
    for field_definition in fields(cls):
        if field_definition.name not in payload:
            continue
        value = payload[field_definition.name]
        current = getattr(cls, field_definition.name, None)
        if is_dataclass(current) and isinstance(value, dict):
            kwargs[field_definition.name] = _build_dataclass(type(current), value)
        elif isinstance(current, dict) and isinstance(value, dict):
            template = dict(current)
            for key, raw in value.items():
                existing = template.get(key)
                template[key] = hex_to_bgr(raw) if isinstance(existing, tuple) and isinstance(raw, str) else _coerce(existing, raw)
            kwargs[field_definition.name] = template
        else:
            kwargs[field_definition.name] = _coerce(current, value)
    return cls(**kwargs)


def _flatten_colours(raw: Any) -> Any:
    """Convert hex colour strings inside a raw config mapping to BGR tuples."""
    if isinstance(raw, dict):
        return {key: _flatten_colours(value) for key, value in raw.items()}
    if isinstance(raw, str) and raw.startswith("#"):
        return hex_to_bgr(raw)
    return raw


def load_settings(
    config_path: str | os.PathLike[str] | None = None,
    *,
    load_env: bool = True,
) -> Settings:
    """Build a :class:`Settings` instance from defaults + YAML + environment.

    Args:
        config_path: Optional explicit YAML path. Defaults to ``config/config.yaml``.
        load_env: Whether to load ``.env`` and ``PUSHUP_*`` environment overrides.

    Returns:
        Fully populated :class:`Settings`.

    Raises:
        FileNotFoundError: If ``config_path`` is given explicitly but missing.
    """
    settings = Settings()
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH

    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        raw = _flatten_colours(raw)
        for section, payload in raw.items():
            if not hasattr(settings, section) or payload is None:
                continue
            if section == "project_root":
                continue
            if not isinstance(payload, dict):
                continue
            for key, value in payload.items():
                set_nested(settings, f"{section}.{key}", value)
        settings.config_path = path.resolve()
    elif config_path is not None:
        raise FileNotFoundError(f"Configuration file not found: {path}")

    if load_env:
        load_dotenv(settings.project_root / ".env", override=False)
        for suffix, dotted in ENV_OVERRIDES.items():
            raw_value = os.getenv(f"{ENV_PREFIX}{suffix}")
            if raw_value is not None and raw_value != "":
                set_nested(settings, dotted, raw_value)

    return settings


def settings_to_dict(settings: Settings) -> Dict[str, Any]:
    """Serialize settings to a JSON-friendly dict (for reports and debugging)."""
    from dataclasses import asdict

    payload = asdict(settings)
    payload["project_root"] = str(settings.project_root)
    payload["config_path"] = str(settings.config_path)
    return payload


def describe_dataclass(instance: Any) -> Dict[str, Any]:
    """Recursively describe a dataclass (used by the developer-mode panel)."""
    if is_dataclass(instance):
        return {f.name: describe_dataclass(getattr(instance, f.name)) for f in fields(instance)}
    if isinstance(instance, dict):
        return {key: describe_dataclass(value) for key, value in instance.items()}
    if isinstance(instance, tuple):
        return list(instance)
    return instance

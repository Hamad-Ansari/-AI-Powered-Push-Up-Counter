"""🏋️ PULSE·REP — AI-Powered Push-Up Counter (Streamlit dashboard).

Real-Time Pose Detection & Exercise Form Analysis.

Run with::

    streamlit run app.py

Layout + wiring only — the heavy lifting lives in :mod:`src`.  The visual
design system (tokens, CSS, HTML/SVG components) lives in :mod:`src.ui`; a
background :class:`~src.video.session_controller.SessionController` thread does
the capture/analyse/render work and the UI polls it for the latest frame.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import streamlit as st

from config.settings import Settings, load_settings
from src.analytics import charts
from src.analysis.movement_analyzer import ExerciseState
from src.ui import components as ui
from src.ui import theme
from src.utils.logger import get_logger, setup_logging
from src.video.session_controller import SessionController, build_source

logger = get_logger(__name__)

P = theme.PALETTE


# --------------------------------------------------------------------------- #
# Page + cached settings
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_settings() -> Settings:
    """Load settings once per server process."""
    settings = load_settings()
    settings.ensure_directories()
    setup_logging(settings.runtime.log_level, settings.log_path)
    return settings


def configure_page() -> None:
    """Set the browser tab title, favicon and inject the design system."""
    st.set_page_config(page_title="PULSE·REP — AI Push-Up Counter", page_icon="🏋️", layout="wide")
    theme.inject()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar(settings: Settings) -> dict:
    """Render the sidebar controls; returns the chosen options."""
    with st.sidebar:
        st.markdown(ui.sidebar_brand(), unsafe_allow_html=True)
        st.caption(theme.BRAND_SUB)

        st.markdown(ui.section("📹", "Input Source"), unsafe_allow_html=True)
        mode = st.radio(
            "Source",
            ["demo", "webcam", "video"],
            label_visibility="collapsed",
            format_func=lambda m: {"demo": "🎬 Demo Video", "webcam": "📷 Live Webcam", "video": "⬆️ Upload Video"}[m],
            help="Demo needs no camera or model weights — great for a quick look.",
        )
        video_path = None
        camera_index = 0
        demo_cycle = 3.0
        demo_form = "good"
        if mode == "video":
            uploaded = st.file_uploader("Upload MP4 / AVI / MOV", type=["mp4", "avi", "mov", "mkv", "webm"])
            if uploaded is not None:
                target = Path(settings.project_root) / "outputs" / "uploads"
                target.mkdir(parents=True, exist_ok=True)
                video_path = target / uploaded.name
                video_path.write_bytes(uploaded.getbuffer())
            elif (Path(settings.project_root) / "assets" / "demo" / "demo_pushup.mp4").exists():
                video_path = Path(settings.project_root) / "assets" / "demo" / "demo_pushup.mp4"
                st.info("No upload yet — using the bundled demo clip.")
        elif mode == "webcam":
            camera_index = st.number_input("Camera index", 0, 10, 0, 1)
            st.caption("Webcam access requires running Streamlit locally.")
        else:  # demo
            demo_cycle = st.slider("Rep duration (s)", 1.2, 5.0, 3.0, 0.1, help="Shorter = faster, more real-life tempo.")
            demo_form = st.selectbox(
                "Demo form",
                ["good", "hips_high", "hips_low", "uneven"],
                format_func=lambda f: {
                    "good": "✅ Good form",
                    "hips_high": "⚠️ Hips too high",
                    "hips_low": "⚠️ Hips sagging",
                    "uneven": "⚠️ Uneven arms",
                }[f],
            )

        st.markdown(ui.section("💾", "Output"), unsafe_allow_html=True)
        export_format = st.selectbox(
            "Processed-video format",
            ["mp4", "avi", "gif"],
            format_func=lambda f: {"mp4": "MP4 video", "avi": "AVI video", "gif": "Animated GIF"}[f],
        )
        realtime = st.toggle(
            "Real-time playback pacing",
            value=bool(settings.video.realtime_pacing),
            help="Play demo/video previews at the source frame rate instead of as-fast-as-possible.",
        )

        st.markdown(ui.section("🧠", "Model"), unsafe_allow_html=True)
        engine = st.selectbox(
            "Pose engine",
            ["demo", "rf-detr"],
            index=0 if settings.model.engine == "demo" else 1,
            format_func=lambda e: {"demo": "Demo (no weights)", "rf-detr": "RF-DETR Keypoint"}[e],
            help="RF-DETR downloads the COCO keypoint checkpoint on first use (needs the `rfdetr` package).",
        )
        confidence = st.slider("Keypoint confidence", 0.1, 0.95, float(settings.model.confidence_threshold), 0.05)

        st.markdown(ui.section("📐", "Thresholds"), unsafe_allow_html=True)
        up_threshold = st.slider("UP angle (°)", 120.0, 175.0, float(settings.pushup.up_threshold), 1.0)
        down_threshold = st.slider("DOWN angle (°)", 60.0, 130.0, float(settings.pushup.down_threshold), 1.0)
        stable_frames = st.slider("Stable frames", 1, 8, int(settings.pushup.stable_frames), 1)

        st.markdown(ui.section("🌊", "Smoothing"), unsafe_allow_html=True)
        smoothing_on = st.toggle("Enable smoothing", value=bool(settings.smoothing.enabled))
        smoothing_method = st.selectbox(
            "Method",
            ["moving_average", "ema"],
            index=0 if settings.smoothing.method == "moving_average" else 1,
            format_func=lambda m: {"moving_average": "Moving average", "ema": "Exponential (EMA)"}[m],
            disabled=not smoothing_on,
        )
        window = st.slider("Window size", 1, 15, int(settings.smoothing.window_size), 1, disabled=not smoothing_on)

        st.divider()
        developer = st.toggle("🛠️ Developer mode", value=bool(settings.runtime.developer_mode))

    form_map = {"good": (0.0, 0.0), "hips_high": (0.10, 0.0), "hips_low": (-0.10, 0.0), "uneven": (0.0, 25.0)}
    hips_offset, arm_asymmetry = form_map.get(demo_form, (0.0, 0.0))
    return {
        "mode": mode,
        "video_path": video_path,
        "camera_index": int(camera_index),
        "demo_cycle": float(demo_cycle),
        "hips_offset": float(hips_offset),
        "arm_asymmetry": float(arm_asymmetry),
        "export_format": export_format,
        "realtime": bool(realtime),
        "engine": engine,
        "confidence": confidence,
        "up_threshold": up_threshold,
        "down_threshold": down_threshold,
        "stable_frames": stable_frames,
        "smoothing_on": smoothing_on,
        "smoothing_method": smoothing_method,
        "window": window,
        "developer": developer,
    }


def apply_overrides(settings: Settings, options: dict) -> Settings:
    """Clone the settings and apply the sidebar overrides."""
    tuned = settings.clone()
    tuned.model.engine = options["engine"]
    tuned.model.confidence_threshold = options["confidence"]
    tuned.pushup.up_threshold = options["up_threshold"]
    tuned.pushup.down_threshold = options["down_threshold"]
    tuned.pushup.stable_frames = options["stable_frames"]
    tuned.smoothing.enabled = options["smoothing_on"]
    tuned.smoothing.method = options["smoothing_method"]
    tuned.smoothing.window_size = options["window"]
    tuned.runtime.developer_mode = options["developer"]
    tuned.video.realtime_pacing = options["realtime"]
    tuned.video.export_format = options["export_format"]
    return tuned


# --------------------------------------------------------------------------- #
# Session lifecycle
# --------------------------------------------------------------------------- #
def config_signature(options: dict) -> str:
    """Stable string identifying the current configuration (to detect changes)."""
    return "|".join(
        str(options[key])
        for key in (
            "mode",
            "video_path",
            "camera_index",
            "demo_cycle",
            "hips_offset",
            "arm_asymmetry",
            "export_format",
            "realtime",
            "engine",
            "confidence",
            "up_threshold",
            "down_threshold",
            "stable_frames",
            "smoothing_on",
            "smoothing_method",
            "window",
        )
    )


def get_controller(settings: Settings, options: dict) -> SessionController:
    """Create (or restart on config change) the session controller in session state."""
    signature = config_signature(options)
    existing = st.session_state.get("controller")
    if existing is not None and st.session_state.get("config_signature") == signature:
        return existing

    if existing is not None:
        existing.stop()

    factory = build_source(
        settings,
        options["mode"],
        video_path=options["video_path"],
        camera_index=options["camera_index"],
        demo_cycle=options["demo_cycle"],
        hips_offset=options["hips_offset"],
        arm_asymmetry=options["arm_asymmetry"],
    )

    # In demo mode the analysed pose must carry the same form fault as the
    # rendered stick figure, so build an adapter that mirrors the source params.
    analyzer = None
    if options["mode"] == "demo":
        from src.pose.demo_adapter import SyntheticPushUpAdapter
        from src.pose.pose_detector import PoseDetector
        from src.video.video_processor import FrameAnalyzer

        adapter = SyntheticPushUpAdapter(
            cycle_seconds=options["demo_cycle"],
            fps=settings.video.output_fps,
            hips_offset=options["hips_offset"],
            arm_asymmetry=options["arm_asymmetry"],
        )
        analyzer = FrameAnalyzer(settings, pose_detector=PoseDetector(settings, adapter=adapter))

    controller = SessionController(
        settings,
        factory,
        analyzer=analyzer,
        auto_export=settings.video.save_processed_video,
        export_format=options["export_format"],
    )
    st.session_state["controller"] = controller
    st.session_state["config_signature"] = signature
    return controller


# --------------------------------------------------------------------------- #
# Live UI blocks (pure HTML via src.ui.components)
# --------------------------------------------------------------------------- #
def _deg(value: float) -> str:
    return f"{value:.0f}°" if math.isfinite(value) else "—"


def build_kpis(result, summary) -> str:
    """Headline KPI row: reps, session time, throughput, tracking."""
    fps = result.fps if result is not None else 0.0
    cards = [
        ui.kpi("🔁", "Total Reps", f"{summary.total_reps}", f"{summary.valid_reps} valid · {summary.invalid_reps} invalid", P["accent"]),
        ui.kpi("⏱️", "Session", summary.duration_label, f"{summary.reps_per_minute:.1f} reps/min", P["accent_2"]),
        ui.kpi("⚡", "FPS", f"{fps:.0f}", f"{summary.average_fps:.1f} avg", P["info"]),
        ui.kpi(
            "🎯",
            "Tracking",
            f"{summary.detection_rate * 100:.0f}%",
            f"{summary.frames_with_pose}/{summary.frames_processed} frames",
            P["good"],
        ),
    ]
    return ui.kpi_row(cards)


def build_live_panel(result) -> str:
    """Right-hand live panel: form ring, state pill, angle meters, feedback."""
    form = result.form if result is not None else None
    angles = result.angles if result is not None else None
    state = result.state if result is not None else ExerciseState.UNKNOWN

    score = form.form_score if form is not None else 0.0
    color = theme.grade_color(form.form_status) if form is not None else P["muted"]
    label, state_color, glyph = theme.state_meta(state)

    parts = [
        f"<div class='pu-card' style='margin-bottom:10px;'>{ui.score_ring(score, color)}</div>",
        f"<div style='text-align:center;margin:2px 0 10px;'>{ui.pill(f'{glyph} {label}', state_color)}</div>",
    ]

    left = angles.left_elbow if angles is not None else float("nan")
    right = angles.right_elbow if angles is not None else float("nan")
    align = angles.body_alignment if angles is not None else float("nan")
    diff = angles.elbow_diff if angles is not None else float("nan")
    parts.append(
        "<div class='pu-card'>"
        + ui.meter("Left elbow", _deg(left), left / 180.0, P["accent_2"])
        + ui.meter("Right elbow", _deg(right), right / 180.0, P["accent_2"])
        + ui.meter("Body alignment", _deg(align), align / 180.0, P["accent"])
        + ui.meter("Arm asymmetry", _deg(diff), diff / 60.0, P["warn"])
        + "</div>"
    )

    if form is None:
        parts.append(ui.feedback_line("Start the session and get into frame to receive live feedback."))
    else:
        grade = form.form_status
        accent = theme.grade_color(grade)
        parts.append(ui.feedback_head(grade.emoji, grade.headline, form.form_score, accent))
        for line in form.feedback[:4]:
            parts.append(ui.feedback_line(line, accent))
    return "".join(parts)


def render_toolbar(controller: SessionController) -> None:
    """Transport controls (start/stop/pause/step/screenshot/reset/export)."""
    col = st.columns(7)
    with col[0]:
        if st.button("▶️ Start", use_container_width=True, type="primary", disabled=controller.running):
            if controller.start():
                st.session_state["session_active"] = True
            else:
                st.session_state["start_error"] = controller.error
    with col[1]:
        if st.button("⏹️ Stop", use_container_width=True, disabled=not controller.running):
            controller.stop()
            st.session_state["session_active"] = False
    with col[2]:
        label = "▶️ Resume" if controller.paused else "⏸️ Pause"
        if st.button(label, use_container_width=True, disabled=not controller.running):
            controller.toggle_pause()
    with col[3]:
        if st.button("⏭️ Step", use_container_width=True, disabled=not controller.running):
            controller.step_frame()
    with col[4]:
        if st.button("📸 Screenshot", use_container_width=True, disabled=not controller.running):
            controller.request_screenshot()
    with col[5]:
        if st.button("🔄 Reset", use_container_width=True):
            controller.analyzer.reset(full=True)
    with col[6]:
        export_label = "⏺️ Stop Export" if controller.exporting else "⏺️ Export Video"
        if st.button(export_label, use_container_width=True, disabled=not controller.running):
            if controller.exporting:
                controller.stop_export()
            else:
                controller.start_export()


# --------------------------------------------------------------------------- #
# Post-session tabs
# --------------------------------------------------------------------------- #
def render_analytics_tab(controller: SessionController, settings: Settings) -> None:
    """Interactive charts + workout summary."""
    session = controller.analyzer.session
    summary = controller.summary()

    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.elbow_angle_chart(session, settings.analytics), use_container_width=True)
        st.plotly_chart(charts.form_score_chart(session, settings.analytics), use_container_width=True)
        st.plotly_chart(charts.reps_over_time_chart(session, settings.analytics), use_container_width=True)
    with right:
        st.plotly_chart(charts.phase_timeline_chart(session, settings.analytics), use_container_width=True)
        st.plotly_chart(charts.elbow_comparison_chart(session, settings.analytics), use_container_width=True)
        st.plotly_chart(charts.arm_symmetry_chart(session, settings.analytics), use_container_width=True)

    st.markdown(ui.section("📋", "Workout Summary"), unsafe_allow_html=True)
    rows = summary.headline_rows()
    cols = st.columns(3)
    for index, (label, value) in enumerate(rows):
        cols[index % 3].markdown(f"**{label}**  \n{value}")


def render_reports_tab(controller: SessionController) -> None:
    """Downloads: frames CSV, JSON report, processed video."""
    session = controller.analyzer.session
    summary = controller.summary()
    dcol = st.columns(3)
    with dcol[0]:
        frame_csv = session.to_dataframe().to_csv(index=False).encode()
        st.download_button("⬇️ Download frames CSV", frame_csv, "pushup_frames.csv", "text/csv", use_container_width=True)
    with dcol[1]:
        paths = controller.generate_reports(("json",)) if summary.total_reps or session.frames else None
        if paths and paths.json_path and paths.json_path.exists():
            st.download_button(
                "⬇️ Download JSON report", paths.json_path.read_bytes(), paths.json_path.name, "application/json", use_container_width=True
            )
        else:
            st.caption("Finish a session to generate the JSON report.")
    with dcol[2]:
        export_path = controller.export_path
        if export_path and export_path.exists():
            st.download_button(
                "⬇️ Download processed video", export_path.read_bytes(), export_path.name, "video/mp4", use_container_width=True
            )
        else:
            st.caption("Export a video to enable download.")


def render_screenshots_tab(settings: Settings) -> None:
    """Latest captured screenshots."""
    shots = sorted(
        settings.screenshots_dir.glob(f"screenshot_*.{settings.output.screenshot_format}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not shots:
        st.caption("No screenshots yet — use 📸 during a session.")
        return
    cols = st.columns(min(3, len(shots)))
    for index, shot in enumerate(shots[:3]):
        cols[index].image(str(shot), use_container_width=True, caption=shot.name)


def render_developer_tab(controller: SessionController, enabled: bool) -> None:
    """Backend diagnostics (developer mode only)."""
    if not enabled:
        st.caption("Enable 🛠️ Developer mode in the sidebar to see backend diagnostics.")
        return
    dev_frame, dev_result = controller.latest()
    st.json(
        {
            "backend": controller.analyzer.pose_detector.adapter.describe(),
            "running": controller.running,
            "paused": controller.paused,
            "error": controller.error,
            "source": controller.source_name,
            "detection_rate": round(controller.analyzer.pose_detector.detection_rate(), 3),
            "visible_keypoints": dev_result.visible_keypoints if dev_result else 0,
        }
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    configure_page()
    base_settings = get_settings()
    options = render_sidebar(base_settings)
    settings = apply_overrides(base_settings, options)

    hero_ph = st.empty()
    controller = get_controller(settings, options)
    if st.session_state.get("start_error"):
        st.error(st.session_state["start_error"])
        st.session_state["start_error"] = None

    st.markdown(ui.section("🎛️", "Session Controls"), unsafe_allow_html=True)
    render_toolbar(controller)

    kpi_ph = st.empty()
    video_col, live_col = st.columns([3, 2], gap="large")
    video_ph = video_col.empty()
    live_ph = live_col.empty()

    def status_of() -> tuple[str, str]:
        if controller.running and controller.paused:
            return "PAUSED", P["warn"]
        if controller.running:
            return "LIVE", P["good"]
        return "IDLE", P["muted"]

    def paint_live(result, summary) -> None:
        status, color = status_of()
        hero_ph.markdown(ui.hero(status, color), unsafe_allow_html=True)
        kpi_ph.markdown(build_kpis(result, summary), unsafe_allow_html=True)
        frame, _ = controller.latest()
        if frame is not None:
            video_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            video_ph.markdown(ui.video_frame_placeholder(), unsafe_allow_html=True)
        live_ph.markdown(build_live_panel(result), unsafe_allow_html=True)

    # Live polling loop: runs while the session is active.
    if controller.running or controller.latest()[0] is not None:
        target_fps = 25.0
        while True:
            frame, result = controller.latest()
            summary = controller.summary()
            paint_live(result, summary)
            if not controller.running:
                break
            time.sleep(1.0 / target_fps)
    else:
        paint_live(None, controller.summary())

    if not controller.running and st.session_state.get("session_active"):
        st.success("Session finished. Analytics and downloads are below.")
        st.session_state["session_active"] = False

    tab_analytics, tab_reports, tab_shots, tab_dev = st.tabs(
        ["📊 Analytics", "📥 Reports & Downloads", "📸 Screenshots", "🛠️ Developer"]
    )
    with tab_analytics:
        render_analytics_tab(controller, settings)
    with tab_reports:
        render_reports_tab(controller)
    with tab_shots:
        render_screenshots_tab(settings)
    with tab_dev:
        render_developer_tab(controller, options["developer"])

    st.markdown(ui.footer(), unsafe_allow_html=True)


if __name__ == "__main__":
    main()

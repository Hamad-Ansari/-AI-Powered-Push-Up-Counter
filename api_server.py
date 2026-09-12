"""Optional FastAPI backend.

The Streamlit app is self-contained and does **not** need this server. Run it
only if you want the analysis exposed as a REST API (e.g. to drive a separate
frontend or integrate with another service)::

    pip install "fastapi>=0.110" "uvicorn[standard]>=0.29" python-multipart
    uvicorn api_server:app --reload

Endpoints:
    GET  /health                liveness + backend status
    POST /analyze/image         analyse a single uploaded image (JSON keypoints/angles/form)
    POST /analyze/video         batch-analyse an uploaded video -> summary + reports
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from config.settings import load_settings
from src.analysis.angle_calculator import compute_angles
from src.pose.pose_detector import PoseDetector
from src.utils.logger import get_logger, setup_logging
from src.video.sources import VideoFileSource
from src.video.video_processor import process_source

logger = get_logger(__name__)

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import JSONResponse
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "FastAPI is optional. Install it with:\n"
        '  pip install "fastapi>=0.110" "uvicorn[standard]>=0.29" python-multipart'
    ) from exc

settings = load_settings()
settings.ensure_directories()
setup_logging(settings.runtime.log_level, settings.log_path)

app = FastAPI(
    title="AI-Powered Push-Up Counter API",
    description="Real-time pose detection & exercise form analysis (RF-DETR).",
    version="1.0.0",
)

_detector: PoseDetector | None = None


def get_detector() -> PoseDetector:
    """Lazily build and cache a shared pose detector."""
    global _detector
    if _detector is None:
        _detector = PoseDetector(settings)
        _detector.load_model()
    return _detector


@app.get("/health")
def health() -> Dict[str, Any]:
    """Liveness probe including the pose backend status."""
    detector = get_detector()
    return {
        "status": "ok",
        "backend": detector.adapter.describe(),
        "engine": settings.model.engine,
    }


def _decode_upload(data: bytes) -> np.ndarray:
    """Decode uploaded bytes into a BGR image array."""
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode the uploaded image")
    return image


@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)) -> JSONResponse:
    """Analyse a single image: keypoints, joint angles and form score."""
    detector = get_detector()
    image = _decode_upload(await file.read())

    pose = detector.detect(image)
    if pose is None:
        return JSONResponse({"detected": False, "message": "No person detected", "keypoints": {}})

    angles = compute_angles(pose)
    return JSONResponse(
        {
            "detected": True,
            "detection_confidence": round(float(pose.detection_confidence), 4),
            "visible_keypoints": pose.valid_count,
            "keypoints": {name: kp.as_dict() for name, kp in pose.keypoints.items() if kp.is_usable()},
            "angles": angles.as_dict(),
            "critical_missing": pose.critical_missing(),
        }
    )


@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...), max_frames: int | None = None) -> JSONResponse:
    """Batch-analyse an uploaded video and return the workout summary."""
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        source = VideoFileSource(tmp_path, fallback_fps=settings.video.output_fps)
        report, analyzer = process_source(source, settings, export_video=False, max_frames=max_frames)
        summary = analyzer.session.summary()
        return JSONResponse(
            {
                "frames_processed": report.frames_processed,
                "error": report.error,
                "summary": summary.to_dict(),
                "reps": [rep.to_dict() for rep in analyzer.session.reps],
            }
        )
    except Exception as exc:  # noqa: BLE001 - return a clean 4xx/5xx
        logger.exception("Video analysis failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)

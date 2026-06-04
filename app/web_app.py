from __future__ import annotations

import sys
import base64
import tempfile
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from driver_state.config import load_config, ensure_project_dirs
from driver_state.inference.predictor import DriverStatePredictor
from driver_state.inference.video_analyzer import VideoAnalyzer
from driver_state.visualization.overlay import draw_overlay
from driver_state.visualization.attention_overlay import overlay_attention_on_frame

DEFAULT_CONFIG = ROOT / "config" / "config.yaml"
DEFAULT_MODEL = ROOT / "artifacts" / "models" / "frame_model_best.keras"

app = FastAPI(title="DriverStateNN Web", version="1.1.0")
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")
OUTPUT_VIDEOS_DIR = ROOT / "artifacts" / "videos"
OUTPUT_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs/videos", StaticFiles(directory=OUTPUT_VIDEOS_DIR), name="outputs_videos")
templates = Jinja2Templates(directory=ROOT / "app" / "templates")

_cfg = None
_predictor: Optional[DriverStatePredictor] = None
_model_path = DEFAULT_MODEL


def get_runtime(model_path: str | Path | None = None):
    global _cfg, _predictor, _model_path
    model_path = Path(model_path or _model_path)
    if _cfg is None:
        _cfg = load_config(DEFAULT_CONFIG)
        ensure_project_dirs(_cfg)
    if _predictor is None or Path(_model_path) != model_path:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Не найдена модель {model_path}. Сначала обучите модель: python scripts/train_frame_model.py"
            )
        _model_path = model_path
        _predictor = DriverStatePredictor(model_path, _cfg)
    return _cfg, _predictor


def decode_base64_image(data_url: str):
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    data = base64.b64decode(data_url)
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def encode_jpeg_base64(frame) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise RuntimeError("Не удалось закодировать кадр")
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/frame")
async def analyze_frame(
    image_base64: str = Form(...),
    attention: bool = Form(True),
    model_path: str | None = Form(None),
):
    try:
        cfg, predictor = get_runtime(model_path)
        frame = decode_base64_image(image_base64)
        if frame is None:
            return JSONResponse({"ok": False, "error": "Кадр не распознан"}, status_code=400)

        # Веб-камера тоже анализируется по полному кадру, как обучающие изображения State Farm.
        # Большой bbox лица/водителя не рисуем, оставляем только Grad-CAM зоны внимания.
        if attention:
            result, att = predictor.predict_with_attention(
                frame,
                attention_threshold=float(cfg.decision.get("attention_threshold", 0.55)),
            )
            overlay_attention_on_frame(frame, None, att.get("heatmap"), att.get("boxes", []))
            boxes = att.get("boxes", [])
        else:
            result = predictor.predict(frame)
            boxes = []

        draw_overlay(frame, None, result)
        return {"ok": True, "result": result, "image": encode_jpeg_base64(frame), "attention_boxes": boxes, "input_mode": "full_frame"}
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.post("/api/video")
async def analyze_uploaded_video(
    file: UploadFile = File(...),
    attention: bool = Form(True),
    model_path: str | None = Form(None),
):
    try:
        cfg, _ = get_runtime(model_path)
        suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)
        analyzer = VideoAnalyzer(model_path or _model_path, cfg, show_attention=attention)
        report = analyzer.analyze(str(tmp_path), save_video=True)
        tmp_path.unlink(missing_ok=True)

        video_url = None
        video_path = report.get("video_path")
        if video_path:
            video_url = f"/outputs/videos/{Path(video_path).name}"

        return {"ok": True, "report": report, "video_url": video_url}
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

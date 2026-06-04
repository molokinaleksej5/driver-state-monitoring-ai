from __future__ import annotations

from pathlib import Path
import csv
import cv2

from driver_state.config import AppConfig
from driver_state.inference.predictor import DriverStatePredictor
from driver_state.reporting.session_report import build_session_report
from driver_state.visualization.overlay import draw_overlay
from driver_state.visualization.attention_overlay import overlay_attention_on_frame
from driver_state.utils.io import timestamp_name


def crop_driver_roi(frame):
    """
    Выделяет область водителя, рук, руля и части салона.

    В модель передаётся не весь широкий кадр, а область, где находятся
    водитель и основные признаки поведения. Это уменьшает влияние дороги,
    стекла и лишних объектов справа.
    """
    h, w = frame.shape[:2]

    x1 = int(w * 0.00)
    y1 = int(h * 0.02)
    x2 = int(w * 0.62)
    y2 = int(h * 1.00)

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    roi = frame[y1:y2, x1:x2]
    roi_box = (x1, y1, x2, y2)

    return roi, roi_box


def shift_attention_boxes(boxes, roi_box):
    """
    Переносит координаты Grad-CAM прямоугольников из ROI в полный кадр.

    Используется формат:
    x1, y1, x2, y2.

    Функция дополнительно приводит координаты к int, потому что иногда
    координаты могут прийти строками из промежуточного словаря.
    """
    if not boxes:
        return []

    x_offset, y_offset, _, _ = roi_box
    x_offset = int(x_offset)
    y_offset = int(y_offset)

    shifted = []

    for box in boxes:
        if isinstance(box, dict):
            if all(k in box for k in ["x1", "y1", "x2", "y2"]):
                x1 = int(float(box["x1"]))
                y1 = int(float(box["y1"]))
                x2 = int(float(box["x2"]))
                y2 = int(float(box["y2"]))

                shifted.append(
                    (
                        x1 + x_offset,
                        y1 + y_offset,
                        x2 + x_offset,
                        y2 + y_offset,
                    )
                )
            else:
                x = int(float(box.get("x", 0)))
                y = int(float(box.get("y", 0)))
                w = int(float(box.get("w", box.get("width", 40))))
                h = int(float(box.get("h", box.get("height", 40))))

                shifted.append(
                    (
                        x + x_offset,
                        y + y_offset,
                        x + w + x_offset,
                        y + h + y_offset,
                    )
                )

            continue

        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue

        x1, y1, x2, y2 = box

        x1 = int(float(x1))
        y1 = int(float(y1))
        x2 = int(float(x2))
        y2 = int(float(y2))

        shifted.append(
            (
                x1 + x_offset,
                y1 + y_offset,
                x2 + x_offset,
                y2 + y_offset,
            )
        )

    return shifted


class VideoAnalyzer:
    """
    Анализирует видео по области водителя.

    Модель получает driver ROI, Grad-CAM строится по этой области,
    затем прямоугольники внимания переносятся обратно на полный кадр.
    """

    def __init__(self, model_path: str | Path, cfg: AppConfig, show_attention: bool = False):
        self.cfg = cfg
        self.predictor = DriverStatePredictor(model_path, cfg)
        self.show_attention = show_attention

    def analyze(self, source: str | int, save_video: bool = True) -> dict:
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть источник видео: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or float(self.cfg.video["output_fps_fallback"])
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logs_dir = self.cfg.path("logs_dir")
        logs_dir.mkdir(parents=True, exist_ok=True)
        csv_path = logs_dir / timestamp_name("session", ".csv")

        writer = None
        video_path = None

        if save_video:
            out_dir = self.cfg.path("artifacts_dir") / "videos"
            out_dir.mkdir(parents=True, exist_ok=True)
            video_path = out_dir / timestamp_name("annotated", ".mp4")

            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

        fields = [
            "frame_idx",
            "time_sec",
            "face_detected",
            "state",
            "confidence",
            "risk_score",
            "risk_level",
            "danger_counter",
            "warning",
            "input_mode",
            "attention_boxes",
        ]

        frame_idx = 0
        process_every = int(self.cfg.video["process_every_n_frame"])

        last_result = {
            "state": "unknown",
            "confidence": 0.0,
            "risk_score": 0.0,
            "risk_level": "green",
            "danger_counter": 0,
            "warning": "",
        }

        last_attention = {
            "heatmap": None,
            "boxes": [],
            "boxes_on_frame": [],
            "roi_box": None,
        }

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            log = csv.DictWriter(f, fieldnames=fields)
            log.writeheader()

            while True:
                ok, frame = cap.read()

                if not ok:
                    break

                if frame_idx % process_every == 0:
                    roi, roi_box = crop_driver_roi(frame)

                    if self.show_attention:
                        last_result, last_attention = self.predictor.predict_with_attention(
                            roi,
                            attention_threshold=float(
                                self.cfg.decision.get("attention_threshold", 0.25)
                            ),
                        )

                        roi_boxes = last_attention.get("boxes", [])
                        boxes_on_frame = shift_attention_boxes(roi_boxes, roi_box)

                        last_attention["roi_box"] = roi_box
                        last_attention["boxes_on_frame"] = boxes_on_frame
                    else:
                        last_result = self.predictor.predict(roi)
                        last_attention = {
                            "heatmap": None,
                            "boxes": [],
                            "boxes_on_frame": [],
                            "roi_box": roi_box,
                        }

                draw_overlay(frame, None, last_result)

                if self.show_attention:
                    overlay_attention_on_frame(
                        frame,
                        None,
                        None,
                        last_attention.get("boxes_on_frame", []),
                    )

                if writer:
                    writer.write(frame)

                row = {
                    "frame_idx": frame_idx,
                    "time_sec": round(frame_idx / fps, 3),
                    "face_detected": True,
                    "state": last_result.get("state", "unknown"),
                    "confidence": last_result.get("confidence", 0.0),
                    "risk_score": last_result.get("risk_score", 0.0),
                    "risk_level": last_result.get("risk_level", "green"),
                    "danger_counter": last_result.get("danger_counter", 0),
                    "warning": last_result.get("warning", ""),
                    "input_mode": "driver_roi",
                    "attention_boxes": str(last_attention.get("boxes_on_frame", [])),
                }

                log.writerow(row)
                frame_idx += 1

        cap.release()

        if writer:
            writer.release()

        return build_session_report(csv_path, self.cfg, video_path)
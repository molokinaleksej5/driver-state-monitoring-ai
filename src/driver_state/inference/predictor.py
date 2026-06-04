from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

from driver_state.config import AppConfig
from driver_state.inference.smoother import ProbabilitySmoother
from driver_state.inference.risk import RiskEstimator
from driver_state.explainability.attention import build_attention


class DriverStatePredictor:
    def __init__(self, model_path: str | Path, cfg: AppConfig, smoothing: bool = True):
        self.cfg = cfg
        self.model = tf.keras.models.load_model(model_path)

        self.smoother = (
            ProbabilitySmoother(int(cfg.decision["smoothing_window"]))
            if smoothing
            else None
        )

        try:
            self.risk = RiskEstimator(
                cfg.class_names,
                cfg.dangerous_states,
                float(cfg.decision["risk_green"]),
                float(cfg.decision["risk_yellow"]),
                float(cfg.decision["risk_red"]),
                confidence_threshold=float(cfg.decision.get("confidence_threshold", 0.78)),
                risk_weights=getattr(cfg, "risk_weights", None),
            )
        except TypeError:
            self.risk = RiskEstimator(
                cfg.class_names,
                cfg.dangerous_states,
                float(cfg.decision["risk_green"]),
                float(cfg.decision["risk_yellow"]),
                float(cfg.decision["risk_red"]),
            )

        self.danger_counter = 0

    def preprocess(self, bgr_img) -> np.ndarray:
        img = cv2.resize(bgr_img, self.cfg.image_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32")
        return np.expand_dims(img, axis=0)

    def _postprocess_probabilities(self, probs: np.ndarray) -> dict:
        probs = np.asarray(probs, dtype="float32")

        if self.smoother:
            probs = self.smoother.update(probs)

        idx = int(np.argmax(probs))
        state = self.cfg.class_names[idx]
        confidence = float(probs[idx])

        confidence_threshold = float(self.cfg.decision.get("confidence_threshold", 0.78))
        danger_frames_threshold = int(self.cfg.decision.get("danger_frames_threshold", 24))

        is_dangerous = state in self.cfg.dangerous_states
        is_confident = confidence >= confidence_threshold

        if state == "c0_safe":
            self.danger_counter = max(0, self.danger_counter - 2)
        elif is_dangerous and is_confident:
            self.danger_counter += 1
        else:
            self.danger_counter = max(0, self.danger_counter - 1)

        risk_score = self.risk.score(
            probs,
            self.danger_counter,
            danger_frames_threshold,
        )

        risk_level = self.risk.level(risk_score)

        warning = risk_level in ["red", "orange"]

        return {
            "state": state,
            "confidence": round(confidence, 4),
            "probabilities": {
                name: float(probs[i])
                for i, name in enumerate(self.cfg.class_names)
            },
            "danger_counter": int(self.danger_counter),
            "risk_score": float(risk_score),
            "risk_level": risk_level,
            "warning": bool(warning),
        }

    def predict(self, frame_img) -> dict:
        batch = self.preprocess(frame_img)
        probs = self.model.predict(batch, verbose=0)[0]
        return self._postprocess_probabilities(probs)

    def _fallback_attention_boxes(self, image_shape) -> list[tuple[int, int, int, int]]:
        h, w = image_shape[:2]

        boxes = [
            (
                int(w * 0.18),
                int(h * 0.12),
                int(w * 0.42),
                int(h * 0.34),
            ),
            (
                int(w * 0.30),
                int(h * 0.42),
                int(w * 0.58),
                int(h * 0.68),
            ),
            (
                int(w * 0.48),
                int(h * 0.52),
                int(w * 0.76),
                int(h * 0.82),
            ),
        ]

        clipped = []

        for x1, y1, x2, y2 in boxes:
            x1 = max(0, min(int(x1), w - 1))
            y1 = max(0, min(int(y1), h - 1))
            x2 = max(x1 + 1, min(int(x2), w))
            y2 = max(y1 + 1, min(int(y2), h))

            clipped.append((x1, y1, x2, y2))

        return clipped

    def predict_with_attention(
        self,
        frame_img,
        attention_threshold: float = 0.25,
    ) -> tuple[dict, dict]:
        batch = self.preprocess(frame_img)
        raw_probs = self.model.predict(batch, verbose=0)[0]
        class_idx = int(np.argmax(raw_probs))

        result = self._postprocess_probabilities(raw_probs)

        try:
            attention = build_attention(
                self.model,
                batch,
                image_size=(frame_img.shape[1], frame_img.shape[0]),
                class_index=class_idx,
                threshold=attention_threshold,
            )

            boxes = attention.boxes or []

            if not boxes:
                boxes = self._fallback_attention_boxes(frame_img.shape)

            attention_payload = {
                "heatmap": attention.heatmap,
                "boxes": boxes,
                "error": "",
            }

        except Exception as exc:
            attention_payload = {
                "heatmap": None,
                "boxes": self._fallback_attention_boxes(frame_img.shape),
                "error": str(exc),
            }

        return result, attention_payload

    def no_face_result(self) -> dict:
        self.danger_counter = max(0, self.danger_counter - 1)

        return {
            "state": "no_face",
            "confidence": 0.0,
            "probabilities": {},
            "danger_counter": int(self.danger_counter),
            "risk_score": 0.0,
            "risk_level": "unknown",
            "warning": False,
        }
from __future__ import annotations

import numpy as np


class RiskEstimator:


    def __init__(
        self,
        class_names: list[str],
        dangerous_states: list[str],
        green: float,
        yellow: float,
        red: float,
        confidence_threshold: float = 0.78,
        risk_weights: dict[str, float] | None = None,
    ):
        self.class_names = class_names
        self.dangerous_states = set(dangerous_states)

        self.green = float(green)
        self.yellow = float(yellow)
        self.red = float(red)
        self.confidence_threshold = float(confidence_threshold)

        self.risk_weights = risk_weights or {
            "c0_safe": 0.0,
            "c1_texting_right": 1.0,
            "c2_phone_right": 0.9,
            "c3_texting_left": 1.0,
            "c4_phone_left": 0.9,
            "c5_radio": 0.45,
            "c6_drinking": 0.65,
            "c7_reaching": 0.85,
            "c8_makeup": 0.8,
            "c9_talking": 0.35,
        }

    def score(self, probs: np.ndarray, danger_counter: int, threshold_frames: int) -> float:
        probs = np.asarray(probs, dtype="float32")

        if probs.ndim != 1:
            probs = probs.reshape(-1)

        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])
        state = self.class_names[top_idx]


        if state == "c0_safe":
            return 0.0

        is_dangerous = state in self.dangerous_states
        state_weight = float(self.risk_weights.get(state, 1.0 if is_dangerous else 0.0))

        if not is_dangerous:
            return 0.0

        temporal = min(1.0, float(danger_counter) / max(1, int(threshold_frames)))


        if confidence < self.confidence_threshold:
            risk = confidence * state_weight * 0.65
            return round(min(risk, self.yellow - 0.01), 4)


        risk = confidence * state_weight * (0.45 + 0.55 * temporal)

        return round(float(min(1.0, risk)), 4)

    def level(self, score: float) -> str:
        score = float(score)

        if score >= self.red:
            return "red"

        if score >= self.yellow:
            return "orange"

        if score >= self.green:
            return "yellow"

        return "green"
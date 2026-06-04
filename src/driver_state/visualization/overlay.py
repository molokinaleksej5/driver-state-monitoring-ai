from __future__ import annotations

import cv2

LEVEL_COLOR = {
    "green": (60, 200, 60),
    "yellow": (0, 220, 255),
    "orange": (0, 140, 255),
    "red": (0, 0, 255),
    "unknown": (180, 180, 180),
}


def draw_overlay(frame, bbox, result: dict) -> None:
    color = LEVEL_COLOR.get(result.get("risk_level", "unknown"), (255, 255, 255))
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    lines = [
        f"state: {result.get('state')}",
        f"confidence: {float(result.get('confidence', 0)):.2f}",
        f"risk: {float(result.get('risk_score', 0)):.2f} / {result.get('risk_level')}",
    ]
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (20, 35 + i * 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if result.get("warning"):
        cv2.putText(frame, "WARNING: unsafe driver state", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

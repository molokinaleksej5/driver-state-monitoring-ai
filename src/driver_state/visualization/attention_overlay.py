from __future__ import annotations

import cv2
import numpy as np


def _clip_box(box, width: int, height: int):
    x1, y1, x2, y2 = box

    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(x1 + 1, min(int(x2), width))
    y2 = max(y1 + 1, min(int(y2), height))

    return x1, y1, x2, y2


def _normalize_box(box):
    if isinstance(box, dict):
        if "x1" in box and "y1" in box and "x2" in box and "y2" in box:
            return (
                int(box["x1"]),
                int(box["y1"]),
                int(box["x2"]),
                int(box["y2"]),
            )

        x = int(box.get("x", 0))
        y = int(box.get("y", 0))
        w = int(box.get("w", box.get("width", 0)))
        h = int(box.get("h", box.get("height", 0)))

        return x, y, x + w, y + h

    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None

    x1, y1, x2, y2 = [int(v) for v in box]

    return x1, y1, x2, y2


def _draw_focus_box(frame, box, color=(0, 255, 255)):
    frame_h, frame_w = frame.shape[:2]

    x1, y1, x2, y2 = _clip_box(box, frame_w, frame_h)

    box_w = x2 - x1
    box_h = y2 - y1

    if box_w < 10 or box_h < 10:
        return

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        color,
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.18,
        frame,
        0.82,
        0,
        frame,
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        4,
    )

    corner = max(16, min(box_w, box_h) // 4)

    cv2.line(frame, (x1, y1), (x1 + corner, y1), color, 6)
    cv2.line(frame, (x1, y1), (x1, y1 + corner), color, 6)

    cv2.line(frame, (x2, y1), (x2 - corner, y1), color, 6)
    cv2.line(frame, (x2, y1), (x2, y1 + corner), color, 6)

    cv2.line(frame, (x1, y2), (x1 + corner, y2), color, 6)
    cv2.line(frame, (x1, y2), (x1, y2 - corner), color, 6)

    cv2.line(frame, (x2, y2), (x2 - corner, y2), color, 6)
    cv2.line(frame, (x2, y2), (x2, y2 - corner), color, 6)

    label = "AI focus"

    label_y = y1 - 10
    if label_y < 24:
        label_y = min(frame_h - 8, y2 + 26)

    cv2.putText(
        frame,
        label,
        (x1, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        3,
        cv2.LINE_AA,
    )


def overlay_attention_on_frame(
    frame,
    roi_bbox=None,
    crop_heatmap=None,
    crop_boxes=None,
    alpha: float = 0.28,
):
    frame_h, frame_w = frame.shape[:2]
    crop_boxes = crop_boxes or []

    if roi_bbox is None:
        x1, y1, x2, y2 = 0, 0, frame_w, frame_h
        frame_boxes = crop_boxes
    else:
        x1, y1, x2, y2 = _clip_box(roi_bbox, frame_w, frame_h)

        frame_boxes = []
        for box in crop_boxes:
            normalized = _normalize_box(box)
            if normalized is None:
                continue

            bx1, by1, bx2, by2 = normalized

            frame_boxes.append(
                (
                    bx1 + x1,
                    by1 + y1,
                    bx2 + x1,
                    by2 + y1,
                )
            )

    if crop_heatmap is not None and x2 > x1 and y2 > y1:
        roi = frame[y1:y2, x1:x2]

        if roi.size != 0:
            heatmap = cv2.resize(crop_heatmap, (roi.shape[1], roi.shape[0]))
            heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
            heatmap = np.clip(heatmap, 0.0, 1.0)

            colored = cv2.applyColorMap(
                np.uint8(255 * heatmap),
                cv2.COLORMAP_JET,
            )

            mixed = cv2.addWeighted(
                roi,
                1 - alpha,
                colored,
                alpha,
                0,
            )

            frame[y1:y2, x1:x2] = mixed

    for box in frame_boxes:
        normalized = _normalize_box(box)

        if normalized is None:
            continue

        _draw_focus_box(
            frame,
            normalized,
            color=(0, 255, 255),
        )

    return frame

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf

Box = Tuple[int, int, int, int]


@dataclass
class AttentionResult:
    heatmap: np.ndarray
    boxes: List[Box]


def _find_last_feature_layer(model: tf.keras.Model) -> str:
    """Возвращает слой, пригодный для Grad-CAM."""
    candidates = []

    for layer in model.layers:
        try:
            shape = layer.output_shape
        except Exception:
            continue

        if isinstance(shape, tuple) and len(shape) == 4:
            candidates.append(layer.name)

    if candidates:
        return candidates[-1]

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer.name

    raise ValueError("Не найден сверточный/feature-слой для построения карты внимания")


def make_gradcam_heatmap(
    model: tf.keras.Model,
    image_batch: np.ndarray,
    class_index: int | None = None,
    feature_layer_name: str | None = None,
) -> np.ndarray:
    """Строит Grad-CAM карту внимания для одного изображения."""
    feature_layer_name = feature_layer_name or _find_last_feature_layer(model)
    feature_layer = model.get_layer(feature_layer_name)

    grad_model = tf.keras.Model(
        model.inputs,
        [feature_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        feature_maps, predictions = grad_model(image_batch, training=False)

        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))

        target = predictions[:, class_index]

    grads = tape.gradient(target, feature_maps)

    if grads is None:
        raise RuntimeError("Grad-CAM не смог вычислить градиенты для выбранного слоя")

    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(feature_maps[0] * weights[0], axis=-1)
    cam = tf.maximum(cam, 0)

    max_value = tf.reduce_max(cam)

    if float(max_value) <= 1e-8:
        cam = tf.zeros_like(cam)
    else:
        cam = cam / (max_value + 1e-8)

    return cam.numpy().astype("float32")


def _clip_box(box: Box, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box

    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(x1 + 1, min(int(x2), width))
    y2 = max(y1 + 1, min(int(y2), height))

    return x1, y1, x2, y2


def _iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    inter = iw * ih

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))

    return inter / float(area_a + area_b - inter + 1e-8)


def _nms(boxes: List[Box], scores: List[float], max_boxes: int = 5, iou_threshold: float = 0.35) -> List[Box]:
    if not boxes:
        return []

    order = np.argsort(scores)[::-1]
    selected: List[Box] = []

    for idx in order:
        candidate = boxes[int(idx)]

        if all(_iou(candidate, existing) < iou_threshold for existing in selected):
            selected.append(candidate)

        if len(selected) >= max_boxes:
            break

    return selected


def _fallback_topk_boxes(
    heatmap_resized: np.ndarray,
    width: int,
    height: int,
    max_boxes: int = 5,
) -> List[Box]:
    """Возвращает top-k зон даже тогда, когда threshold не дал контуров.

    Это нужно для демонстрации дипломного проекта: на видео должны быть видны
    небольшие прямоугольники областей внимания нейросети, даже если Grad-CAM
    получился слабым или размытым.
    """
    boxes: List[Box] = []
    scores: List[float] = []

    grid_cols = 7
    grid_rows = 5

    cell_w = max(20, width // grid_cols)
    cell_h = max(20, height // grid_rows)

    for row in range(grid_rows):
        for col in range(grid_cols):
            x1 = col * cell_w
            y1 = row * cell_h
            x2 = width if col == grid_cols - 1 else (col + 1) * cell_w
            y2 = height if row == grid_rows - 1 else (row + 1) * cell_h

            region = heatmap_resized[y1:y2, x1:x2]

            if region.size == 0:
                continue

            score = float(np.mean(region) + 0.5 * np.max(region))

            if score <= 0.03:
                continue

            pad_x = int(cell_w * 0.10)
            pad_y = int(cell_h * 0.10)

            box = _clip_box(
                (
                    x1 + pad_x,
                    y1 + pad_y,
                    x2 - pad_x,
                    y2 - pad_y,
                ),
                width,
                height,
            )

            boxes.append(box)
            scores.append(score)

    return _nms(boxes, scores, max_boxes=max_boxes, iou_threshold=0.25)


def heatmap_to_boxes(
    heatmap: np.ndarray,
    image_size: Tuple[int, int],
    threshold: float = 0.25,
    min_area_ratio: float = 0.001,
    max_boxes: int = 5,
) -> List[Box]:
    """Преобразует Grad-CAM карту в несколько прямоугольников внимания.

    Возвращаемый формат:
        (x1, y1, x2, y2)

    Если обычный поиск контуров ничего не нашел, используется fallback:
    выбираются несколько наиболее активных зон по сетке.
    """
    width, height = image_size

    if heatmap is None:
        return []

    resized = cv2.resize(heatmap, (width, height))
    resized = np.nan_to_num(resized, nan=0.0, posinf=0.0, neginf=0.0)
    resized = np.clip(resized, 0.0, 1.0)

    mask = (resized >= threshold).astype("uint8") * 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    min_area = width * height * min_area_ratio

    boxes: List[Box] = []
    scores: List[float] = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w < 12 or h < 12:
            continue

        x1, y1, x2, y2 = _clip_box((x, y, x + w, y + h), width, height)

        region = resized[y1:y2, x1:x2]

        if region.size == 0:
            continue

        score = float(np.mean(region) + np.max(region))

        boxes.append((x1, y1, x2, y2))
        scores.append(score)

    selected = _nms(
        boxes,
        scores,
        max_boxes=max_boxes,
        iou_threshold=0.35,
    )

    if selected:
        return selected

    return _fallback_topk_boxes(
        resized,
        width=width,
        height=height,
        max_boxes=max_boxes,
    )


def build_attention(
    model: tf.keras.Model,
    image_batch: np.ndarray,
    image_size: Tuple[int, int],
    class_index: int | None = None,
    threshold: float = 0.25,
) -> AttentionResult:
    heatmap = make_gradcam_heatmap(
        model,
        image_batch,
        class_index=class_index,
    )

    boxes = heatmap_to_boxes(
        heatmap,
        image_size=image_size,
        threshold=threshold,
        min_area_ratio=0.001,
        max_boxes=5,
    )

    return AttentionResult(
        heatmap=heatmap,
        boxes=boxes,
    )


def draw_attention_on_crop(
    crop_bgr: np.ndarray,
    heatmap: np.ndarray,
    boxes: List[Box],
    alpha: float = 0.32,
) -> np.ndarray:
    """Накладывает Grad-CAM и прямоугольники на crop."""
    h, w = crop_bgr.shape[:2]

    resized = cv2.resize(heatmap, (w, h))
    colored = cv2.applyColorMap(
        np.uint8(255 * resized),
        cv2.COLORMAP_JET,
    )

    out = cv2.addWeighted(
        crop_bgr,
        1 - alpha,
        colored,
        alpha,
        0,
    )

    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(
            out,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2,
        )

        cv2.putText(
            out,
            "AI focus",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return out


def map_crop_boxes_to_frame(boxes: List[Box], face_bbox: Box) -> List[Box]:
    """Переносит координаты crop/ROI на полный кадр."""
    x0, y0, _, _ = face_bbox

    return [
        (
            int(x1 + x0),
            int(y1 + y0),
            int(x2 + x0),
            int(y2 + y0),
        )
        for x1, y1, x2, y2 in boxes
    ]
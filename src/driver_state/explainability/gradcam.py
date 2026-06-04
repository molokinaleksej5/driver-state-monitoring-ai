from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf


def make_gradcam_heatmap(model: tf.keras.Model, image_batch: np.ndarray, last_conv_layer_name: str | None = None) -> np.ndarray:
    if last_conv_layer_name is None:
        conv_layers = [l.name for l in model.layers if len(getattr(l, "output_shape", [])) == 4]
        last_conv_layer_name = conv_layers[-1] if conv_layers else None
    if last_conv_layer_name is None:
        raise ValueError("Не удалось определить сверточный слой для Grad-CAM")
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def save_gradcam_overlay(bgr_image, heatmap: np.ndarray, output_path: str | Path, alpha: float = 0.40) -> None:
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap = cv2.resize(heatmap, (bgr_image.shape[1], bgr_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(bgr_image, 1 - alpha, colored, alpha, 0)
    cv2.imwrite(str(output_path), overlay)

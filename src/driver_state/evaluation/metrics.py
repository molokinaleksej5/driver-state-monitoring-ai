from __future__ import annotations

from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from driver_state.config import AppConfig
from driver_state.data.datasets import make_image_dataset
from driver_state.utils.io import save_json
from driver_state.visualization.plots import plot_confusion_matrix


def evaluate_frame_model(model_path: str | Path, cfg: AppConfig, split: str = "test") -> dict:
    ds_dir = cfg.path(f"{split}_dir")
    ds = make_image_dataset(ds_dir, cfg, shuffle=False)
    model = tf.keras.models.load_model(model_path)
    y_true, y_prob = [], []
    for x, y in ds:
        p = model.predict(x, verbose=0)
        y_prob.append(p)
        y_true.append(y.numpy())
    y_true = np.vstack(y_true)
    y_prob = np.vstack(y_prob)
    true_idx = np.argmax(y_true, axis=1)
    pred_idx = np.argmax(y_prob, axis=1)
    report = classification_report(true_idx, pred_idx, target_names=cfg.class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(true_idx, pred_idx).tolist()
    try:
        auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
    except ValueError:
        auc = None
    result = {"split": split, "classification_report": report, "confusion_matrix": cm, "roc_auc_ovr": auc}
    save_json(result, cfg.path("reports_dir") / f"evaluation_{split}.json")
    plot_confusion_matrix(cm, cfg.class_names, cfg.path("figures_dir") / f"confusion_matrix_{split}.png")
    return result

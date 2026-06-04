from __future__ import annotations

from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from driver_state.config import AppConfig, ensure_project_dirs
from driver_state.data.datasets import create_frame_datasets
from driver_state.models.frame_cnn import build_frame_model, unfreeze_for_finetuning
from driver_state.utils.io import save_json
from driver_state.visualization.plots import plot_history


def _class_weights_from_dir(train_dir: Path, class_names: list[str]) -> dict[int, float]:
    labels = []
    for idx, name in enumerate(class_names):
        labels.extend([idx] * len(list((train_dir / name).glob("*"))))
    if not labels:
        return {}
    weights = compute_class_weight("balanced", classes=np.arange(len(class_names)), y=np.array(labels))
    return {i: float(w) for i, w in enumerate(weights)}


def train_frame_model(cfg: AppConfig) -> Path:
    ensure_project_dirs(cfg)
    train_ds, val_ds, _ = create_frame_datasets(cfg)
    model = build_frame_model(cfg)
    model_dir = cfg.path("models_dir")
    best_path = model_dir / "frame_model_best.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(best_path), save_best_only=True, monitor="val_accuracy", mode="max"),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2),
    ]
    class_weight = _class_weights_from_dir(cfg.path("train_dir"), cfg.class_names) if cfg.training.get("class_weighting", True) else None
    hist1 = model.fit(train_ds, validation_data=val_ds, epochs=int(cfg.training["frame_epochs_stage1"]), callbacks=callbacks, class_weight=class_weight)
    plot_history(hist1.history, cfg.path("figures_dir") / "frame_stage1_history.png")
    model = unfreeze_for_finetuning(model, cfg)
    hist2 = model.fit(train_ds, validation_data=val_ds, epochs=int(cfg.training["frame_epochs_stage2"]), callbacks=callbacks, class_weight=class_weight)
    plot_history(hist2.history, cfg.path("figures_dir") / "frame_stage2_history.png")
    final_path = model_dir / "frame_model_final.keras"
    model.save(str(final_path))
    save_json({"class_names": cfg.class_names, "best_model": str(best_path), "final_model": str(final_path)}, model_dir / "frame_model_meta.json")
    return best_path

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models
from driver_state.config import AppConfig


def build_frame_model(cfg: AppConfig) -> tf.keras.Model:
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.07),
        layers.RandomZoom(0.12),
        layers.RandomContrast(0.15),
        layers.RandomBrightness(0.10),
    ], name="augmentation")

    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*cfg.image_size, 3),
    )
    base.trainable = False

    inputs = layers.Input(shape=(*cfg.image_size, 3), name="frame")
    x = augmentation(inputs)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(float(cfg.training["dropout"]))(x)
    x = layers.Dense(256, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(cfg.num_classes, activation="softmax", name="state_output")(x)
    model = models.Model(inputs, outputs, name="EfficientNetB0_driver_state")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(float(cfg.training["learning_rate"])),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc", multi_label=True)],
    )
    return model


def unfreeze_for_finetuning(model: tf.keras.Model, cfg: AppConfig, fine_tune_at: int = 180) -> tf.keras.Model:
    base = next((layer for layer in model.layers if "efficientnet" in layer.name.lower()), None)
    if base is None:
        return model
    base.trainable = True
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(float(cfg.training["fine_tune_learning_rate"])),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc", multi_label=True)],
    )
    return model


def build_feature_extractor(frame_model: tf.keras.Model) -> tf.keras.Model:
    return tf.keras.Model(frame_model.input, frame_model.get_layer("dense_features").output, name="frame_feature_extractor")

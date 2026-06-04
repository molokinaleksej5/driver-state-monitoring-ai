from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models
from driver_state.config import AppConfig


def build_temporal_model(cfg: AppConfig, feature_dim: int = 256) -> tf.keras.Model:
    inputs = layers.Input(shape=(cfg.sequence_length, feature_dim), name="frame_features_sequence")
    x = layers.Masking(mask_value=0.0)(inputs)
    x = layers.Bidirectional(layers.GRU(128, return_sequences=True))(x)
    x = layers.Attention(name="temporal_attention")([x, x])
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(128, activation="relu")(x)
    state = layers.Dense(cfg.num_classes, activation="softmax", name="sequence_state")(x)
    risk = layers.Dense(1, activation="sigmoid", name="risk_score")(x)
    model = models.Model(inputs, [state, risk], name="CNN_GRU_driver_risk")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(float(cfg.training["learning_rate"])),
        loss={"sequence_state": "categorical_crossentropy", "risk_score": "binary_crossentropy"},
        loss_weights={"sequence_state": 1.0, "risk_score": 0.45},
        metrics={"sequence_state": ["accuracy"], "risk_score": [tf.keras.metrics.AUC(name="risk_auc")]},
    )
    return model

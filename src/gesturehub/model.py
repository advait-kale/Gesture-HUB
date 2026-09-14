"""Small 1D-CNN for IMU gesture classification.

Kept deliberately tiny (~a few thousand params) so it quantizes to int8 and
runs under TFLite Micro on an ESP32.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn(input_len: int, n_channels: int, n_classes: int) -> tf.keras.Model:
    inp = layers.Input(shape=(input_len, n_channels), name="imu")
    x = layers.Conv1D(16, 5, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(32, 5, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(n_classes, activation="softmax", name="probs")(x)

    return models.Model(inp, out, name="gesture_cnn")

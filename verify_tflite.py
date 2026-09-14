"""Sanity-check the exported int8 TFLite against the float Keras model.

Rebuilds the same train/test split as train.py, runs both models on the test
set, and reports agreement + accuracy. Catches quantization drift before flashing.
"""
from __future__ import annotations

import pathlib

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.gesturehub import data

SEED = 42
OUT = pathlib.Path("out")


def main():
    X, y, class_names = data.load_dataset("dataset/dataset", target_len=128)

    # same splits as train.py to recover the exact test set + train stats
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    X_tr, _, _, _ = train_test_split(
        X_tmp, y_tmp, test_size=0.2, stratify=y_tmp, random_state=SEED
    )
    mean, std = data.channel_stats(X_tr)
    X_test_s = data.standardize(X_test, mean, std)

    # float keras
    keras = tf.keras.models.load_model(OUT / "gesture_model.keras")
    keras_pred = keras.predict(X_test_s, verbose=0).argmax(axis=1)

    # int8 tflite
    interp = tf.lite.Interpreter(model_path=str(OUT / "gesture_model.tflite"))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    outp = interp.get_output_details()[0]
    scale, zp = inp["quantization"]

    tfl_pred = []
    for i in range(len(X_test_s)):
        q = np.round(X_test_s[i] / scale + zp).astype(np.int8)
        interp.set_tensor(inp["index"], q[None, ...])
        interp.invoke()
        tfl_pred.append(int(interp.get_tensor(outp["index"])[0].argmax()))
    tfl_pred = np.array(tfl_pred)

    keras_acc = (keras_pred == y_test).mean()
    tfl_acc = (tfl_pred == y_test).mean()
    agree = (keras_pred == tfl_pred).mean()
    print(f"test samples:      {len(y_test)}")
    print(f"keras accuracy:    {keras_acc:.3f}")
    print(f"tflite accuracy:   {tfl_acc:.3f}")
    print(f"keras/tflite agree:{agree:.3f}")
    if agree < 1.0:
        mism = np.where(keras_pred != tfl_pred)[0]
        for i in mism:
            print(f"  mismatch idx {i}: true={class_names[y_test[i]]} "
                  f"keras={class_names[keras_pred[i]]} tflite={class_names[tfl_pred[i]]}")


if __name__ == "__main__":
    main()

"""Load the GesturePod IMU dataset and turn variable-length CSV gestures into
fixed-length [T, 6] tensors.

Each CSV is one gesture instance with columns:
    rel_t_ms, ax, ay, az, gx, gy, gz
Accel in g, gyro in deg/s, sampled ~200 Hz.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]
N_CHANNELS = len(CHANNELS)


def find_classes(dataset_dir: pathlib.Path) -> list[str]:
    """Sorted list of gesture class names (subdirectory names)."""
    return sorted(p.name for p in dataset_dir.iterdir() if p.is_dir())


def resample_to_length(x: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resample a [n, C] signal to [target_len, C].

    Resampling (not padding) keeps the whole gesture shape regardless of how
    long the recording was, which matters here since files run ~330-500 rows.
    """
    n = x.shape[0]
    if n == target_len:
        return x.astype(np.float32)
    src = np.linspace(0.0, 1.0, num=n)
    dst = np.linspace(0.0, 1.0, num=target_len)
    out = np.empty((target_len, x.shape[1]), dtype=np.float32)
    for c in range(x.shape[1]):
        out[:, c] = np.interp(dst, src, x[:, c])
    return out


def load_dataset(dataset_dir: str | pathlib.Path, target_len: int = 128):
    """Load every CSV into X:[N, target_len, 6], y:[N] int labels.

    Returns (X, y, class_names).
    """
    dataset_dir = pathlib.Path(dataset_dir)
    class_names = find_classes(dataset_dir)
    label_of = {c: i for i, c in enumerate(class_names)}

    xs: list[np.ndarray] = []
    ys: list[int] = []
    for cls in class_names:
        for csv in sorted((dataset_dir / cls).glob("*.csv")):
            df = pd.read_csv(csv)
            sig = df[CHANNELS].to_numpy(dtype=np.float32)
            if sig.shape[0] < 2:
                continue
            xs.append(resample_to_length(sig, target_len))
            ys.append(label_of[cls])

    X = np.stack(xs, axis=0)
    y = np.asarray(ys, dtype=np.int64)
    return X, y, class_names


def channel_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over a training set of shape [N, T, C]."""
    mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
    std = X.reshape(-1, X.shape[-1]).std(axis=0)
    std[std < 1e-6] = 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)

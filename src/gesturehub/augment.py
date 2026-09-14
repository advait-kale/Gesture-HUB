"""Data augmentation for IMU gesture windows.

122 raw samples is far too few for a CNN, so we synthesize plausible variants.
All ops act on a single [T, 6] window (channels 0-2 accel, 3-5 gyro) and model
real-world variation: sensor noise, mounting-angle differences, speed/force
differences between people.
"""
from __future__ import annotations

import numpy as np

ACC = slice(0, 3)
GYR = slice(3, 6)


def jitter(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Additive Gaussian sensor noise (sigma is a fraction of channel std)."""
    scale = x.std(axis=0, keepdims=True) * sigma
    return x + rng.normal(0.0, 1.0, size=x.shape).astype(np.float32) * scale


def scaling(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Random per-channel gain ~ N(1, sigma) (force / amplitude differences)."""
    factor = rng.normal(1.0, sigma, size=(1, x.shape[1])).astype(np.float32)
    return x * factor


def time_warp(x: np.ndarray, sigma: float, n_knots: int, rng: np.random.Generator) -> np.ndarray:
    """Smoothly speed up / slow down parts of the gesture (speed differences)."""
    t = x.shape[0]
    knot_x = np.linspace(0, t - 1, num=n_knots)
    knot_y = knot_x + rng.normal(0.0, sigma * t / n_knots, size=n_knots)
    knot_y = np.sort(knot_y)
    knot_y = (knot_y - knot_y[0]) / (knot_y[-1] - knot_y[0]) * (t - 1)
    warped_idx = np.interp(np.arange(t), knot_x, knot_y)
    out = np.empty_like(x)
    grid = np.arange(t)
    for c in range(x.shape[1]):
        out[:, c] = np.interp(warped_idx, grid, x[:, c])
    return out


def magnitude_warp(x: np.ndarray, sigma: float, n_knots: int, rng: np.random.Generator) -> np.ndarray:
    """Smoothly varying gain along time (slow amplitude drift)."""
    t = x.shape[0]
    knot_x = np.linspace(0, t - 1, num=n_knots)
    knot_y = rng.normal(1.0, sigma, size=(n_knots, x.shape[1]))
    curve = np.empty_like(x)
    grid = np.arange(t)
    for c in range(x.shape[1]):
        curve[:, c] = np.interp(grid, knot_x, knot_y[:, c])
    return x * curve.astype(np.float32)


def _rot_matrix(max_deg: float, rng: np.random.Generator) -> np.ndarray:
    """Small random 3D rotation (mounting-angle variation)."""
    ang = np.deg2rad(rng.uniform(-max_deg, max_deg, size=3))
    cx, cy, cz = np.cos(ang)
    sx, sy, sz = np.sin(ang)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return (rz @ ry @ rx).astype(np.float32)


def rotate(x: np.ndarray, max_deg: float, rng: np.random.Generator) -> np.ndarray:
    """Rotate accel and gyro triads by the same random small rotation."""
    r = _rot_matrix(max_deg, rng)
    out = x.copy()
    out[:, ACC] = x[:, ACC] @ r.T
    out[:, GYR] = x[:, GYR] @ r.T
    return out


def augment_once(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply the full random augmentation stack to one window."""
    x = rotate(x, max_deg=15.0, rng=rng)
    x = time_warp(x, sigma=0.2, n_knots=4, rng=rng)
    x = magnitude_warp(x, sigma=0.15, n_knots=4, rng=rng)
    x = scaling(x, sigma=0.1, rng=rng)
    x = jitter(x, sigma=0.05, rng=rng)
    return x.astype(np.float32)


def expand(X: np.ndarray, y: np.ndarray, n_aug: int, seed: int = 0):
    """Return X,y with n_aug augmented copies appended per original window.

    The originals are kept as-is; copies are appended. Feed only the TRAIN
    split here (never val/test).
    """
    rng = np.random.default_rng(seed)
    aug_x = [X]
    aug_y = [y]
    for _ in range(n_aug):
        batch = np.stack([augment_once(X[i], rng) for i in range(len(X))], axis=0)
        aug_x.append(batch)
        aug_y.append(y.copy())
    return np.concatenate(aug_x, axis=0), np.concatenate(aug_y, axis=0)

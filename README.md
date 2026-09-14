# Gesture-HUB

IMU gesture recognition, GesturePod-style (see `CHI19_GesturePod.pdf`). A small
1D-CNN classifies hand gestures from a 6-axis IMU (accel + gyro), then exports
to **int8 TFLite** for **TFLite Micro on an ESP32**.

## Dataset

`dataset/dataset/<class>/*.csv`, one gesture instance per file. Columns:

```
rel_t_ms, ax, ay, az, gx, gy, gz
```

Accel in g, gyro in deg/s, sampled ~200 Hz (~5 ms step). Five classes:
`double_swipe, double_tap, left_twist, right_twist, twirl` (122 files total;
`double_tap` is over-represented — handled with class weights).

## Approach

- Each variable-length recording is **resampled to 128 steps × 6 channels**, so
  the whole gesture is captured regardless of how long it took.
- Only 122 raw samples, so training uses **heavy augmentation** (3D rotation,
  time-warp, magnitude-warp, scaling, jitter) — see `src/gesturehub/augment.py`.
- Per-channel standardization is fit on the train split only.
- Small Conv1D net (~few k params) → quantizes cleanly to int8.
- Honest accuracy via **stratified 5-fold CV**; a final model is trained on a
  train/val split and scored on a held-out test split.

## Setup

```bash
uv sync          # Python 3.12, TensorFlow 2.19, sklearn, pandas
```

## Train + export

```bash
.venv/Scripts/python.exe train.py                 # full run (CV + export)
.venv/Scripts/python.exe train.py --skip-cv        # skip CV, faster
```

Outputs land in `out/`:

| file | use |
|------|-----|
| `gesture_model.keras` | full-precision model |
| `gesture_model.tflite` | int8 quantized model (~24 KB) |
| `gesture_model_data.cc/.h` | the tflite bytes as a C array |
| `gesture_params.h` | input len, labels, per-channel mean/std, input quant |

## Deploy to ESP32

1. Copy `out/gesture_model_data.cc`, `.h`, and `gesture_params.h` into
   `esp32/`.
2. Install a TFLite Micro Arduino library (e.g. `esp-tflite-micro`).
3. In `esp32/gesture_infer.ino`, implement `capture_gesture()` to read your IMU
   and detect gesture start/stop. The sketch already does resample → normalize
   → quantize → invoke → argmax using the exported constants.

## Layout

```
src/gesturehub/
  data.py      # load CSVs, resample, standardize
  augment.py   # IMU augmentations
  model.py     # 1D-CNN
  export.py    # int8 TFLite + C-source export
train.py       # orchestration: CV, final train, eval, export
esp32/         # on-device inference scaffold
```

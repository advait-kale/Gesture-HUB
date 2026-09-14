"""Export a trained Keras model to int8 TFLite and C sources for ESP32.

Produces, in the output dir:
    gesture_model.tflite      - int8 quantized model
    gesture_model_data.cc/.h  - the .tflite bytes as a C array (TFLite Micro)
    gesture_params.h          - input len, channels, labels, per-channel
                                normalization constants (mean/std)
"""
from __future__ import annotations

import pathlib

import numpy as np
import tensorflow as tf


def to_int8_tflite(model: tf.keras.Model, rep_data: np.ndarray) -> bytes:
    """Full-integer int8 quantization. rep_data: standardized [N, T, C] float32."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def rep_gen():
        for i in range(len(rep_data)):
            yield [rep_data[i : i + 1].astype(np.float32)]

    converter.representative_dataset = rep_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def _c_array(name: str, data: bytes) -> str:
    lines = [f"const unsigned char {name}[] = {{"]
    for i in range(0, len(data), 12):
        chunk = ", ".join(f"0x{b:02x}" for b in data[i : i + 12])
        lines.append(f"  {chunk},")
    lines.append("};")
    lines.append(f"const unsigned int {name}_len = {len(data)};")
    return "\n".join(lines)


def write_c_model(tflite_bytes: bytes, out_dir: pathlib.Path, name: str = "gesture_model_data"):
    (out_dir / f"{name}.cc").write_text(
        f'#include "{name}.h"\n\nalignas(16) {_c_array(name, tflite_bytes)}\n'
    )
    (out_dir / f"{name}.h").write_text(
        f"#pragma once\n\nextern const unsigned char {name}[];\n"
        f"extern const unsigned int {name}_len;\n"
    )


def write_params_header(
    out_dir: pathlib.Path,
    input_len: int,
    n_channels: int,
    class_names: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    in_scale: float,
    in_zero_point: int,
):
    labels = ",\n  ".join(f'"{c}"' for c in class_names)
    mean_c = ", ".join(f"{v:.8f}f" for v in mean)
    std_c = ", ".join(f"{v:.8f}f" for v in std)
    text = f"""#pragma once
// Auto-generated. Preprocessing constants for on-device inference.
//
// On the ESP32: resample the captured gesture to GESTURE_INPUT_LEN samples,
// then for each channel c compute:
//     std_val   = (raw - GESTURE_MEAN[c]) / GESTURE_STD[c];
//     quantized = round(std_val / GESTURE_IN_SCALE) + GESTURE_IN_ZERO_POINT;
// clamp to [-128, 127] and feed as the int8 model input.

#define GESTURE_INPUT_LEN {input_len}
#define GESTURE_N_CHANNELS {n_channels}
#define GESTURE_N_CLASSES {len(class_names)}

// Channel order: ax, ay, az, gx, gy, gz
static const float GESTURE_MEAN[GESTURE_N_CHANNELS] = {{ {mean_c} }};
static const float GESTURE_STD[GESTURE_N_CHANNELS]  = {{ {std_c} }};

// Input quantization (float -> int8)
#define GESTURE_IN_SCALE {in_scale:.10f}f
#define GESTURE_IN_ZERO_POINT {in_zero_point}

static const char *GESTURE_LABELS[GESTURE_N_CLASSES] = {{
  {labels}
}};
"""
    (out_dir / "gesture_params.h").write_text(text)


def input_quant_params(tflite_bytes: bytes) -> tuple[float, int]:
    """Read the int8 input tensor's (scale, zero_point) from the tflite model."""
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    q = interp.get_input_details()[0]["quantization"]
    return float(q[0]), int(q[1])

#pragma once
// Auto-generated. Preprocessing constants for on-device inference.
//
// On the ESP32: resample the captured gesture to GESTURE_INPUT_LEN samples,
// then for each channel c compute:
//     std_val   = (raw - GESTURE_MEAN[c]) / GESTURE_STD[c];
//     quantized = round(std_val / GESTURE_IN_SCALE) + GESTURE_IN_ZERO_POINT;
// clamp to [-128, 127] and feed as the int8 model input.

#define GESTURE_INPUT_LEN 128
#define GESTURE_N_CHANNELS 6
#define GESTURE_N_CLASSES 2

// Channel order: ax, ay, az, gx, gy, gz
static const float GESTURE_MEAN[GESTURE_N_CHANNELS] = { -0.06759345f, -0.94199753f, 0.04703888f, 7.01835489f, 15.12718773f, -1.18540454f };
static const float GESTURE_STD[GESTURE_N_CHANNELS]  = { 0.32603613f, 0.32603794f, 0.33883062f, 21.27513504f, 78.60173035f, 59.71561432f };

// Input quantization (float -> int8)
#define GESTURE_IN_SCALE 0.0590782873f
#define GESTURE_IN_ZERO_POINT -26

static const char *GESTURE_LABELS[GESTURE_N_CLASSES] = {
  "double_tap",
  "swirl"
};

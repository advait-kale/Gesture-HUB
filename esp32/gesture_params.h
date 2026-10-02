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
#define GESTURE_N_CLASSES 5

// Channel order: ax, ay, az, gx, gy, gz
static const float GESTURE_MEAN[GESTURE_N_CHANNELS] = { -4663.23046875f, 2424.98510742f, 1330.80871582f, 30.14147949f, 578.72589111f, -0.81335133f };
static const float GESTURE_STD[GESTURE_N_CHANNELS]  = { 7238.12841797f, 5490.96484375f, 4197.38574219f, 9379.73437500f, 7308.43652344f, 5578.12011719f };

// Input quantization (float -> int8)
#define GESTURE_IN_SCALE 0.0546201803f
#define GESTURE_IN_ZERO_POINT -10

static const char *GESTURE_LABELS[GESTURE_N_CLASSES] = {
  "double_swipe",
  "double_tap",
  "left_twist",
  "right_twist",
  "twirl"
};

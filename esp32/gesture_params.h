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
static const float GESTURE_MEAN[GESTURE_N_CHANNELS] = { 0.01021926f, -0.70202267f, 0.44872695f, 5.37685585f, 20.41485786f, -0.15686098f };
static const float GESTURE_STD[GESTURE_N_CHANNELS]  = { 0.42875281f, 0.41946965f, 0.44097516f, 40.66690445f, 99.31622314f, 38.72165298f };

// Input quantization (float -> int8)
#define GESTURE_IN_SCALE 0.0474178977f
#define GESTURE_IN_ZERO_POINT 1

static const char *GESTURE_LABELS[GESTURE_N_CLASSES] = {
  "double_swipe",
  "double_tap",
  "left_twist",
  "right_twist",
  "twirl"
};

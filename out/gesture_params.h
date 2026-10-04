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
#define GESTURE_N_CLASSES 4

// Channel order: ax, ay, az, gx, gy, gz
static const float GESTURE_MEAN[GESTURE_N_CHANNELS] = { -0.02002133f, -0.96465075f, 0.04211516f, 5.29558659f, 11.06392860f, -1.40419054f };
static const float GESTURE_STD[GESTURE_N_CHANNELS]  = { 0.30722305f, 0.23311853f, 0.29244348f, 18.88832474f, 87.99509430f, 55.09752655f };

// Input quantization (float -> int8)
#define GESTURE_IN_SCALE 0.0756350532f
#define GESTURE_IN_ZERO_POINT -36

static const char *GESTURE_LABELS[GESTURE_N_CLASSES] = {
  "double_tap",
  "left_twist",
  "right_twist",
  "swirl"
};

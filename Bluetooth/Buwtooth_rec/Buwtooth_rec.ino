// ESP32 gesture inference (TFLite Micro) with Bluetooth Serial output.
//
// Runs the int8 CNN exported by train.py. You supply:
//   - IMU capture (MPU6050 etc.) in mpu6050_capture.ino
//   - gesture start/stop detection
// This file handles: resample -> normalize -> quantize -> invoke -> argmax,
// and prints results to both USB Serial and Bluetooth Serial.
//
// Board: DOIT ESP32 DEVKIT V1
// Partition scheme: Huge APP (3MB No OTA/1MB SPIFFS)

#include <math.h>
#include <stdarg.h>

#include "BluetoothSerial.h"

#include "gesture_model_data.h"
#include "gesture_params.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

BluetoothSerial SerialBT;

namespace {
const tflite::Model *g_model = nullptr;
tflite::MicroInterpreter *g_interp = nullptr;
TfLiteTensor *g_input = nullptr;TfLiteTensor *g_output = nullptr;

// Arena: 24KB model + small buffers. Bump if AllocateTensors() fails.
constexpr int kArenaSize = 40 * 1024;
uint8_t g_arena[kArenaSize];
}  // namespace

// ---- Defined in mpu6050_capture.ino ---------------------------------------
extern int capture_gesture(float raw[][GESTURE_N_CHANNELS], int max_rows);
extern void mpu_setup();
// ---------------------------------------------------------------------------

// Print to USB Serial and, if a client is connected, to Bluetooth too.
void logf(const char *fmt, ...) {
  char buf[128];
  va_list args;
  va_start(args, fmt);
  vsnprintf(buf, sizeof(buf), fmt, args);
  va_end(args);

  Serial.print(buf);
  if (SerialBT.hasClient()) {
    SerialBT.print(buf);
  }
}

// Linear resample raw[n][C] -> out[GESTURE_INPUT_LEN][C].
static void resample(const float *raw, int n, float *out) {
  for (int c = 0; c < GESTURE_N_CHANNELS; c++) {
    for (int t = 0; t < GESTURE_INPUT_LEN; t++) {
      float pos = (float)t * (n - 1) / (GESTURE_INPUT_LEN - 1);
      int i0 = (int)floorf(pos);
      int i1 = (i0 + 1 < n) ? i0 + 1 : i0;
      float frac = pos - i0;
      float a = raw[i0 * GESTURE_N_CHANNELS + c];
      float b = raw[i1 * GESTURE_N_CHANNELS + c];
      out[t * GESTURE_N_CHANNELS + c] = a + frac * (b - a);
    }
  }
}

void setup() {
  Serial.begin(115200);
  SerialBT.begin("ESP32-Gesture");  // Bluetooth device name

  tflite::InitializeTarget();
  mpu_setup();

  g_model = tflite::GetModel(gesture_model_data);
  if (g_model->version() != TFLITE_SCHEMA_VERSION) {
    logf("model schema mismatch\n");
    while (true) {}
  }

  static tflite::MicroMutableOpResolver<9> resolver;
  resolver.AddConv2D();
  resolver.AddFullyConnected();
  resolver.AddMaxPool2D();
  resolver.AddMean();      // GlobalAveragePooling1D
  resolver.AddRelu();
  resolver.AddReshape();
  resolver.AddSoftmax();
  resolver.AddQuantize();
  resolver.AddExpandDims();

  static tflite::MicroInterpreter interp(g_model, resolver, g_arena, kArenaSize);
  g_interp = &interp;
  if (g_interp->AllocateTensors() != kTfLiteOk) {
    logf("AllocateTensors failed\n");
    while (true) {}
  }
  g_input = g_interp->input(0);
  g_output = g_interp->output(0);
  logf("ready\n");
}

void loop() {
  static float raw[512][GESTURE_N_CHANNELS];
  static float resampled[GESTURE_INPUT_LEN * GESTURE_N_CHANNELS];

  int n = capture_gesture(raw, 512);
  if (n < 2) return;

  resample(&raw[0][0], n, resampled);

  // normalize + quantize into the int8 input tensor
  const float in_scale = GESTURE_IN_SCALE;
  const int in_zp = GESTURE_IN_ZERO_POINT;
  for (int t = 0; t < GESTURE_INPUT_LEN; t++) {
    for (int c = 0; c < GESTURE_N_CHANNELS; c++) {
      int idx = t * GESTURE_N_CHANNELS + c;
      float std_val = (resampled[idx] - GESTURE_MEAN[c]) / GESTURE_STD[c];
      int q = lroundf(std_val / in_scale) + in_zp;
      if (q < -128) q = -128;
      if (q > 127) q = 127;
      g_input->data.int8[idx] = (int8_t)q;
    }
  }

  if (g_interp->Invoke() != kTfLiteOk) {
    logf("invoke failed\n");
    return;
  }

  // dequantize output, argmax
  float out_scale = g_output->params.scale;
  int out_zp = g_output->params.zero_point;
  int best = 0;
  float best_p = -1.0f;
  for (int i = 0; i < GESTURE_N_CLASSES; i++) {
    float p = (g_output->data.int8[i] - out_zp) * out_scale;
    if (p > best_p) { best_p = p; best = i; }
  }

  // Confidence gate: the model has no "idle/unknown" class, so it always picks
  // one of the classes. Reject low-confidence picks (random motion, half-gestures).
  // Tune the threshold against real device data.
  const float kConfThreshold = 0.70f;
  if (best_p < kConfThreshold) {
    logf("(rejected: %s %.2f < %.2f)\n",
         GESTURE_LABELS[best], best_p, kConfThreshold);
    return;
  }
  logf("gesture: %s (%.2f)\n", GESTURE_LABELS[best], best_p);
}
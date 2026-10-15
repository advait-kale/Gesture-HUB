// ESP32 gesture inference with a Wi-Fi REST API.
// Board: ESP32 Dev Module; MPU6050 SDA=21, SCL=22, AD0=GND.

#include <math.h>
#include <WiFi.h>
#include <WebServer.h>

#include "gesture_model_data.h"
#include "gesture_params.h"
#include "wifi_config.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {
const tflite::Model *g_model = nullptr;
tflite::MicroInterpreter *g_interp = nullptr;
TfLiteTensor *g_input = nullptr;
TfLiteTensor *g_output = nullptr;

constexpr int kArenaSize = 40 * 1024;
uint8_t g_arena[kArenaSize];

WebServer g_server(80);
bool g_serverStarted = false;
bool g_modelReady = false;
bool g_hasPrediction = false;
bool g_predictionAccepted = false;
uint32_t g_eventId = 0;
uint32_t g_predictionTimeMs = 0;
float g_confidence = 0.0f;
int g_predictionClass = -1;
uint32_t g_lastWifiAttemptMs = 0;

constexpr float kConfidenceThreshold = 0.70f;
constexpr uint32_t kWifiRetryMs = 10000;

void sendJson(int statusCode, const String &body) {
  g_server.sendHeader("Access-Control-Allow-Origin", "*");
  g_server.sendHeader("Cache-Control", "no-store");
  g_server.send(statusCode, "application/json", body);
}

void handleRoot() {
  sendJson(200,
           "{\"name\":\"Gesture-HUB API\",\"endpoints\":["
           "\"GET /api/status\",\"GET /api/result\"]}");
}

void handleStatus() {
  String body = "{\"ready\":";
  body += g_modelReady ? "true" : "false";
  body += ",\"wifi_connected\":";
  const bool connected = WiFi.status() == WL_CONNECTED;
  body += connected ? "true" : "false";
  body += ",\"ip\":\"";
  if (connected) body += WiFi.localIP().toString();
  body += "\",\"event_id\":";
  body += String(g_eventId);
  body += "}";
  sendJson(200, body);
}

void handleLatestGesture() {
  if (!g_hasPrediction) {
    sendJson(200, "{\"available\":false,\"message\":\"No gesture detected yet\"}");
    return;
  }

  char body[192];
  snprintf(body, sizeof(body),
           "{\"available\":true,\"event_id\":%lu,\"gesture\":\"%s\","
           "\"confidence\":%.4f,\"accepted\":%s,\"timestamp_ms\":%lu}",
           static_cast<unsigned long>(g_eventId),
           GESTURE_LABELS[g_predictionClass], g_confidence,
           g_predictionAccepted ? "true" : "false",
           static_cast<unsigned long>(g_predictionTimeMs));
  sendJson(200, String(body));
}

void handleOptions() {
  g_server.sendHeader("Access-Control-Allow-Origin", "*");
  g_server.sendHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
  g_server.sendHeader("Access-Control-Allow-Headers", "Content-Type");
  g_server.send(204);
}

void beginApiServer() {
  g_server.on("/", HTTP_GET, handleRoot);
  g_server.on("/api/status", HTTP_GET, handleStatus);
  g_server.on("/api/result", HTTP_GET, handleLatestGesture);
  g_server.on("/api/gesture/latest", HTTP_GET, handleLatestGesture);
  g_server.on("/", HTTP_OPTIONS, handleOptions);
  g_server.on("/api/status", HTTP_OPTIONS, handleOptions);
  g_server.on("/api/result", HTTP_OPTIONS, handleOptions);
  g_server.on("/api/gesture/latest", HTTP_OPTIONS, handleOptions);
  g_server.onNotFound([]() {
    sendJson(404, "{\"error\":\"not_found\"}");
  });
  g_server.begin();
  g_serverStarted = true;
  Serial.print("API ready at http://");
  Serial.println(WiFi.localIP());
}

void connectWifi() {
  if (WiFi.status() == WL_CONNECTED) {
    if (!g_serverStarted) beginApiServer();
    return;
  }

  const uint32_t now = millis();
  if (now - g_lastWifiAttemptMs < kWifiRetryMs) return;
  g_lastWifiAttemptMs = now;

  if (strcmp(WIFI_SSID, "YOUR_WIFI_NAME") == 0) {
    Serial.println("Set WIFI_SSID and WIFI_PASSWORD in wifi_config.h");
    return;
  }

  Serial.printf("Connecting to Wi-Fi: %s\n", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
}

void resample(const float *raw, int n, float *out) {
  for (int c = 0; c < GESTURE_N_CHANNELS; c++) {
    for (int t = 0; t < GESTURE_INPUT_LEN; t++) {
      const float pos = static_cast<float>(t) * (n - 1) /
                        (GESTURE_INPUT_LEN - 1);
      const int i0 = static_cast<int>(floorf(pos));
      const int i1 = (i0 + 1 < n) ? i0 + 1 : i0;
      const float frac = pos - i0;
      const float a = raw[i0 * GESTURE_N_CHANNELS + c];
      const float b = raw[i1 * GESTURE_N_CHANNELS + c];
      out[t * GESTURE_N_CHANNELS + c] = a + frac * (b - a);
    }
  }
}

void recordPrediction() {
  const float outScale = g_output->params.scale;
  const int outZeroPoint = g_output->params.zero_point;
  int best = 0;
  float bestConfidence = -1.0f;
  for (int i = 0; i < GESTURE_N_CLASSES; i++) {
    const float confidence = (g_output->data.int8[i] - outZeroPoint) * outScale;
    if (confidence > bestConfidence) {
      bestConfidence = confidence;
      best = i;
    }
  }

  g_predictionClass = best;
  g_confidence = bestConfidence;
  g_predictionAccepted = bestConfidence >= kConfidenceThreshold;
  g_predictionTimeMs = millis();
  g_eventId++;
  g_hasPrediction = true;

  Serial.printf("gesture: %s (%.2f)%s\n", GESTURE_LABELS[best], bestConfidence,
                g_predictionAccepted ? "" : " [below threshold]");
}
}  // namespace

extern int capture_gesture(float raw[][GESTURE_N_CHANNELS], int max_rows);
extern void mpu_setup();

// Called by loop() and capture_gesture(), including while waiting for motion.
void service_wifi() {
  connectWifi();
  if (g_serverStarted && WiFi.status() == WL_CONNECTED) {
    g_server.handleClient();
  }
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("BOOT: ESP32 gesture Wi-Fi API starting");

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  g_lastWifiAttemptMs = millis();

  tflite::InitializeTarget();
  mpu_setup();

  g_model = tflite::GetModel(gesture_model_data);
  if (g_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("model schema mismatch");
    while (true) delay(1000);
  }

  static tflite::MicroMutableOpResolver<9> resolver;
  resolver.AddConv2D();
  resolver.AddFullyConnected();
  resolver.AddMaxPool2D();
  resolver.AddMean();
  resolver.AddRelu();
  resolver.AddReshape();
  resolver.AddSoftmax();
  resolver.AddQuantize();
  resolver.AddExpandDims();

  static tflite::MicroInterpreter interpreter(g_model, resolver, g_arena, kArenaSize);
  g_interp = &interpreter;
  if (g_interp->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors failed");
    while (true) delay(1000);
  }
  g_input = g_interp->input(0);
  g_output = g_interp->output(0);
  g_modelReady = true;

  Serial.println("Model ready; waiting for Wi-Fi and gestures");
  service_wifi();
}

void loop() {
  static float raw[512][GESTURE_N_CHANNELS];
  static float resampled[GESTURE_INPUT_LEN * GESTURE_N_CHANNELS];

  service_wifi();
  const int n = capture_gesture(raw, 512);
  if (n < 2) return;

  resample(&raw[0][0], n, resampled);
  for (int t = 0; t < GESTURE_INPUT_LEN; t++) {
    for (int c = 0; c < GESTURE_N_CHANNELS; c++) {
      const int index = t * GESTURE_N_CHANNELS + c;
      const float standardized =
          (resampled[index] - GESTURE_MEAN[c]) / GESTURE_STD[c];
      int quantized = lroundf(standardized / GESTURE_IN_SCALE) +
                      GESTURE_IN_ZERO_POINT;
      if (quantized < -128) quantized = -128;
      if (quantized > 127) quantized = 127;
      g_input->data.int8[index] = static_cast<int8_t>(quantized);
    }
  }

  if (g_interp->Invoke() != kTfLiteOk) {
    Serial.println("invoke failed");
    return;
  }
  recordPrediction();
}

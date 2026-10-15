// MPU6050 capture for gesture_infer.ino.
//
// Provides capture_gesture(): reads the IMU at ~200 Hz and returns one full
// gesture as rows of {ax,ay,az (g), gx,gy,gz (deg/s)} -- the SAME units the
// model was trained on. Getting units/ranges right matters more than anything
// else here; wrong scaling => wrong predictions.
//
// Wiring (ESP32 default I2C): SDA=GPIO21, SCL=GPIO22, MPU6050 AD0->GND (0x68).
// No external lib needed -- raw register reads over Wire.

#include <Wire.h>
#include <math.h>

#include "gesture_params.h"  // GESTURE_N_CHANNELS

// Keep the Bluetooth client status visible while this routine waits for motion.
extern void service_bluetooth();

namespace {
constexpr uint8_t MPU_ADDR = 0x68;
constexpr uint8_t REG_PWR_MGMT_1 = 0x6B;
constexpr uint8_t REG_SMPLRT_DIV = 0x19;
constexpr uint8_t REG_CONFIG = 0x1A;
constexpr uint8_t REG_GYRO_CONFIG = 0x1B;
constexpr uint8_t REG_ACCEL_CONFIG = 0x1C;
constexpr uint8_t REG_ACCEL_XOUT_H = 0x3B;

// Ranges must match the training data. Dataset accel ~1g at rest and gyro in
// deg/s, consistent with the sensitive default ranges below.
//   ACCEL_CONFIG 0x00 -> +/-2g   -> 16384 LSB/g
//   GYRO_CONFIG  0x00 -> +/-250  -> 131.0 LSB/(deg/s)
constexpr float ACC_LSB_PER_G = 16384.0f;
constexpr float GYR_LSB_PER_DPS = 131.0f;

// ~200 Hz loop (5 ms), matching the ~5 ms step in the CSVs.
constexpr uint32_t SAMPLE_US = 5000;

// Motion gate on |accel| deviation from 1g (rest).
constexpr float START_G = 0.15f;    // begin capture above this
constexpr float STOP_G = 0.08f;     // "quiet" below this
constexpr uint32_t QUIET_MS = 250;  // end after this much continuous quiet
constexpr int MIN_ROWS = 20;        // ignore blips
}  // namespace

static void mpu_write(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

// Call from setup() before inference. (setup() is in gesture_infer.ino; add a
// call to mpu_setup() there, or move Wire init as you prefer.)
void mpu_setup() {
  Wire.begin();          // default SDA=21, SCL=22 on most ESP32 boards
  Wire.setClock(400000);
  mpu_write(REG_PWR_MGMT_1, 0x00);    // wake
  mpu_write(REG_SMPLRT_DIV, 0x04);    // 1kHz/(1+4) = 200 Hz sample rate
  mpu_write(REG_CONFIG, 0x03);        // DLPF ~44 Hz
  mpu_write(REG_GYRO_CONFIG, 0x00);   // +/-250 deg/s
  mpu_write(REG_ACCEL_CONFIG, 0x00);  // +/-2g
  delay(50);
}

static void read_sample(float out[GESTURE_N_CHANNELS]) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(REG_ACCEL_XOUT_H);
  Wire.endTransmission(false);
  Wire.requestFrom(MPU_ADDR, (uint8_t)14);  // 6 accel, 2 temp, 6 gyro

  int16_t ax = (Wire.read() << 8) | Wire.read();
  int16_t ay = (Wire.read() << 8) | Wire.read();
  int16_t az = (Wire.read() << 8) | Wire.read();
  Wire.read(); Wire.read();  // skip temperature
  int16_t gx = (Wire.read() << 8) | Wire.read();
  int16_t gy = (Wire.read() << 8) | Wire.read();
  int16_t gz = (Wire.read() << 8) | Wire.read();

  out[0] = ax / ACC_LSB_PER_G;
  out[1] = ay / ACC_LSB_PER_G;
  out[2] = az / ACC_LSB_PER_G;
  out[3] = gx / GYR_LSB_PER_DPS;
  out[4] = gy / GYR_LSB_PER_DPS;
  out[5] = gz / GYR_LSB_PER_DPS;
}

// Blocks until a gesture is captured. Returns rows written to raw[][], or 0.
int capture_gesture(float raw[][GESTURE_N_CHANNELS], int max_rows) {
  float s[GESTURE_N_CHANNELS];
  uint32_t next = micros();

  // wait for motion to start
  while (true) {
    while ((int32_t)(micros() - next) < 0) {}
    next += SAMPLE_US;
    service_bluetooth();
    read_sample(s);
    float amag = sqrtf(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]);
    if (fabsf(amag - 1.0f) > START_G) break;
  }

  // record until quiet for QUIET_MS
  int n = 0;
  uint32_t quiet_ms = 0;
  while (n < max_rows) {
    while ((int32_t)(micros() - next) < 0) {}
    next += SAMPLE_US;
    service_bluetooth();
    read_sample(s);
    for (int c = 0; c < GESTURE_N_CHANNELS; c++) raw[n][c] = s[c];
    n++;

    float amag = sqrtf(s[0]*s[0] + s[1]*s[1] + s[2]*s[2]);
    if (fabsf(amag - 1.0f) < STOP_G) {
      quiet_ms += SAMPLE_US / 1000;
      if (quiet_ms >= QUIET_MS) break;
    } else {
      quiet_ms = 0;
    }
  }

  return (n >= MIN_ROWS) ? n : 0;
}

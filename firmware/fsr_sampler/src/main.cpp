// Force sensor sampler: reads the four jaw sensors at 500 Hz and streams binary frames over USB serial.
//
// Each sensor is one half of a voltage divider: sensor from 3V3 to its pin, a 10 kOhm resistor from the
// pin to GND. Pressing harder lowers the sensor's resistance, so the pin voltage rises.
//
// Frame, 18 bytes, little-endian:
//   A5 5A | seq u16 | t_us u32 | mV u16 x 4 (SD, SP, MD, MP) | CRC-16/CCITT-FALSE u16 over seq..mV
// wbc/reflex/fsr_link.py reads these. A frame is skipped, not queued, when the host isn't keeping up,
// so the host sees it as a gap in seq.
#include <Arduino.h>

// ADC1 pins only. SD/SP = static jaw distal/proximal, MD/MP = moving jaw distal/proximal.
static const uint8_t PINS[4] = {32, 33, 34, 35};
static const uint32_t PERIOD_US = 2000;  // 500 Hz
static const int OVERSAMPLE = 4;         // readings averaged per channel per frame
static const uint32_t BAUD = 921600;

uint8_t frame[18] = {0xA5, 0x5A};
uint16_t seq = 0;
uint32_t next_us;

static uint16_t crc16(const uint8_t *data, size_t n) {
  uint16_t crc = 0xFFFF;
  while (n--) {
    crc ^= (uint16_t)(*data++) << 8;
    for (int i = 0; i < 8; i++) crc = (crc & 0x8000) ? (crc << 1) ^ 0x1021 : crc << 1;
  }
  return crc;
}

static void put16(uint8_t *p, uint16_t v) {
  p[0] = v & 0xFF;
  p[1] = v >> 8;
}

static void put32(uint8_t *p, uint32_t v) {
  for (int i = 0; i < 4; i++) p[i] = (v >> (8 * i)) & 0xFF;
}

void setup() {
  Serial.setTxBufferSize(1024);
  Serial.begin(BAUD);
  analogReadResolution(12);
  for (uint8_t pin : PINS) analogSetPinAttenuation(pin, ADC_11db);  // the full 0-3.1 V range
  next_us = micros();
}

void loop() {
  uint32_t now = micros();
  if ((int32_t)(now - next_us) < 0) return;
  next_us += PERIOD_US;
  if ((int32_t)(now - next_us) > 0) next_us = now + PERIOD_US;  // fell behind: skip ahead, don't burst

  put16(frame + 2, seq++);
  put32(frame + 4, now);
  for (int ch = 0; ch < 4; ch++) {
    uint32_t sum = 0;
    for (int k = 0; k < OVERSAMPLE; k++) sum += analogReadMilliVolts(PINS[ch]);  // uses the chip's ADC calibration
    put16(frame + 8 + 2 * ch, sum / OVERSAMPLE);
  }
  put16(frame + 16, crc16(frame + 2, 14));
  if (Serial.availableForWrite() >= (int)sizeof(frame)) Serial.write(frame, sizeof(frame));
}

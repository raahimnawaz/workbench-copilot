// Demo board for Act 1: reads a force sensor and prints millivolts over serial, one line per reading.
// With the sensor's jumper on a different GPIO than SENSOR_PIN, the readings ignore every press, but the
// board still runs and prints, like the real bug. wbc/agent/dut.py reads these lines.
//
//   mv 1873              one reading, every SAMPLE_MS
//   # reading GPIO 15    every 50 readings, so a reader that joins late still learns the pin
#include <Arduino.h>

#include "pins.h"

unsigned long readings = 0;

void setup() {
  Serial.begin(115200);
  analogSetPinAttenuation(SENSOR_PIN, ADC_11db);  // the full 0-3.3 V range
}

void loop() {
  if (readings % 50 == 0) Serial.printf("# reading GPIO %d\n", SENSOR_PIN);
  Serial.printf("mv %u\n", (unsigned)analogReadMilliVolts(SENSOR_PIN));
  readings++;
  delay(SAMPLE_MS);
}

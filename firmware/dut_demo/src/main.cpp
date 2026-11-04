// Demo board for Act 1: lights an LED and keeps it on, saying so over serial.
#include <Arduino.h>

#include "pins.h"

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);
}

void loop() {
  Serial.println("led on");
  delay(REPORT_MS);
}

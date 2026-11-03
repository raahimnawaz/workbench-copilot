// Demo board for Act 1: blinks an LED and logs each blink over serial.
#include <Arduino.h>

#include "pins.h"

unsigned long blinks = 0;

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
}

void loop() {
  digitalWrite(LED_PIN, HIGH);
  delay(BLINK_MS);
  digitalWrite(LED_PIN, LOW);
  delay(BLINK_MS);
  Serial.printf("blink %lu\n", blinks++);
}

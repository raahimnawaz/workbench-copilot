// Demo board for Act 1: blinks an LED and logs each blink over serial.
// With the LED's jumper on the wrong GPIO the LED stays dark, but the board still runs and logs, like the real bug.
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

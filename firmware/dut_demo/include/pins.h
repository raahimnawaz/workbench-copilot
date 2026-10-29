// Demo board wiring: an LED from LED_PIN to GND, through a 330 ohm resistor.
// Pins are for a classic ESP32 DevKit. On an ESP32-S3, pick a pin that board exposes.
// The planted bug for Act 1: the LED's jumper goes to GPIO 4, three pins past GPIO 5, but this file says 5.
#pragma once

#define LED_PIN  5     // the LED's anode, through its jumper
#define BLINK_MS 250   // the LED is on this long, then off this long

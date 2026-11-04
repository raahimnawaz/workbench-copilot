// Demo board wiring: an LED from LED_PIN to GND, through a 330 ohm resistor.
// Pins are for a classic ESP32 DevKit. On an ESP32-S3, pick a pin that board exposes.
#pragma once

#define LED_PIN  5     // the LED's anode, through its jumper
#define REPORT_MS 500  // how often it says over serial that the LED is on

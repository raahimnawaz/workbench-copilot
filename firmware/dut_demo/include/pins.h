// Demo board wiring: SSD1306 128x64 OLED on SPI.
// Pins are for a classic ESP32 DevKit. On an ESP32-S3, pick pins that board exposes.
// The planted bug for Act 1: the CS jumper goes to GPIO 13, but this file says 5.
#pragma once

#define OLED_SCK  18   // D0 on the display
#define OLED_MOSI 23   // D1 on the display
#define OLED_CS   5    // chip select
#define OLED_DC   16   // data/command
#define OLED_RES  17   // reset

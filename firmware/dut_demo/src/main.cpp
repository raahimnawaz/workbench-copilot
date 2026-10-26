// Demo board for Act 1: draws a counter on an SPI OLED and logs each frame over serial.
// With the wrong CS pin the display stays dark, but setup still succeeds, just like the real bug.
#include <Arduino.h>
#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#include "pins.h"

Adafruit_SSD1306 display(128, 64, &SPI, OLED_DC, OLED_RES, OLED_CS);
unsigned long frame = 0;

void setup() {
  Serial.begin(115200);
  SPI.begin(OLED_SCK, -1, OLED_MOSI, OLED_CS);
  if (!display.begin(SSD1306_SWITCHCAPVCC)) {
    Serial.println("display buffer allocation failed");
    for (;;) delay(1000);
  }
  display.setTextColor(SSD1306_WHITE);
}

void loop() {
  display.clearDisplay();
  display.setTextSize(2);
  display.setCursor(0, 0);
  display.println("Workbench");
  display.setCursor(0, 32);
  display.print("frame ");
  display.println(frame);
  display.display();
  Serial.printf("frame %lu\n", frame++);
  delay(500);
}

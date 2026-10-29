// Demo board wiring: a force sensor from 3.3 V to SENSOR_PIN, with 10 kOhm from SENSOR_PIN to GND.
// Pressing the sensor lowers its resistance, so SENSOR_PIN's voltage rises from about 0 toward 3.3 V.
// The planted bug for Act 1: the sensor's jumper goes to GPIO 4, but this file says 15.
#pragma once

#define SENSOR_PIN 15    // the sensor's divider, through its jumper
#define SAMPLE_MS  20    // one reading every 20 ms, printed over serial

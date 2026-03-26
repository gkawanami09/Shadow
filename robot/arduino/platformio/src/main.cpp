#include <Arduino.h>

String buffer = "";

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
  Serial.begin(115200);
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '\n') {
      buffer.trim();

      if (buffer == "LED_ON") {
        digitalWrite(LED_BUILTIN, HIGH);
        Serial.println("OK:LED_ON");
      }
      else if (buffer == "LED_OFF") {
        digitalWrite(LED_BUILTIN, LOW);
        Serial.println("OK:LED_OFF");
      }
      else if (buffer == "PING") {
        Serial.println("PONG");
      }
      else {
        Serial.print("CMD:");
        Serial.println(buffer);
      }

      buffer = "";
    } else {
      buffer += c;
    }
  }
}
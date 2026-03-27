#include <Arduino.h>

// Ajuste os pinos conforme sua ligacao fisica.
// Driver TB6612 da frente (lado esquerdo = A, lado direito = B)
const int AIN1_F = 2;
const int AIN2_F = 3;
const int PWMA_F = 5;   // PWM
const int BIN1_F = 4;
const int BIN2_F = 7;
const int PWMB_F = 6;   // PWM

// Driver TB6612 de tras (lado esquerdo = A, lado direito = B)
const int AIN1_R = 8;
const int AIN2_R = 12;
const int PWMA_R = 9;   // PWM
const int BIN1_R = 13;
const int BIN2_R = 11;
const int PWMB_R = 10;  // PWM

// Standby (pode ligar os dois STBY juntos neste pino)
const int STBY = A0;

const int DEFAULT_SPEED = 180;
const unsigned long TURN_180_MS = 1200;

String buffer = "";
bool turn_active = false;
unsigned long turn_until = 0;
int turn_speed = DEFAULT_SPEED;

void setMotor(int in1, int in2, int pwm, int speed) {
  speed = constrain(speed, -255, 255);
  if (speed > 0) {
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
    analogWrite(pwm, speed);
  } else if (speed < 0) {
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
    analogWrite(pwm, -speed);
  } else {
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
    analogWrite(pwm, 0);
  }
}

void setLeft(int speed) {
  setMotor(AIN1_F, AIN2_F, PWMA_F, speed);
  setMotor(AIN1_R, AIN2_R, PWMA_R, speed);
}

void setRight(int speed) {
  setMotor(BIN1_F, BIN2_F, PWMB_F, speed);
  setMotor(BIN1_R, BIN2_R, PWMB_R, speed);
}

void stopAll() {
  setLeft(0);
  setRight(0);
}

void startTurn180(int speed) {
  turn_active = true;
  turn_speed = constrain(speed, 0, 255);
  turn_until = millis() + TURN_180_MS;
  // Giro no lugar (direita)
  setLeft(turn_speed);
  setRight(-turn_speed);
}

void applyCommand(char cmd, int speed) {
  speed = constrain(speed, 0, 255);
  turn_active = false;

  switch (cmd) {
    case 'F':
      setLeft(speed);
      setRight(speed);
      break;
    case 'B':
      setLeft(-speed);
      setRight(-speed);
      break;
    case 'L':
      setLeft(-speed);
      setRight(speed);
      break;
    case 'R':
      setLeft(speed);
      setRight(-speed);
      break;
    case 'S':
      stopAll();
      break;
    case 'U':
      startTurn180(speed);
      break;
    default:
      stopAll();
      break;
  }
}

void handleLine(String line) {
  line.trim();
  if (line.length() == 0) {
    return;
  }
  char cmd = toupper(line.charAt(0));
  int speed = DEFAULT_SPEED;
  int comma = line.indexOf(',');
  if (comma >= 0 && comma + 1 < line.length()) {
    speed = line.substring(comma + 1).toInt();
  }
  applyCommand(cmd, speed);
}

void setup() {
  pinMode(AIN1_F, OUTPUT);
  pinMode(AIN2_F, OUTPUT);
  pinMode(PWMA_F, OUTPUT);
  pinMode(BIN1_F, OUTPUT);
  pinMode(BIN2_F, OUTPUT);
  pinMode(PWMB_F, OUTPUT);

  pinMode(AIN1_R, OUTPUT);
  pinMode(AIN2_R, OUTPUT);
  pinMode(PWMA_R, OUTPUT);
  pinMode(BIN1_R, OUTPUT);
  pinMode(BIN2_R, OUTPUT);
  pinMode(PWMB_R, OUTPUT);

  pinMode(STBY, OUTPUT);
  digitalWrite(STBY, HIGH);

  stopAll();
  Serial.begin(115200);
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      handleLine(buffer);
      buffer = "";
    } else if (c != '\r') {
      buffer += c;
    }
  }

  if (turn_active && millis() >= turn_until) {
    stopAll();
    turn_active = false;
  }
}

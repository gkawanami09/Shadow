#include <Arduino.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

struct PinosMotor {
  uint8_t pino_in1;
  uint8_t pino_in2;
  uint8_t pino_pwm;
};

// Mapeamento
const PinosMotor MOTOR_TRASEIRO_ESQUERDO = {2, 4, 3};
const PinosMotor MOTOR_TRASEIRO_DIREITO = {13, 12, 11};
const PinosMotor MOTOR_FRENTE_ESQUERDO = {10, 9, 5};
const PinosMotor MOTOR_FRENTE_DIREITO = {8, 7, 6};

const int VELOCIDADE_PADRAO = 150;
const int VELOCIDADE_GIRO_180_PADRAO = 130;
const int VELOCIDADE_GIRO_90_PADRAO = 140;
const unsigned long DURACAO_GIRO_180_MS = 1100;
const unsigned long DURACAO_GIRO_90_MS = 520;

const size_t TAMANHO_BUFFER_SERIAL = 64;
char buffer_serial[TAMANHO_BUFFER_SERIAL];
size_t indice_buffer_serial = 0;

bool manobra_programada_ativa = false;
unsigned long manobra_programada_ate_ms = 0;

int limitar_pwm_positivo(int valor) {
  return constrain(valor, 0, 255);
}

char* avancar_espacos(char* texto) {
  while (*texto == ' ' || *texto == '\t') {
    ++texto;
  }
  return texto;
}

void remover_espacos_finais(char* texto) {
  int tamanho = strlen(texto);
  while (tamanho > 0) {
    char atual = texto[tamanho - 1];
    if (atual != ' ' && atual != '\t') {
      break;
    }
    texto[tamanho - 1] = '\0';
    --tamanho;
  }
}

void definir_motor(const PinosMotor& motor, int velocidade_assinada) {
  velocidade_assinada = constrain(velocidade_assinada, -255, 255);

  if (velocidade_assinada > 0) {
    digitalWrite(motor.pino_in1, HIGH);
    digitalWrite(motor.pino_in2, LOW);
    analogWrite(motor.pino_pwm, velocidade_assinada);
    return;
  }

  if (velocidade_assinada < 0) {
    digitalWrite(motor.pino_in1, LOW);
    digitalWrite(motor.pino_in2, HIGH);
    analogWrite(motor.pino_pwm, -velocidade_assinada);
    return;
  }

  digitalWrite(motor.pino_in1, LOW);
  digitalWrite(motor.pino_in2, LOW);
  analogWrite(motor.pino_pwm, 0);
}

void definir_lado_esquerdo(int velocidade_assinada) {
  definir_motor(MOTOR_FRENTE_ESQUERDO, velocidade_assinada);
  definir_motor(MOTOR_TRASEIRO_ESQUERDO, velocidade_assinada);
}

void definir_lado_direito(int velocidade_assinada) {
  definir_motor(MOTOR_FRENTE_DIREITO, velocidade_assinada);
  definir_motor(MOTOR_TRASEIRO_DIREITO, velocidade_assinada);
}

void definir_velocidades_diferenciais(int velocidade_esquerda, int velocidade_direita) {
  int velocidade_esquerda_limitada = constrain(velocidade_esquerda, -255, 255);
  int velocidade_direita_limitada = constrain(velocidade_direita, -255, 255);
  definir_lado_esquerdo(velocidade_esquerda_limitada);
  definir_lado_direito(velocidade_direita_limitada);
}

void parar_todos() {
  definir_lado_esquerdo(0);
  definir_lado_direito(0);
}

void iniciar_manobra_programada(int velocidade_esquerda, int velocidade_direita, unsigned long duracao_ms) {
  manobra_programada_ativa = true;
  manobra_programada_ate_ms = millis() + duracao_ms;
  definir_lado_esquerdo(velocidade_esquerda);
  definir_lado_direito(velocidade_direita);
}

void iniciar_giro_180(int velocidade) {
  int velocidade_giro = limitar_pwm_positivo(velocidade);
  if (velocidade_giro == 0) {
    velocidade_giro = VELOCIDADE_GIRO_180_PADRAO;
  }

  iniciar_manobra_programada(velocidade_giro, -velocidade_giro, DURACAO_GIRO_180_MS);
}

void iniciar_giro_90_esquerda(int velocidade) {
  int velocidade_giro = limitar_pwm_positivo(velocidade);
  if (velocidade_giro == 0) {
    velocidade_giro = VELOCIDADE_GIRO_90_PADRAO;
  }

  iniciar_manobra_programada(-velocidade_giro, velocidade_giro, DURACAO_GIRO_90_MS);
}

void iniciar_giro_90_direita(int velocidade) {
  int velocidade_giro = limitar_pwm_positivo(velocidade);
  if (velocidade_giro == 0) {
    velocidade_giro = VELOCIDADE_GIRO_90_PADRAO;
  }

  iniciar_manobra_programada(velocidade_giro, -velocidade_giro, DURACAO_GIRO_90_MS);
}

bool aplicar_comando(const char* comando, int velocidade_principal, int velocidade_direita) {
  if (comando == nullptr || *comando == '\0') {
    return false;
  }

  int velocidade = limitar_pwm_positivo(velocidade_principal);
  bool comando_simples = (comando[0] != '\0' && comando[1] == '\0');

  if (strcmp(comando, "L90") == 0) {
    iniciar_giro_90_esquerda(velocidade);
    return true;
  }

  if (strcmp(comando, "R90") == 0) {
    iniciar_giro_90_direita(velocidade);
    return true;
  }

  if (!comando_simples) {
    return false;
  }

  switch (comando[0]) {
    case 'F':
      manobra_programada_ativa = false;
      definir_lado_esquerdo(velocidade);
      definir_lado_direito(velocidade);
      return true;

    case 'B':
      manobra_programada_ativa = false;
      definir_lado_esquerdo(-velocidade);
      definir_lado_direito(-velocidade);
      return true;

    case 'L':
      manobra_programada_ativa = false;
      definir_lado_esquerdo(-velocidade);
      definir_lado_direito(velocidade);
      return true;

    case 'R':
      manobra_programada_ativa = false;
      definir_lado_esquerdo(velocidade);
      definir_lado_direito(-velocidade);
      return true;

    case 'S':
      manobra_programada_ativa = false;
      parar_todos();
      return true;

    case 'U':
      iniciar_giro_180(velocidade);
      return true;

    case 'D':
      manobra_programada_ativa = false;
      definir_velocidades_diferenciais(velocidade_principal, velocidade_direita);
      return true;

    default:
      return false;
  }
}

void processar_linha_serial(char* linha) {
  if (linha == nullptr) {
    return;
  }

  char* inicio = avancar_espacos(linha);
  remover_espacos_finais(inicio);
  if (*inicio == '\0') {
    return;
  }

  char comando = static_cast<char>(toupper(static_cast<unsigned char>(*inicio)));
  char* separador = strchr(inicio, ',');
  char comando_texto[12];
  size_t tamanho_comando = 0;

  while (inicio[tamanho_comando] != '\0' &&
         inicio[tamanho_comando] != ',' &&
         inicio[tamanho_comando] != ' ' &&
         inicio[tamanho_comando] != '\t' &&
         tamanho_comando < (sizeof(comando_texto) - 1)) {
    comando_texto[tamanho_comando] =
        static_cast<char>(toupper(static_cast<unsigned char>(inicio[tamanho_comando])));
    ++tamanho_comando;
  }
  comando_texto[tamanho_comando] = '\0';

  if (strcmp(comando_texto, "D") == 0) {
    char* primeira_virgula = separador;
    if (primeira_virgula == nullptr) {
      return;
    }

    char* fim_esquerda = nullptr;
    long valor_esquerda = strtol(primeira_virgula + 1, &fim_esquerda, 10);
    if (fim_esquerda == primeira_virgula + 1 || fim_esquerda == nullptr || *fim_esquerda != ',') {
      return;
    }

    char* fim_direita = nullptr;
    long valor_direita = strtol(fim_esquerda + 1, &fim_direita, 10);
    if (fim_direita == fim_esquerda + 1 || fim_direita == nullptr) {
      return;
    }

    fim_direita = avancar_espacos(fim_direita);
    if (*fim_direita != '\0') {
      return;
    }

    aplicar_comando("D", static_cast<int>(valor_esquerda), static_cast<int>(valor_direita));
    return;
  }

  int velocidade = VELOCIDADE_PADRAO;
  char* resto = separador != nullptr ? separador : (inicio + tamanho_comando);
  resto = avancar_espacos(resto);

  if (*resto == '\0') {
    aplicar_comando(comando_texto, velocidade, velocidade);
    return;
  }

  if (*resto != ',') {
    return;
  }

  char* fim_velocidade = nullptr;
  long valor_velocidade = strtol(resto + 1, &fim_velocidade, 10);
  if (fim_velocidade == resto + 1 || fim_velocidade == nullptr) {
    return;
  }

  fim_velocidade = avancar_espacos(fim_velocidade);
  if (*fim_velocidade != '\0') {
    return;
  }

  velocidade = static_cast<int>(valor_velocidade);
  aplicar_comando(comando_texto, velocidade, velocidade);
}

void configurar_pinos() {
  pinMode(MOTOR_TRASEIRO_ESQUERDO.pino_in1, OUTPUT);
  pinMode(MOTOR_TRASEIRO_ESQUERDO.pino_in2, OUTPUT);
  pinMode(MOTOR_TRASEIRO_ESQUERDO.pino_pwm, OUTPUT);

  pinMode(MOTOR_TRASEIRO_DIREITO.pino_in1, OUTPUT);
  pinMode(MOTOR_TRASEIRO_DIREITO.pino_in2, OUTPUT);
  pinMode(MOTOR_TRASEIRO_DIREITO.pino_pwm, OUTPUT);

  pinMode(MOTOR_FRENTE_ESQUERDO.pino_in1, OUTPUT);
  pinMode(MOTOR_FRENTE_ESQUERDO.pino_in2, OUTPUT);
  pinMode(MOTOR_FRENTE_ESQUERDO.pino_pwm, OUTPUT);

  pinMode(MOTOR_FRENTE_DIREITO.pino_in1, OUTPUT);
  pinMode(MOTOR_FRENTE_DIREITO.pino_in2, OUTPUT);
  pinMode(MOTOR_FRENTE_DIREITO.pino_pwm, OUTPUT);

  parar_todos();
}

void setup() {
  configurar_pinos();
  Serial.begin(115200);
}

void loop() {
  while (Serial.available() > 0) {
    char caractere = static_cast<char>(Serial.read());

    if (caractere == '\n') {
      buffer_serial[indice_buffer_serial] = '\0';
      processar_linha_serial(buffer_serial);
      indice_buffer_serial = 0;
      continue;
    }

    if (caractere == '\r') {
      continue;
    }

    if (!isprint(static_cast<unsigned char>(caractere)) && caractere != '\t') {
      continue;
    }

    if (indice_buffer_serial < TAMANHO_BUFFER_SERIAL - 1) {
      buffer_serial[indice_buffer_serial++] = caractere;
    } else {
      // Descarta pacote muito grande para evitar comandos parciais perigosos.
      indice_buffer_serial = 0;
    }
  }

  if (manobra_programada_ativa && static_cast<long>(millis() - manobra_programada_ate_ms) >= 0) {
    manobra_programada_ativa = false;
    parar_todos();
  }
}

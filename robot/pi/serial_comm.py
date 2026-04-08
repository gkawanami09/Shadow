try:
    import serial
except Exception:  # pragma: no cover
    serial = None


COMANDO_PARAR = "S"
COMANDO_RETO = "F"
COMANDO_ESQUERDA = "L"
COMANDO_DIREITA = "R"
COMANDO_GIRO_180 = "U"
COMANDO_DIFERENCIAL = "D"


def _limitar_pwm(valor):
    return max(0, min(255, int(valor)))


def _limitar_pwm_assinado(valor):
    return max(-255, min(255, int(valor)))


def _formatar_comando(comando, velocidade=None):
    if velocidade is None:
        return f"{comando}\n"
    return f"{comando},{_limitar_pwm(velocidade)}\n"


def _formatar_comando_diferencial(velocidade_esquerda, velocidade_direita):
    esquerda = _limitar_pwm_assinado(velocidade_esquerda)
    direita = _limitar_pwm_assinado(velocidade_direita)
    return f"{COMANDO_DIFERENCIAL},{esquerda},{direita}\n"


def abrir_serial(porta=None, baud=115200, port=None):
    if porta is None:
        porta = port
    if porta is None or serial is None:
        return None
    return serial.Serial(port=porta, baudrate=baud, timeout=0.1)


def enviar_serial(ser, comando, velocidade=None):
    if ser is None:
        return
    pacote = _formatar_comando(comando, velocidade)
    ser.write(pacote.encode("ascii"))


def enviar_velocidades_diferenciais(ser, velocidade_esquerda, velocidade_direita):
    if ser is None:
        return
    pacote = _formatar_comando_diferencial(velocidade_esquerda, velocidade_direita)
    ser.write(pacote.encode("ascii"))


def parar(ser):
    enviar_serial(ser, COMANDO_PARAR)


def reto(ser, velocidade=None):
    enviar_serial(ser, COMANDO_RETO, velocidade)


def reto_forte(ser):
    enviar_serial(ser, COMANDO_RETO, 255)


def virar_esquerda(ser, velocidade=None):
    enviar_serial(ser, COMANDO_ESQUERDA, velocidade)


def virar_direita(ser, velocidade=None):
    enviar_serial(ser, COMANDO_DIREITA, velocidade)


def corrigir_esquerda(ser, velocidade=None):
    enviar_serial(ser, COMANDO_ESQUERDA, velocidade)


def corrigir_direita(ser, velocidade=None):
    enviar_serial(ser, COMANDO_DIREITA, velocidade)


def beco(ser, velocidade=None):
    enviar_serial(ser, COMANDO_GIRO_180, velocidade)


def giro_180(ser, velocidade=None):
    enviar_serial(ser, COMANDO_GIRO_180, velocidade)


def parar_vermelho(ser):
    enviar_serial(ser, COMANDO_PARAR)


def fechar_serial(ser):
    if ser is not None:
        ser.close()

try:
    import serial
except Exception:  # pragma: no cover
    serial = None


def abrir_serial(port, baud=115200):
    if port is None or serial is None:
        return None
    return serial.Serial(port=port, baudrate=baud, timeout=0.1)


CMD_PARAR = "S"
CMD_RETO = "F"
CMD_ESQUERDA = "L"
CMD_DIREITA = "R"
CMD_GIRO_180 = "U"


def _formatar_comando(comando, velocidade=None):
    if velocidade is None:
        return f"{comando}\n"
    velocidade = max(0, min(255, int(velocidade)))
    return f"{comando},{velocidade}\n"


def enviar_serial(ser, comando, velocidade=None):
    if ser is None:
        return
    payload = _formatar_comando(comando, velocidade)
    ser.write(payload.encode("ascii"))


def parar(ser):
    enviar_serial(ser, CMD_PARAR)


def reto(ser, velocidade=None):
    enviar_serial(ser, CMD_RETO, velocidade)


def reto_forte(ser):
    enviar_serial(ser, CMD_RETO, 255)


def virar_esquerda(ser, velocidade=None):
    enviar_serial(ser, CMD_ESQUERDA, velocidade)


def virar_direita(ser, velocidade=None):
    enviar_serial(ser, CMD_DIREITA, velocidade)


def corrigir_esquerda(ser, velocidade=None):
    enviar_serial(ser, CMD_ESQUERDA, velocidade)


def corrigir_direita(ser, velocidade=None):
    enviar_serial(ser, CMD_DIREITA, velocidade)


def beco(ser, velocidade=None):
    enviar_serial(ser, CMD_GIRO_180, velocidade)


def giro_180(ser, velocidade=None):
    enviar_serial(ser, CMD_GIRO_180, velocidade)


def parar_vermelho(ser):
    enviar_serial(ser, CMD_PARAR)


def fechar_serial(ser):
    if ser is not None:
        ser.close()

try:
    import serial
except Exception:  # pragma: no cover
    serial = None


COMANDO_PARAR = "S"
COMANDO_DIFERENCIAL = "D"
COMANDO_GIRO_90_ESQUERDA = "L90"
COMANDO_GIRO_90_DIREITA = "R90"


def _limitar_pwm(valor):
    return max(0, min(255, int(valor)))


def _limitar_pwm_assinado(valor):
    return max(-255, min(255, int(valor)))


def _formatar_comando_simples(comando, velocidade=None):
    if velocidade is None:
        return f"{comando}\n"
    return f"{comando},{_limitar_pwm(velocidade)}\n"


def _formatar_comando_diferencial(velocidade_esquerda, velocidade_direita):
    esquerda = _limitar_pwm_assinado(velocidade_esquerda)
    direita = _limitar_pwm_assinado(velocidade_direita)
    return f"{COMANDO_DIFERENCIAL},{esquerda},{direita}\n"


def _remapear_lados_para_hardware(velocidade_esquerda, velocidade_direita):
    # O cabeamento atual responde com os lados fisicos invertidos; este remapeamento
    # faz a API Python continuar semanticamente correta sem tocar no firmware.
    return velocidade_direita, velocidade_esquerda


def abrir_serial(porta=None, baud=115200, port=None):
    if porta is None:
        porta = port
    if porta is None or serial is None:
        return None
    return serial.Serial(port=porta, baudrate=baud, timeout=0.1)


def _enviar_texto(ser, texto):
    if ser is None:
        return
    ser.write(texto.encode("ascii"))


def enviar_velocidades_diferenciais(ser, velocidade_esquerda, velocidade_direita):
    velocidade_esquerda_hw, velocidade_direita_hw = _remapear_lados_para_hardware(
        velocidade_esquerda,
        velocidade_direita,
    )
    _enviar_texto(ser, _formatar_comando_diferencial(velocidade_esquerda_hw, velocidade_direita_hw))


def giro_90_esquerda(ser, velocidade=None):
    _enviar_texto(ser, _formatar_comando_simples(COMANDO_GIRO_90_DIREITA, velocidade))


def giro_90_direita(ser, velocidade=None):
    _enviar_texto(ser, _formatar_comando_simples(COMANDO_GIRO_90_ESQUERDA, velocidade))


def parar(ser):
    _enviar_texto(ser, _formatar_comando_simples(COMANDO_PARAR))


def fechar_serial(ser):
    if ser is not None:
        ser.close()

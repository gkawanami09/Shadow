import argparse
import sys
import time

from serial_comm import (
    abrir_serial,
    enviar_velocidades_diferenciais,
    fechar_serial,
    parar,
)


def analisar_argumentos():
    analisador = argparse.ArgumentParser(
        description=(
            "Teste temporizado do driver: verifica frente, re e giros "
            "mandando velocidades diferenciais conhecidas."
        ),
    )
    analisador.add_argument("--port", required=True, help="Porta serial, por exemplo /dev/ttyACM0.")
    analisador.add_argument("--baud", type=int, default=115200, help="Baudrate da serial.")
    analisador.add_argument("--velocidade", type=int, default=120, help="PWM absoluto de 0 a 255.")
    analisador.add_argument(
        "--duracao-etapa",
        type=float,
        default=1.5,
        help="Duracao de cada etapa de movimento em segundos.",
    )
    analisador.add_argument(
        "--pausa",
        type=float,
        default=0.5,
        help="Pausa entre etapas em segundos.",
    )
    analisador.add_argument(
        "--intervalo-envio",
        type=float,
        default=0.10,
        help="Intervalo entre reenvios do comando durante cada etapa.",
    )
    return analisador.parse_args()


def limitar_pwm(valor):
    return max(0, min(255, int(round(valor))))


def executar_etapa(ser, nome, velocidade_esquerda, velocidade_direita, duracao, intervalo_envio):
    print(
        f"{nome}: esquerda={velocidade_esquerda:+d} direita={velocidade_direita:+d} "
        f"por {duracao:.2f}s",
        flush=True,
    )

    inicio = time.monotonic()
    proximo_envio = inicio
    while True:
        agora = time.monotonic()
        if agora >= (inicio + duracao):
            break

        if agora >= proximo_envio:
            enviar_velocidades_diferenciais(ser, velocidade_esquerda, velocidade_direita)
            proximo_envio = agora + intervalo_envio

        time.sleep(0.01)


def principal():
    parametros = analisar_argumentos()
    velocidade = limitar_pwm(parametros.velocidade)
    duracao_etapa = max(0.1, float(parametros.duracao_etapa))
    pausa = max(0.1, float(parametros.pausa))
    intervalo_envio = max(0.02, float(parametros.intervalo_envio))

    try:
        ser = abrir_serial(porta=parametros.port, baud=parametros.baud)
    except Exception as excecao:
        print(f"Erro ao abrir serial em {parametros.port}: {excecao}", file=sys.stderr)
        return 1

    if ser is None:
        print("Serial indisponivel. Verifique pyserial e a porta informada.", file=sys.stderr)
        return 1

    etapas = [
        ("ambos_frente", velocidade, velocidade),
        ("ambos_re", -velocidade, -velocidade),
        ("esquerda_frente_direita_parada", velocidade, 0),
        ("esquerda_re_direita_parada", -velocidade, 0),
        ("esquerda_parada_direita_frente", 0, velocidade),
        ("esquerda_parada_direita_re", 0, -velocidade),
        ("giro_esquerda", -velocidade, velocidade),
        ("giro_direita", velocidade, -velocidade),
    ]

    print("Teste de resposta do driver iniciado.", flush=True)
    print(
        "Observe cada roda em cada etapa para confirmar se o sentido bate com o comando enviado.",
        flush=True,
    )

    try:
        parar(ser)
        time.sleep(pausa)

        for nome, velocidade_esquerda, velocidade_direita in etapas:
            executar_etapa(
                ser,
                nome,
                velocidade_esquerda,
                velocidade_direita,
                duracao_etapa,
                intervalo_envio,
            )
            parar(ser)
            time.sleep(pausa)

    except KeyboardInterrupt:
        print("Teste interrompido pelo usuario.", flush=True)
    finally:
        try:
            parar(ser)
            time.sleep(0.2)
            parar(ser)
        finally:
            fechar_serial(ser)

    print("Teste finalizado. Robo parado.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(principal())

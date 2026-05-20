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
            "Teste Python-only para o lado esquerdo em re. "
            "Manda esquerda negativa e direita parada em varias velocidades."
        ),
    )
    analisador.add_argument("--port", required=True, help="Porta serial, por exemplo /dev/ttyACM0.")
    analisador.add_argument("--baud", type=int, default=115200, help="Baudrate da serial.")
    analisador.add_argument(
        "--velocidade-max",
        type=int,
        default=250,
        help="Maior PWM absoluto usado na varredura.",
    )
    analisador.add_argument(
        "--passo",
        type=int,
        default=10,
        help="Passo entre velocidades na varredura.",
    )
    analisador.add_argument(
        "--duracao-etapa",
        type=float,
        default=0.35,
        help="Duracao de cada velocidade em segundos.",
    )
    analisador.add_argument(
        "--pausa",
        type=float,
        default=0.15,
        help="Pausa entre etapas.",
    )
    analisador.add_argument(
        "--intervalo-envio",
        type=float,
        default=0.08,
        help="Intervalo entre reenvios do comando durante cada etapa.",
    )
    return analisador.parse_args()


def limitar_pwm(valor):
    return max(0, min(255, int(round(valor))))


def executar_etapa(ser, velocidade_esquerda, velocidade_direita, duracao, intervalo_envio):
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
    velocidade_max = limitar_pwm(parametros.velocidade_max)
    passo = max(1, int(abs(parametros.passo)))
    duracao_etapa = max(0.05, float(parametros.duracao_etapa))
    pausa = max(0.0, float(parametros.pausa))
    intervalo_envio = max(0.02, float(parametros.intervalo_envio))

    try:
        ser = abrir_serial(porta=parametros.port, baud=parametros.baud)
    except Exception as excecao:
        print(f"Erro ao abrir serial em {parametros.port}: {excecao}", file=sys.stderr)
        return 1

    if ser is None:
        print("Serial indisponivel. Verifique pyserial e a porta informada.", file=sys.stderr)
        return 1

    velocidades = list(range(velocidade_max, -1, -passo))
    if velocidades[-1] != 0:
        velocidades.append(0)

    print("Teste iniciado: esquerda em re, direita parada.", flush=True)
    print("Observe se os dois motores da esquerda acompanham a varredura.", flush=True)

    try:
        parar(ser)
        time.sleep(max(0.1, pausa))

        for velocidade in velocidades:
            velocidade_esquerda = -velocidade
            print(
                f"etapa: esquerda={velocidade_esquerda:+d} direita=+0 por {duracao_etapa:.2f}s",
                flush=True,
            )
            executar_etapa(
                ser,
                velocidade_esquerda,
                0,
                duracao_etapa,
                intervalo_envio,
            )
            parar(ser)
            if pausa > 0:
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

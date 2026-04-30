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
        description="Teste bruto de giro para a direita: lado esquerdo para frente e lado direito em re.",
    )
    analisador.add_argument("--port", required=True, help="Porta serial, por exemplo /dev/ttyACM0.")
    analisador.add_argument("--baud", type=int, default=115200, help="Baudrate da serial.")
    analisador.add_argument(
        "--velocidade",
        type=int,
        default=120,
        help="Velocidade absoluta usada no teste (0 a 255).",
    )
    analisador.add_argument(
        "--duracao",
        type=float,
        default=2.0,
        help="Duracao do teste em segundos.",
    )
    analisador.add_argument(
        "--intervalo-envio",
        type=float,
        default=0.10,
        help="Intervalo entre reenvios do comando durante o teste.",
    )
    return analisador.parse_args()


def limitar_pwm(valor):
    return max(0, min(255, int(round(valor))))


def principal():
    parametros = analisar_argumentos()
    velocidade = limitar_pwm(parametros.velocidade)
    duracao = max(0.1, float(parametros.duracao))
    intervalo_envio = max(0.02, float(parametros.intervalo_envio))

    try:
        ser = abrir_serial(porta=parametros.port, baud=parametros.baud)
    except Exception as excecao:
        print(f"Erro ao abrir serial em {parametros.port}: {excecao}", file=sys.stderr)
        return 1

    if ser is None:
        print("Serial indisponivel. Verifique pyserial e a porta informada.", file=sys.stderr)
        return 1

    inicio = time.monotonic()
    proximo_envio = inicio

    print(
        "Teste iniciado: giro para a direita no lugar "
        f"(esquerda=+{velocidade}, direita=-{velocidade}) por {duracao:.2f}s.",
        flush=True,
    )

    try:
        parar(ser)
        time.sleep(0.2)

        while True:
            agora = time.monotonic()
            if agora >= (inicio + duracao):
                break

            if agora >= proximo_envio:
                enviar_velocidades_diferenciais(ser, velocidade, -velocidade)
                proximo_envio = agora + intervalo_envio

            time.sleep(0.01)

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

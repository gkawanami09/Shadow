import argparse
import os
import sys
import time

try:
    import cv2
except ModuleNotFoundError as excecao:
    if excecao.name == "cv2":
        print(
            "Dependencia ausente: OpenCV (cv2).\n"
            "Instale com um destes comandos e tente novamente:\n"
            "  sudo apt update && sudo apt install -y python3-opencv python3-numpy\n"
            "ou\n"
            "  python3 -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(1)
    raise

from camera import fechar_camera, iniciar_camera, ler_frame
from stream import ServidorStream
from vision import ConfiguracaoVisao, EstadoVisao, analisar_quadro


ESTADO_SEGUINDO = "SEGUINDO_LINHA"
ESTADO_SEM_LINHA = "SEM_LINHA"


def analisar_argumentos():
    analisador = argparse.ArgumentParser(
        description="Modo visao somente para seguidor de linha preto.",
    )
    analisador.add_argument("--device", type=int, default=None, help="Indice da camera USB.")
    analisador.add_argument("--width", type=int, default=640, help="Largura da imagem.")
    analisador.add_argument("--height", type=int, default=480, help="Altura da imagem.")
    analisador.add_argument("--fps", type=int, default=30, help="Taxa de quadros desejada.")

    analisador.add_argument("--show", action="store_true", help="Mostra janela local.")
    analisador.add_argument("--no-show", action="store_true", help="Nao mostra janela local.")
    analisador.add_argument("--debug-path", default=None, help="Salva continuamente o quadro de debug.")
    analisador.add_argument("--stream", action="store_true", help="Publica stream MJPEG no navegador.")
    analisador.add_argument("--stream-host", default="0.0.0.0", help="Host do stream HTTP.")
    analisador.add_argument("--stream-port", type=int, default=8080, help="Porta do stream HTTP.")
    analisador.add_argument("--print-every", type=float, default=0.20, help="Intervalo minimo de logs.")

    analisador.add_argument("--roi", type=float, default=0.45)
    analisador.add_argument("--limiar-binario", type=int, default=None)
    analisador.add_argument("--invert", action="store_true", help="Inverte polaridade da linha.")
    analisador.add_argument("--area-minima-contorno", type=int, default=180)
    analisador.add_argument("--area-minima-linha", type=int, default=320)
    analisador.add_argument("--suavizacao-erro", type=float, default=0.40)
    analisador.add_argument("--limiar-confianca", type=float, default=0.10)

    return analisador.parse_args()


def criar_configuracao_visao(parametros):
    return ConfiguracaoVisao(
        roi=parametros.roi,
        limiar_binario=parametros.limiar_binario,
        inverter_linha=parametros.invert,
        area_minima_contorno=parametros.area_minima_contorno,
        area_minima_linha=parametros.area_minima_linha,
        suavizacao_erro=parametros.suavizacao_erro,
        limiar_confianca=parametros.limiar_confianca,
    )


def _inferir_estado_visual(dados_visao, limiar_confianca):
    if dados_visao["linha_encontrada"] and dados_visao["confianca_linha"] >= limiar_confianca:
        return ESTADO_SEGUINDO
    return ESTADO_SEM_LINHA


def _imprimir_status(dados_visao, estado_visual):
    print(
        " | ".join(
            [
                f"estado={estado_visual}",
                f"erro={dados_visao['erro_linha']:+.3f}",
                f"conf={dados_visao['confianca_linha']:.2f}",
                f"tempo_sem_linha={dados_visao['tempo_sem_linha']:.2f}s",
            ]
        ),
        flush=True,
    )


def principal():
    parametros = analisar_argumentos()
    configuracao_visao = criar_configuracao_visao(parametros)
    estado_visao = EstadoVisao()

    tem_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    exibir_janela = parametros.show or (tem_display and not parametros.no_show)
    servidor_stream = None

    try:
        priorizar_picamera2 = parametros.device is None
        camera = iniciar_camera(
            device=parametros.device,
            width=parametros.width,
            height=parametros.height,
            framerate=parametros.fps,
            prefer_picamera2=priorizar_picamera2,
            fallback_picamera2=True,
        )
    except RuntimeError as excecao:
        print(f"Erro ao abrir camera: {excecao}", file=sys.stderr)
        return 1

    print(
        f"Camera pronta via {camera['backend']} ({camera['device']}) em {parametros.width}x{parametros.height}@{parametros.fps}."
    )
    print("Modo visao somente ativo.")

    if exibir_janela:
        cv2.namedWindow("visao_debug", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("visao_debug", max(640, parametros.width), max(480, parametros.height))

    if parametros.stream:
        servidor_stream = ServidorStream(host=parametros.stream_host, port=parametros.stream_port)
        servidor_stream.iniciar()
        print(
            f"Stream ativo em http://127.0.0.1:{parametros.stream_port} e http://IP_DA_RASPBERRY:{parametros.stream_port}."
        )

    instante_ultimo_log = 0.0
    quantidade_quadros = 0
    inicio_execucao = time.monotonic()

    try:
        while True:
            quadro_bgr = ler_frame(camera)
            if quadro_bgr is None:
                print("Falha ao ler quadro da camera.", file=sys.stderr)
                break

            dados_visao = analisar_quadro(quadro_bgr, configuracao_visao, estado_visao)
            estado_visual = _inferir_estado_visual(dados_visao, parametros.limiar_confianca)
            quantidade_quadros += 1

            quadro_debug = dados_visao["quadro_debug"].copy()
            cv2.putText(
                quadro_debug,
                f"estado_visual={estado_visual}",
                (12, quadro_debug.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            agora = time.monotonic()
            if parametros.print_every <= 0 or (agora - instante_ultimo_log) >= parametros.print_every:
                _imprimir_status(dados_visao, estado_visual)
                instante_ultimo_log = agora

            if parametros.debug_path:
                cv2.imwrite(parametros.debug_path, quadro_debug)

            if servidor_stream is not None:
                servidor_stream.atualizar_quadro(quadro_debug)

            if exibir_janela:
                cv2.imshow("visao_debug", quadro_debug)
                tecla = cv2.waitKey(1) & 0xFF
                if tecla in (27, ord("q")):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        fechar_camera(camera)
        if servidor_stream is not None:
            servidor_stream.parar()
        if exibir_janela:
            cv2.destroyAllWindows()

    duracao = max(0.001, time.monotonic() - inicio_execucao)
    fps_medio = quantidade_quadros / duracao
    print(f"Encerrado. quadros={quantidade_quadros}, fps_medio={fps_medio:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(principal())

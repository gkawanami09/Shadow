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
        description="Modo visao simplificado: correcao de linha, curva de 90 e verde.",
    )
    analisador.add_argument("--device", type=int, default=None, help="Indice da camera USB.")
    analisador.add_argument("--width", type=int, default=640, help="Largura da imagem.")
    analisador.add_argument("--height", type=int, default=480, help="Altura da imagem.")
    analisador.add_argument("--fps", type=int, default=30, help="Taxa de quadros desejada.")
    analisador.add_argument(
        "--autofocus",
        dest="autofocus",
        action="store_true",
        default=True,
        help="Liga foco automatico continuo quando a camera suportar.",
    )
    analisador.add_argument(
        "--no-autofocus",
        dest="autofocus",
        action="store_false",
        help="Desliga foco automatico e mantem o foco manual/padrao.",
    )
    analisador.add_argument(
        "--focus-value",
        type=float,
        default=None,
        help="Foco manual. Webcam USB costuma usar 0-255; camera CSI usa LensPosition.",
    )
    analisador.add_argument(
        "--camera-buffer-drop",
        type=int,
        default=2,
        help="Quantidade de frames antigos descartados por leitura para reduzir atraso.",
    )

    analisador.add_argument("--show", action="store_true", help="Mostra janela local.")
    analisador.add_argument("--no-show", action="store_true", help="Nao mostra janela local.")
    analisador.add_argument("--debug-path", default=None, help="Salva continuamente o quadro de debug.")
    analisador.add_argument(
        "--debug-write-interval",
        type=float,
        default=0.25,
        help="Intervalo minimo entre escritas em --debug-path.",
    )
    analisador.add_argument("--stream", action="store_true", help="Publica stream MJPEG no navegador.")
    analisador.add_argument("--stream-host", default="0.0.0.0", help="Host do stream HTTP.")
    analisador.add_argument("--stream-port", type=int, default=8080, help="Porta do stream HTTP.")
    analisador.add_argument(
        "--stream-jpeg-quality",
        type=int,
        default=70,
        help="Qualidade JPEG do stream (menor = menos latencia).",
    )
    analisador.add_argument("--print-every", type=float, default=0.20, help="Intervalo minimo de logs.")

    analisador.add_argument("--roi", type=float, default=0.48)
    analisador.add_argument("--limiar-binario", type=int, default=None)
    analisador.add_argument(
        "--invert",
        action="store_true",
        help="Use apenas quando a linha for clara em fundo escuro.",
    )
    analisador.add_argument(
        "--sem-limiar-adaptativo",
        action="store_true",
        help="Desliga o limiar adaptativo de apoio contra brilho.",
    )
    analisador.add_argument("--bloco-limiar-adaptativo", type=int, default=41)
    analisador.add_argument("--constante-limiar-adaptativo", type=int, default=9)
    analisador.add_argument("--area-minima-contorno", type=int, default=180)
    analisador.add_argument("--area-minima-linha", type=int, default=320)
    analisador.add_argument("--suavizacao-erro", type=float, default=0.40)
    analisador.add_argument("--limiar-confianca", type=float, default=0.10)
    analisador.add_argument("--faixa-base-contorno", type=float, default=0.16)
    analisador.add_argument("--margem-lateral-descarte", type=float, default=0.10)
    analisador.add_argument("--lookahead-fracao", type=float, default=0.42)
    analisador.add_argument("--lookahead-minimo-pixels", type=int, default=18)
    analisador.add_argument("--limiar-confianca-curva-90", type=float, default=0.18)
    analisador.add_argument("--limiar-confianca-lookahead-curva-90", type=float, default=0.14)
    analisador.add_argument("--limiar-erro-lookahead-curva-90", type=float, default=0.36)
    analisador.add_argument("--limiar-delta-erro-curva-90", type=float, default=0.10)
    analisador.add_argument("--limiar-erro-base-curva-90", type=float, default=0.30)
    analisador.add_argument("--faixa-superior-curva-90", type=float, default=0.48)
    analisador.add_argument("--faixa-inferior-curva-90", type=float, default=0.24)
    analisador.add_argument("--densidade-lateral-curva-90", type=float, default=0.16)
    analisador.add_argument("--densidade-oposta-max-curva-90", type=float, default=0.08)
    analisador.add_argument("--densidade-base-centro-curva-90", type=float, default=0.10)
    analisador.add_argument("--roi-verde", type=float, default=0.75)
    analisador.add_argument("--verde-h-min", type=int, default=35)
    analisador.add_argument("--verde-h-max", type=int, default=95)
    analisador.add_argument("--verde-s-min", type=int, default=60)
    analisador.add_argument("--verde-v-min", type=int, default=45)
    analisador.add_argument("--area-minima-verde", type=int, default=180)

    return analisador.parse_args()


def criar_configuracao_visao(parametros):
    return ConfiguracaoVisao(
        roi=parametros.roi,
        limiar_binario=parametros.limiar_binario,
        inverter_linha=parametros.invert,
        usar_limiar_adaptativo=not parametros.sem_limiar_adaptativo,
        bloco_limiar_adaptativo=parametros.bloco_limiar_adaptativo,
        constante_limiar_adaptativo=parametros.constante_limiar_adaptativo,
        area_minima_contorno=parametros.area_minima_contorno,
        area_minima_linha=parametros.area_minima_linha,
        suavizacao_erro=parametros.suavizacao_erro,
        limiar_confianca=parametros.limiar_confianca,
        faixa_base_contorno=parametros.faixa_base_contorno,
        margem_lateral_descarte=parametros.margem_lateral_descarte,
        lookahead_fracao=parametros.lookahead_fracao,
        lookahead_minimo_pixels=parametros.lookahead_minimo_pixels,
        limiar_confianca_curva_90=parametros.limiar_confianca_curva_90,
        limiar_confianca_lookahead_curva_90=parametros.limiar_confianca_lookahead_curva_90,
        limiar_erro_lookahead_curva_90=parametros.limiar_erro_lookahead_curva_90,
        limiar_delta_erro_curva_90=parametros.limiar_delta_erro_curva_90,
        limiar_erro_base_curva_90=parametros.limiar_erro_base_curva_90,
        faixa_superior_curva_90=parametros.faixa_superior_curva_90,
        faixa_inferior_curva_90=parametros.faixa_inferior_curva_90,
        densidade_lateral_curva_90=parametros.densidade_lateral_curva_90,
        densidade_oposta_max_curva_90=parametros.densidade_oposta_max_curva_90,
        densidade_base_centro_curva_90=parametros.densidade_base_centro_curva_90,
        roi_verde=parametros.roi_verde,
        verde_h_min=parametros.verde_h_min,
        verde_h_max=parametros.verde_h_max,
        verde_s_min=parametros.verde_s_min,
        verde_v_min=parametros.verde_v_min,
        area_minima_verde=parametros.area_minima_verde,
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
                f"lookahead={dados_visao.get('erro_lookahead', 0.0):+.3f}",
                f"conf={dados_visao['confianca_linha']:.2f}",
                f"curva90={dados_visao.get('confianca_curva_90', 0.0):.2f}",
                f"verde={'SIM' if dados_visao.get('verde_detectado') else 'NAO'}",
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
    gerar_debug_visual = bool(exibir_janela or parametros.stream or parametros.debug_path)
    servidor_stream = None

    try:
        priorizar_picamera2 = parametros.device is None
        camera = iniciar_camera(
            device=parametros.device,
            width=parametros.width,
            height=parametros.height,
            framerate=parametros.fps,
            autofocus=parametros.autofocus,
            focus_value=parametros.focus_value,
            quadros_descartados_por_leitura=parametros.camera_buffer_drop,
            prefer_picamera2=priorizar_picamera2,
            fallback_picamera2=True,
        )
    except RuntimeError as excecao:
        print(f"Erro ao abrir camera: {excecao}", file=sys.stderr)
        return 1

    print(
        f"Camera pronta via {camera['backend']} ({camera['device']}) em "
        f"{parametros.width}x{parametros.height}@{parametros.fps} "
        f"| foco={camera.get('descricao_foco', 'padrao')} "
        f"| drop={camera.get('quadros_descartados_por_leitura', 0)}."
    )
    print("Modo visao simplificado ativo.")

    if exibir_janela:
        cv2.namedWindow("visao_debug", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("visao_debug", max(640, parametros.width), max(480, parametros.height))

    if parametros.stream:
        servidor_stream = ServidorStream(
            host=parametros.stream_host,
            port=parametros.stream_port,
            qualidade_jpeg=parametros.stream_jpeg_quality,
        )
        servidor_stream.iniciar()
        print(
            f"Stream ativo em http://127.0.0.1:{parametros.stream_port} e http://IP_DA_RASPBERRY:{parametros.stream_port}."
        )

    instante_ultimo_log = 0.0
    instante_ultima_gravacao_debug = 0.0
    quantidade_quadros = 0
    inicio_execucao = time.monotonic()

    try:
        while True:
            quadro_bgr = ler_frame(camera)
            if quadro_bgr is None:
                print("Falha ao ler quadro da camera.", file=sys.stderr)
                break

            dados_visao = analisar_quadro(
                quadro_bgr,
                configuracao_visao,
                estado_visao,
                gerar_debug=gerar_debug_visual,
            )
            estado_visual = _inferir_estado_visual(dados_visao, parametros.limiar_confianca)
            quantidade_quadros += 1

            quadro_debug = None
            if gerar_debug_visual and dados_visao["quadro_debug"] is not None:
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

            if quadro_debug is not None:
                if parametros.debug_path and (
                    parametros.debug_write_interval <= 0
                    or (agora - instante_ultima_gravacao_debug) >= parametros.debug_write_interval
                ):
                    cv2.imwrite(parametros.debug_path, quadro_debug)
                    instante_ultima_gravacao_debug = agora

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
    
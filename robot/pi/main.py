import argparse
import os
import sys
import time

try:
    import cv2
except ModuleNotFoundError as exc:
    if exc.name == "cv2":
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

from camera import iniciar_camera, ler_frame, fechar_camera
from stream import StreamServer
from vision import VisionConfig, VisionState, analyze_frame


def parse_args():
    parser = argparse.ArgumentParser(
        description="Modo visao somente para seguidor de linha na Raspberry Pi."
    )
    parser.add_argument("--device", type=int, default=None, help="Forca um indice de camera.")
    parser.add_argument("--width", type=int, default=640, help="Largura da captura.")
    parser.add_argument("--height", type=int, default=480, help="Altura da captura.")
    parser.add_argument("--fps", type=int, default=30, help="FPS desejado.")
    parser.add_argument("--roi", type=float, default=0.40, help="Fracao inferior usada como ROI.")
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="Threshold fixo. Se omitido, usa Otsu.",
    )
    parser.add_argument("--invert", action="store_true", help="Inverte a polaridade da linha.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Mostra a janela de debug.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Nao mostra a janela mesmo se houver display.",
    )
    parser.add_argument(
        "--debug-path",
        default=None,
        help="Salva continuamente o ultimo frame de debug nesse caminho.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Publica o frame de debug via navegador (MJPEG).",
    )
    parser.add_argument(
        "--stream-host",
        default="0.0.0.0",
        help="Host do servidor de stream.",
    )
    parser.add_argument(
        "--stream-port",
        type=int,
        default=8080,
        help="Porta do servidor de stream.",
    )
    parser.add_argument(
        "--print-every",
        type=float,
        default=0.20,
        help="Intervalo minimo entre logs no terminal.",
    )
    parser.add_argument("--gap-tempo", type=float, default=0.35)
    parser.add_argument("--intersecao-largura", type=float, default=0.55)
    parser.add_argument("--giro-180-tempo", type=float, default=1.2)
    parser.add_argument("--giro-180-offset", type=float, default=1.0)
    parser.add_argument("--beco-cooldown", type=float, default=2.0)
    parser.add_argument("--verde-hmin", type=int, default=35)
    parser.add_argument("--verde-hmax", type=int, default=95)
    parser.add_argument("--verde-smin", type=int, default=60)
    parser.add_argument("--verde-vmin", type=int, default=60)
    parser.add_argument("--verde-area-min", type=int, default=250)
    parser.add_argument("--verde-zona", type=float, default=0.45)
    return parser.parse_args()


def build_config(args):
    return VisionConfig(
        roi=args.roi,
        threshold=args.threshold,
        invert=args.invert,
        gap_tempo=args.gap_tempo,
        intersecao_largura=args.intersecao_largura,
        giro_180_tempo=args.giro_180_tempo,
        giro_180_offset=args.giro_180_offset,
        beco_cooldown=args.beco_cooldown,
        verde_hmin=args.verde_hmin,
        verde_hmax=args.verde_hmax,
        verde_smin=args.verde_smin,
        verde_vmin=args.verde_vmin,
        verde_area_min=args.verde_area_min,
        verde_zona=args.verde_zona,
    )


def print_status(result):
    print(
        " | ".join(
            [
                f"estado={result['state']}",
                f"offset={result['offset']:+.3f}",
                f"confidence={result['confidence']:.2f}",
                f"intersecao={'SIM' if result['intersection'] else 'NAO'}",
                f"decisao_visual={result['suggested_command']}",
                f"verde={result['green_state']}",
                f"verde_info={result['green_detail']}",
            ]
        ),
        flush=True,
    )


def main():
    args = parse_args()
    config = build_config(args)
    state = VisionState()
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    should_show = args.show or (has_display and not args.no_show)
    stream_server = None

    try:
        camera = iniciar_camera(
            device=args.device,
            width=args.width,
            height=args.height,
            framerate=args.fps,
            prefer_picamera2=False,
            fallback_picamera2=True,
        )
    except RuntimeError as exc:
        print(f"Erro ao abrir camera: {exc}", file=sys.stderr)
        return 1

    print(
        f"Camera pronta via {camera['backend']} ({camera['device']}) "
        f"em {args.width}x{args.height}@{args.fps}."
    )
    print("Modo visao somente ativo. Pressione Ctrl+C para sair.")
    if should_show:
        cv2.namedWindow("vision_debug", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("vision_debug", max(640, args.width), max(480, args.height))
        print("Janela de debug ativa em 'vision_debug'.")
    else:
        print(
            "Janela de debug desativada. Use --show em um desktop local "
            "ou acompanhe por --debug-path."
        )
    if args.stream:
        stream_server = StreamServer(host=args.stream_host, port=args.stream_port)
        stream_server.start()
        print(
            f"Stream web ativo em http://127.0.0.1:{args.stream_port} "
            f"(na Raspberry) e http://IP_DA_RASPBERRY:{args.stream_port} (no notebook)."
        )

    last_print = 0.0
    frame_count = 0
    start = time.monotonic()

    try:
        while True:
            frame = ler_frame(camera)
            if frame is None:
                print("Falha ao ler frame da camera.", file=sys.stderr)
                break

            result = analyze_frame(frame, config, state)
            frame_count += 1
            now = time.monotonic()

            if args.print_every <= 0 or (now - last_print) >= args.print_every:
                print_status(result)
                last_print = now

            if args.debug_path:
                cv2.imwrite(args.debug_path, result["debug_frame"])
            if stream_server is not None:
                stream_server.update_frame(result["debug_frame"])

            if should_show:
                cv2.imshow("vision_debug", result["debug_frame"])
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        fechar_camera(camera)
        if stream_server is not None:
            stream_server.stop()
        if should_show:
            cv2.destroyAllWindows()

    elapsed = max(0.001, time.monotonic() - start)
    print(f"Encerrado. Frames={frame_count}, FPS medio={frame_count / elapsed:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

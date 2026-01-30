import argparse
import os
import time

import cv2
import numpy as np

from camera import fechar_camera, iniciar_camera, ler_frame
from serial_comm import abrir_serial, enviar_serial, fechar_serial


def tem_display():
    if os.name == "nt":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def criar_mascara_linha(frame, roi_ratio, threshold, invert):
    height, width = frame.shape[:2]
    roi_height = max(1, int(height * roi_ratio))
    y_start = height - roi_height
    roi = frame[y_start:height, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    if threshold is None:
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)

    if invert:
        mask = cv2.bitwise_not(mask)

    return mask, y_start, roi_height


def encontrar_linha(frame, roi_ratio, threshold, invert, min_area):
    height, width = frame.shape[:2]
    mask, y_start, roi_height = criar_mascara_linha(frame, roi_ratio, threshold, invert)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    debug = frame.copy()

    if not contours:
        return None, 0.0, debug

    largest = max(contours, key=cv2.contourArea)
    area = int(cv2.contourArea(largest))
    if area < min_area:
        return None, 0.0, debug

    m = cv2.moments(largest)
    if m["m00"] == 0:
        return None, 0.0, debug

    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"]) + y_start

    cv2.drawContours(debug[y_start:height, :], [largest], -1, (0, 255, 0), 2)
    cv2.circle(debug, (cx, cy), 5, (0, 0, 255), -1)
    cv2.line(debug, (width // 2, height), (width // 2, height - roi_height), (255, 0, 0), 2)

    confidence = min(1.0, area / float(roi_height * width))
    return (cx, cy), confidence, debug, mask, y_start, roi_height


def detectar_intersecao(mask, largura_min_ratio=0.55, linhas_teste=3):
    altura, largura = mask.shape[:2]
    if altura < 5:
        return False

    # verifica se a linha ocupa uma largura grande em algumas linhas da ROI
    amostras = []
    for i in range(linhas_teste):
        y = int((i + 1) * altura / (linhas_teste + 1))
        linha = mask[y : y + 1, :]
        pixels = int(np.count_nonzero(linha))
        amostras.append(pixels)

    limite = int(largura * largura_min_ratio)
    return any(p > limite for p in amostras)


def criar_mascara_verde(frame, y_start, roi_height, hmin, hmax, smin, vmin):
    roi = frame[y_start : y_start + roi_height, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array([hmin, smin, vmin])
    upper = np.array([hmax, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def detectar_marcadores_verdes(mask_verde, area_min, y_start):
    contours, _ = cv2.findContours(mask_verde, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centros = []
    for cnt in contours:
        area = int(cv2.contourArea(cnt))
        if area < area_min:
            continue
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"]) + y_start
        centros.append((cx, cy, area))
    return centros


def calcular_offset(cx, width):
    center = width / 2.0
    return (cx - center) / center


def main():
    parser = argparse.ArgumentParser(description="Seguidor de linha (visao) - Raspberry Pi")
    parser.add_argument("--device", type=int, default=0, help="Indice da camera (webcam).")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--roi", type=float, default=0.35, help="Porcentagem inferior da imagem")
    parser.add_argument("--threshold", type=int, default=None, help="0-255. Se omitido, usa Otsu.")
    parser.add_argument("--invert", action="store_true", help="Inverte a mascara (linha preta).")
    parser.add_argument("--min-area", type=int, default=600, help="Area minima do contorno da linha.")
    parser.add_argument("--gap-tempo", type=float, default=0.35, help="Tempo maximo sem linha (gap).")
    parser.add_argument("--intersecao-largura", type=float, default=0.55, help="Razao para detectar intersecao.")
    parser.add_argument("--giro-180-tempo", type=float, default=1.2, help="Tempo de giro 180 (seg).")
    parser.add_argument("--giro-180-offset", type=float, default=1.0, help="Offset usado no giro 180.")
    parser.add_argument("--beco-cooldown", type=float, default=2.0, help="Intervalo minimo entre beco sem saida.")
    parser.add_argument("--verde-hmin", type=int, default=40)
    parser.add_argument("--verde-hmax", type=int, default=90)
    parser.add_argument("--verde-smin", type=int, default=50)
    parser.add_argument("--verde-vmin", type=int, default=50)
    parser.add_argument("--verde-area-min", type=int, default=200)
    parser.add_argument("--verde-zona", type=float, default=0.45, help="Porcentagem inferior da ROI.")
    parser.add_argument("--port", type=str, default=None, help="Porta serial (ex: /dev/ttyUSB0).")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--no-picamera2", action="store_true", help="Forca usar OpenCV/V4L2.")
    parser.add_argument("--show", action="store_true", help="Mostra janela (requer display).")
    parser.add_argument("--debug-path", type=str, default=None, help="Salva frame debug (jpg).")
    parser.add_argument("--debug-interval", type=float, default=1.0, help="Intervalo em segundos.")
    args = parser.parse_args()

    cam = iniciar_camera(
        device=args.device,
        width=args.width,
        height=args.height,
        framerate=args.fps,
        prefer_picamera2=not args.no_picamera2,
    )
    ser = abrir_serial(port=args.port, baud=args.baud)

    last_debug = 0.0
    ultimo_offset = 0.0
    ultimo_tempo_linha = time.time()
    giro_ativo = False
    inicio_giro = 0.0
    ultimo_beco = -999.0
    show = args.show and tem_display()

    try:
        while True:
            frame = ler_frame(cam)
            if frame is None:
                time.sleep(0.01)
                continue

            point, confidence, debug, mask_linha, y_start, roi_height = encontrar_linha(
                frame=frame,
                roi_ratio=args.roi,
                threshold=args.threshold,
                invert=args.invert,
                min_area=args.min_area,
            )

            estado = "linha"
            offset = 0.0
            if point is not None:
                offset = calcular_offset(point[0], frame.shape[1])
                ultimo_offset = offset
                ultimo_tempo_linha = time.time()
            else:
                tempo_sem_linha = time.time() - ultimo_tempo_linha
                if tempo_sem_linha <= args.gap_tempo:
                    estado = "gap"
                    offset = ultimo_offset
                else:
                    estado = "linha_perdida"
                    offset = 0.0
                    confidence = 0.0

            intersecao = False
            decisao = "reto"
            tem_verde = False
            tem_verde_fora = False
            verde_falso = False
            beco_sem_saida = False
            direcao_verde = "centro"
            marcadores = []
            marcadores_fora = []
            zona_y = None

            if point is not None:
                intersecao = detectar_intersecao(
                    mask_linha, largura_min_ratio=args.intersecao_largura
                )
                mask_verde = criar_mascara_verde(
                    frame,
                    y_start,
                    roi_height,
                    args.verde_hmin,
                    args.verde_hmax,
                    args.verde_smin,
                    args.verde_vmin,
                )
                marcadores_todos = detectar_marcadores_verdes(
                    mask_verde, args.verde_area_min, y_start
                )
                zona_y = y_start + int(roi_height * (1.0 - args.verde_zona))
                marcadores = [m for m in marcadores_todos if m[1] >= zona_y]
                marcadores_fora = [m for m in marcadores_todos if m[1] < zona_y]

                tem_verde = len(marcadores) > 0
                tem_verde_fora = len(marcadores_fora) > 0
                beco_sem_saida = intersecao and len(marcadores) >= 2

                if tem_verde:
                    media_x = sum(m[0] for m in marcadores) / float(len(marcadores))
                    if media_x < (frame.shape[1] / 2.0):
                        direcao_verde = "esquerda"
                    else:
                        direcao_verde = "direita"

                if intersecao and tem_verde:
                    if beco_sem_saida:
                        decisao = "oposto_" + direcao_verde
                    else:
                        decisao = direcao_verde
                elif intersecao:
                    decisao = "reto"
                elif tem_verde:
                    verde_falso = True
                elif tem_verde_fora:
                    # Verde depois da intersecao: deve ser ignorado
                    verde_falso = True

            agora = time.time()
            if giro_ativo:
                if (agora - inicio_giro) <= args.giro_180_tempo:
                    estado = "giro_180_dir"
                    offset = args.giro_180_offset
                    confidence = 0.0
                    decisao = "giro_180_dir"
                else:
                    giro_ativo = False

            if (
                not giro_ativo
                and beco_sem_saida
                and (agora - ultimo_beco) >= args.beco_cooldown
            ):
                giro_ativo = True
                inicio_giro = agora
                ultimo_beco = agora
                estado = "giro_180_dir"
                offset = args.giro_180_offset
                confidence = 0.0
                decisao = "giro_180_dir"

            if point is not None:
                print(
                    f"offset={offset:.3f} confidence={confidence:.2f} estado={estado} "
                    f"intersecao={intersecao} decisao={decisao}"
                )
            else:
                print(f"estado={estado}")

            enviar_serial(ser, offset=offset, confidence=confidence)

            if marcadores or marcadores_fora:
                for cx, cy, _ in marcadores:
                    cor = (0, 165, 255) if beco_sem_saida else (0, 255, 0)
                    cv2.circle(debug, (cx, cy), 6, cor, 2)
                for cx, cy, _ in marcadores_fora:
                    cv2.circle(debug, (cx, cy), 6, (0, 0, 255), 2)
                if zona_y is not None:
                    cv2.line(debug, (0, zona_y), (frame.shape[1], zona_y), (0, 255, 255), 1)

            texto = f"{estado} | intersecao={intersecao} | {decisao}"
            if verde_falso:
                texto += " | verde_falso"
            if beco_sem_saida:
                texto += " | beco_sem_saida"
            cv2.putText(
                debug,
                texto,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if args.debug_path and (time.time() - last_debug) >= args.debug_interval:
                cv2.imwrite(args.debug_path, debug)
                last_debug = time.time()

            if show:
                cv2.imshow("line-follow", debug)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        fechar_camera(cam)
        fechar_serial(ser)
        if show:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

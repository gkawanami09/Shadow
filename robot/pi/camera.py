import glob
import os
import time

import cv2


CANDIDATOS_DISPOSITIVO_PADRAO = (0, 1, 2)
TENTATIVAS_LEITURA_INICIAL = 10
INTERVALO_LEITURA_INICIAL_S = 0.05


def _candidatos_backend_opencv():
    candidatos = [("auto", cv2.CAP_ANY)]

    # Em Linux, CAP_V4L2 costuma ser mais estavel para webcams USB.
    if os.name == "posix" and hasattr(cv2, "CAP_V4L2"):
        candidatos.insert(0, ("v4l2", cv2.CAP_V4L2))

    # Em Windows, CAP_DSHOW geralmente evita atrasos na abertura.
    if os.name == "nt" and hasattr(cv2, "CAP_DSHOW"):
        candidatos.insert(0, ("dshow", cv2.CAP_DSHOW))

    vistos = set()
    unicos = []
    for nome, valor in candidatos:
        if valor in vistos:
            continue
        vistos.add(valor)
        unicos.append((nome, valor))
    return unicos


def _listar_indices_video_linux():
    if os.name != "posix":
        return []

    encontrados = []
    for caminho in sorted(glob.glob("/dev/video*")):
        sufixo = caminho.replace("/dev/video", "", 1)
        if sufixo.isdigit():
            encontrados.append(int(sufixo))
    return encontrados


def _montar_candidatos_dispositivo(device, device_candidates):
    if device is not None:
        return [int(device)]

    candidatos = []
    vistos = set()

    for indice in _listar_indices_video_linux():
        if indice not in vistos:
            candidatos.append(indice)
            vistos.add(indice)

    for indice in device_candidates:
        indice = int(indice)
        if indice not in vistos:
            candidatos.append(indice)
            vistos.add(indice)

    return candidatos


def _abrir_dispositivo_opencv(indice_dispositivo, largura, altura, taxa_quadros):
    erros_backend = []
    for nome_backend, backend in _candidatos_backend_opencv():
        captura = cv2.VideoCapture(indice_dispositivo, backend)
        if not captura.isOpened():
            captura.release()
            erros_backend.append(f"{nome_backend}: nao abriu")
            continue

        captura.set(cv2.CAP_PROP_FRAME_WIDTH, int(largura))
        captura.set(cv2.CAP_PROP_FRAME_HEIGHT, int(altura))
        captura.set(cv2.CAP_PROP_FPS, int(taxa_quadros))

        quadro_valido = None
        for _ in range(TENTATIVAS_LEITURA_INICIAL):
            ok, quadro = captura.read()
            if ok and quadro is not None and quadro.size > 0:
                quadro_valido = quadro
                break
            time.sleep(INTERVALO_LEITURA_INICIAL_S)

        if quadro_valido is not None:
            return captura, None

        captura.release()
        erros_backend.append(f"{nome_backend}: abriu mas nao retornou quadro valido")

    if not erros_backend:
        return None, "nao abriu"
    return None, " | ".join(erros_backend)


def _abrir_picamera2(largura, altura):
    try:
        from picamera2 import Picamera2

        camera_pi = Picamera2()
        configuracao = camera_pi.create_preview_configuration(
            main={"size": (int(largura), int(altura)), "format": "RGB888"}
        )
        camera_pi.configure(configuracao)
        camera_pi.start()
        time.sleep(0.20)

        quadro = camera_pi.capture_array()
        if quadro is None or quadro.size == 0:
            camera_pi.stop()
            camera_pi.close()
            return None, "picamera2 abriu mas nao retornou quadro valido"

        return camera_pi, None
    except ModuleNotFoundError as excecao:
        if excecao.name == "picamera2":
            return (
                None,
                "modulo picamera2 ausente (instale com: sudo apt install -y python3-picamera2)",
            )
        return None, str(excecao)
    except Exception as excecao:
        return None, str(excecao)


def iniciar_camera(
    device=None,
    width=640,
    height=480,
    framerate=30,
    prefer_picamera2=False,
    fallback_picamera2=True,
    device_candidates=CANDIDATOS_DISPOSITIVO_PADRAO,
):
    width = int(width)
    height = int(height)
    framerate = int(framerate)
    if width <= 0 or height <= 0:
        raise RuntimeError("Resolucao invalida. Use valores positivos para --width e --height.")

    estado_camera = {
        "largura": width,
        "altura": height,
        "captura": None,
        "picamera2": None,
        "backend": None,
        "device": None,
    }

    tentativas = []
    candidatos = _montar_candidatos_dispositivo(device, device_candidates)

    if not prefer_picamera2:
        for candidato in candidatos:
            captura, erro = _abrir_dispositivo_opencv(candidato, width, height, framerate)
            if captura is not None:
                estado_camera["captura"] = captura
                estado_camera["backend"] = "opencv"
                estado_camera["device"] = candidato
                return estado_camera
            tentativas.append(f"opencv({candidato}): {erro}")

    if prefer_picamera2 or fallback_picamera2:
        camera_pi, erro = _abrir_picamera2(width, height)
        if camera_pi is not None:
            estado_camera["picamera2"] = camera_pi
            estado_camera["backend"] = "picamera2"
            estado_camera["device"] = "picamera2"
            return estado_camera
        tentativas.append(f"picamera2: {erro}")

    if prefer_picamera2:
        for candidato in candidatos:
            captura, erro = _abrir_dispositivo_opencv(candidato, width, height, framerate)
            if captura is not None:
                estado_camera["captura"] = captura
                estado_camera["backend"] = "opencv"
                estado_camera["device"] = candidato
                return estado_camera
            tentativas.append(f"opencv({candidato}): {erro}")

    detalhes = "; ".join(tentativas) if tentativas else "nenhuma tentativa executada"
    sugestoes = []
    if os.name == "posix" and "picamera2" in detalhes:
        sugestoes.append("se usar camera CSI, instale python3-picamera2 no sistema")
    if os.name == "posix":
        sugestoes.append("se usar USB, confira permissoes e existencia de /dev/video*")
    sugestoes.append("teste tambem com --device 0 (ou outro indice)")

    detalhes_sugestao = " | ".join(sugestoes)
    raise RuntimeError(
        "Nao foi possivel abrir a camera. "
        f"Tentativas: {detalhes}. "
        f"Sugestoes: {detalhes_sugestao}"
    )


def ler_frame(estado_camera):
    camera_pi = estado_camera.get("picamera2")
    if camera_pi is not None:
        quadro_rgb = camera_pi.capture_array()
        if quadro_rgb is None or quadro_rgb.size == 0:
            return None
        return cv2.cvtColor(quadro_rgb, cv2.COLOR_RGB2BGR)

    captura = estado_camera.get("captura")
    if captura is None:
        return None

    ok, quadro = captura.read()
    if not ok or quadro is None or quadro.size == 0:
        return None
    return quadro


def fechar_camera(estado_camera):
    camera_pi = estado_camera.get("picamera2")
    if camera_pi is not None:
        camera_pi.stop()
        camera_pi.close()
        estado_camera["picamera2"] = None

    captura = estado_camera.get("captura")
    if captura is not None:
        captura.release()
        estado_camera["captura"] = None

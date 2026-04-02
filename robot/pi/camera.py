import time

import cv2


CANDIDATOS_DISPOSITIVO_PADRAO = (0, 1, 2)


def _abrir_dispositivo_opencv(indice_dispositivo, largura, altura, taxa_quadros):
    captura = cv2.VideoCapture(indice_dispositivo)
    if not captura.isOpened():
        captura.release()
        return None, "nao abriu"

    captura.set(cv2.CAP_PROP_FRAME_WIDTH, int(largura))
    captura.set(cv2.CAP_PROP_FRAME_HEIGHT, int(altura))
    captura.set(cv2.CAP_PROP_FPS, int(taxa_quadros))

    ok, quadro = captura.read()
    if not ok or quadro is None or quadro.size == 0:
        captura.release()
        return None, "abriu mas nao retornou quadro valido"

    return captura, None


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
    estado_camera = {
        "largura": int(width),
        "altura": int(height),
        "captura": None,
        "picamera2": None,
        "backend": None,
        "device": None,
    }

    tentativas = []
    candidatos = [int(device)] if device is not None else list(device_candidates)

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
    raise RuntimeError(
        "Nao foi possivel abrir a camera. "
        f"Tentativas: {detalhes}"
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

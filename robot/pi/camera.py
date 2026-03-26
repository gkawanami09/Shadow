import time

import cv2


DEFAULT_DEVICE_CANDIDATES = (0, 1, 2)


def _abrir_opencv_device(device, width, height, framerate):
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        cap.release()
        return None, "nao abriu"

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, framerate)

    ok, frame = cap.read()
    if not ok or frame is None or frame.size == 0:
        cap.release()
        return None, "abriu mas nao entregou frame valido"

    return cap, None


def _abrir_picamera2(width, height):
    try:
        from picamera2 import Picamera2

        picam = Picamera2()
        config = picam.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        picam.configure(config)
        picam.start()
        time.sleep(0.2)
        frame = picam.capture_array()
        if frame is None or frame.size == 0:
            picam.stop()
            picam.close()
            return None, "Picamera2 abriu mas nao entregou frame valido"
        return picam, None
    except Exception as exc:
        return None, str(exc)


def iniciar_camera(
    device=None,
    width=640,
    height=480,
    framerate=30,
    prefer_picamera2=False,
    fallback_picamera2=True,
    device_candidates=DEFAULT_DEVICE_CANDIDATES,
):
    state = {
        "width": width,
        "height": height,
        "picamera2": None,
        "cap": None,
        "backend": None,
        "device": None,
    }

    tentativas = []
    devices = [device] if device is not None else list(device_candidates)

    if not prefer_picamera2:
        for candidate in devices:
            cap, erro = _abrir_opencv_device(candidate, width, height, framerate)
            if cap is not None:
                state["cap"] = cap
                state["backend"] = "opencv"
                state["device"] = candidate
                return state
            tentativas.append(f"OpenCV device {candidate}: {erro}")

    if prefer_picamera2 or fallback_picamera2:
        picam, erro = _abrir_picamera2(width, height)
        if picam is not None:
            state["picamera2"] = picam
            state["backend"] = "picamera2"
            state["device"] = "picamera2"
            return state
        tentativas.append(f"Picamera2: {erro}")

    if prefer_picamera2:
        for candidate in devices:
            cap, erro = _abrir_opencv_device(candidate, width, height, framerate)
            if cap is not None:
                state["cap"] = cap
                state["backend"] = "opencv"
                state["device"] = candidate
                return state
            tentativas.append(f"OpenCV device {candidate}: {erro}")

    detalhe = "; ".join(tentativas) if tentativas else "nenhuma tentativa executada"
    raise RuntimeError(
        "Nao foi possivel abrir uma camera valida. "
        f"Tentativas: {detalhe}"
    )


def ler_frame(state):
    picam = state.get("picamera2")
    if picam is not None:
        frame = picam.capture_array()
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cap = state.get("cap")
    if cap is None:
        return None
    ok, frame = cap.read()
    if not ok:
        return None
    return frame


def fechar_camera(state):
    picam = state.get("picamera2")
    if picam is not None:
        picam.stop()
        picam.close()
        state["picamera2"] = None

    cap = state.get("cap")
    if cap is not None:
        cap.release()
        state["cap"] = None

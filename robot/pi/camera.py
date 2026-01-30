import time

import cv2


def iniciar_camera(
    device=0,
    width=640,
    height=480,
    framerate=30,
    prefer_picamera2=True,
):
    state = {
        "width": width,
        "height": height,
        "picamera2": None,
        "cap": None,
    }

    if prefer_picamera2:
        try:
            from picamera2 import Picamera2

            picam = Picamera2()
            config = picam.create_preview_configuration(
                main={"size": (width, height), "format": "RGB888"}
            )
            picam.configure(config)
            picam.start()
            time.sleep(0.2)
            state["picamera2"] = picam
        except Exception:
            state["picamera2"] = None

    if state["picamera2"] is None:
        cap = cv2.VideoCapture(device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, framerate)
        if not cap.isOpened():
            raise RuntimeError("Nao foi possivel abrir a camera via OpenCV.")
        state["cap"] = cap

    return state


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

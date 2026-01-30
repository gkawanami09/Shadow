try:
    import serial
except Exception:  # pragma: no cover
    serial = None


def abrir_serial(port, baud=115200):
    if port is None or serial is None:
        return None
    return serial.Serial(port=port, baudrate=baud, timeout=0.1)


def enviar_serial(ser, offset, confidence):
    if ser is None:
        return
    payload = f"{offset:.3f},{confidence:.3f}\n"
    ser.write(payload.encode("ascii"))


def fechar_serial(ser):
    if ser is not None:
        ser.close()

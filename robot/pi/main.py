import time
from serial_comm import abrir_serial, enviar_serial, fechar_serial

PORTA = "/dev/ttyACM0"
BAUD = 115200

def main():
    ser = abrir_serial(PORTA, BAUD)

    if ser is None:
        print("Nao foi possivel abrir a serial.")
        return

    time.sleep(2)

    print("Enviando PING...")
    enviar_serial(ser, "PING")
    time.sleep(1)

    while ser.in_waiting:
        print(ser.readline().decode(errors="ignore").strip())

    print("Ligando LED...")
    enviar_serial(ser, "LED_ON")
    time.sleep(2)

    while ser.in_waiting:
        print(ser.readline().decode(errors="ignore").strip())

    print("Desligando LED...")
    enviar_serial(ser, "LED_OFF")
    time.sleep(2)

    while ser.in_waiting:
        print(ser.readline().decode(errors="ignore").strip())

    fechar_serial(ser)

if __name__ == "__main__":
    main()
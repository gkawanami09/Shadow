Raspberry Pi 4 + Camera Rev 1.3 (visao para seguidor de linha)

Este modulo roda no Raspberry Pi e estima o deslocamento da linha para enviar ao Arduino.

Requisitos (Raspberry Pi OS atual):
- Habilitar a camera no raspi-config (Interface Options > Camera)
- Pacotes do sistema:
  - python3, python3-venv
  - python3-opencv, python3-numpy, python3-serial
  - python3-picamera2 (se estiver usando a camera oficial CSI)

Instalacao sugerida:
1) (Opcional) crie um venv:
   python3 -m venv .venv
   source .venv/bin/activate
2) Instale dependencias Python:
   pip install -r requirements.txt
3) Se o picamera2 nao estiver disponivel via pip, instale via apt:
   sudo apt install python3-picamera2

Como rodar (modo headless via SSH):
python3 main.py --debug-path /tmp/line_debug.jpg --invert

Argumentos uteis:
- --invert: use quando a linha for preta em fundo claro
- --roi 0.35: ajusta a faixa inferior da imagem usada para detectar a linha
- --threshold 120: fixa o limiar em vez de usar Otsu
- --show: abre janela (somente se houver display)
- --device 0: escolhe a camera (webcam)
- --gap-tempo 0.35: tempo maximo sem linha para considerar gap
- --intersecao-largura 0.55: ajuste da deteccao de intersecoes
- --giro-180-tempo 1.2: tempo de giro quando detectar beco sem saida
- --giro-180-offset 1.0: offset usado no giro (direita)
- --beco-cooldown 2.0: intervalo minimo entre beco sem saida
- --verde-hmin/--verde-hmax: faixa de cor verde no HSV
- --verde-smin/--verde-vmin: saturacao/valor minimos para verde
- --verde-area-min: area minima do marcador verde
- --verde-zona 0.45: parte inferior da ROI onde o verde vale
- --port /dev/ttyUSB0: envia offset/confirmacao via serial

Saidas:
- Imprime no console: offset (entre -1 e 1) e confidence (0-1)
- Se --debug-path for usado, salva um jpg com a deteccao desenhada

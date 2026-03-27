Raspberry Pi 4 para seguidor de linha (visao e controle)

Este modulo roda no Raspberry Pi e pode operar em dois modos:
- main.py: visao somente (debug, sem serial)
- control.py: visao + envio de comandos seriais para o Arduino

Requisitos (Raspberry Pi OS atual):
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

Como rodar (visao somente):
python3 main.py --debug-path /tmp/line_debug.jpg $(cat run_args.txt)

Como rodar (controle + Arduino):
python3 control.py --port /dev/ttyACM0 --baud 115200 $(cat run_args.txt)

Como rodar por SSH vendo no navegador:
python3 main.py --stream --debug-path /tmp/line_debug.jpg $(cat run_args.txt)

Depois, no notebook, abra:
http://IP_DA_RASPBERRY:8080

Argumentos uteis:
- --invert: use quando a linha for preta em fundo claro
- --roi 0.35: ajusta a faixa inferior da imagem usada para detectar a linha
- --threshold 120: fixa o limiar em vez de usar Otsu
- --show: abre janela (somente se houver display)
- --stream: publica o frame de debug em MJPEG para acesso pelo navegador
- --stream-host 0.0.0.0: host do servidor HTTP
- --stream-port 8080: porta do servidor HTTP
- --device 0: forca um indice especifico de camera
- --gap-tempo 0.35: tempo maximo sem linha para considerar gap
- --intersecao-largura 0.55: ajuste da deteccao de intersecoes
- --intersecao-lado-min 0.20: minimo de linha em ambos os lados para intersecao completa
- --giro-180-tempo 1.2: tempo de giro quando detectar beco sem saida
- --giro-180-offset 1.0: offset usado no giro (direita)
- --beco-cooldown 2.0: intervalo minimo entre beco sem saida
- --verde-hmin/--verde-hmax: faixa de cor verde no HSV
- --verde-smin/--verde-vmin: saturacao/valor minimos para verde
- --verde-area-min: area minima do marcador verde
- --verde-zona 0.45: parte inferior da ROI onde o verde vale
- --vermelho-*: faixa e area para detectar vermelho
- --vermelho-tempo-parado 20: tempo parado ao detectar vermelho
- --velocidade-reto/--velocidade-giro/--velocidade-u: velocidades do controle (control.py)
- --comando-intervalo 0.1: intervalo minimo para reenviar comando (control.py)
- sem --device, a webcam USB e autodetectada tentando 0, depois 1, depois 2
- --show: abre janela de debug local
- --print-every 0.2: controla a frequencia de logs no terminal

Saidas:
- Imprime no console: estado, offset, confidence, intersecao, decisao visual e status do verde
- Se --debug-path for usado, salva um jpg com a deteccao desenhada
- Se --stream for usado, disponibiliza a visualizacao remota no navegador

Raspberry Pi 4 em modo visao somente para seguidor de linha

Este modulo roda no Raspberry Pi, abre a camera localmente e classifica a cena sem depender de Arduino ou serial.

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

Como rodar:
python3 main.py --debug-path /tmp/line_debug.jpg --invert

Como rodar por SSH vendo no navegador:
python3 main.py --stream --debug-path /tmp/line_debug.jpg

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
- --giro-180-tempo 1.2: tempo de giro quando detectar beco sem saida
- --giro-180-offset 1.0: offset usado no giro (direita)
- --beco-cooldown 2.0: intervalo minimo entre beco sem saida
- --verde-hmin/--verde-hmax: faixa de cor verde no HSV
- --verde-smin/--verde-vmin: saturacao/valor minimos para verde
- --verde-area-min: area minima do marcador verde
- --verde-zona 0.45: parte inferior da ROI onde o verde vale
- sem --device, a webcam USB e autodetectada tentando 0, depois 1, depois 2
- --show: abre janela de debug local
- --print-every 0.2: controla a frequencia de logs no terminal

Saidas:
- Imprime no console: estado, offset, confidence, intersecao, decisao visual e status do verde
- Se --debug-path for usado, salva um jpg com a deteccao desenhada
- Se --stream for usado, disponibiliza a visualizacao remota no navegador

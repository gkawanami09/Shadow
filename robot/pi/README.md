Raspberry Pi 4 para seguidor de linha simplificado

Este modulo do Pi foi enxugado para manter apenas:
- correcao de linha
- deteccao e execucao de curva de 90 graus
- deteccao de verde

Arquivos principais:
- `main.py`: visao e debug
- `control.py`: visao + envio de comandos seriais para o Arduino
- `vision.py`: linha, lookahead, curva de 90 e verde

Requisitos:
- `python3`
- `opencv-python`
- `numpy`
- `pyserial`
- `python3-picamera2` se estiver usando camera CSI no Raspberry Pi OS

Como rodar visao:
`python3 main.py --debug-path /tmp/line_debug.jpg $(cat run_args.txt)`

Como rodar controle:
`python3 control.py --port /dev/ttyACM0 --baud 115200 $(cat run_args.txt)`

Como abrir stream no navegador:
`python3 main.py --stream --debug-path /tmp/line_debug.jpg $(cat run_args.txt)`

Depois abra:
`http://IP_DA_RASPBERRY:8080`

Argumentos mais uteis:
- `--roi 0.40`: ajusta a faixa inferior da imagem usada para a linha
- `--limiar-binario 120`: fixa o limiar em vez de usar Otsu
- `--invert`: usa linha clara em fundo escuro
- `--lookahead-fracao 0.42`: ponto antecipado para leitura da curva
- `--velocidade-base` e `--velocidade-curva`: velocidades da correcao
- `--velocidade-giro-90`: velocidade do comando de 90 graus
- `--tempo-giro-90`: tempo da manobra de 90 graus
- `--stream`: publica o debug via MJPEG
- `--device 0`: força um indice especifico de camera

Saidas:
- console com estado, erro, lookahead, curva de 90 e verde
- quadro de debug em janela, arquivo ou stream

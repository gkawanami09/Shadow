import argparse
import os
import sys
import time
from dataclasses import dataclass, field

try:
    import cv2
except ModuleNotFoundError as excecao:
    if excecao.name == "cv2":
        print(
            "Dependencia ausente: OpenCV (cv2).\n"
            "Instale com um destes comandos e tente novamente:\n"
            "  sudo apt update && sudo apt install -y python3-opencv python3-numpy\n"
            "ou\n"
            "  python3 -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(1)
    raise

from camera import fechar_camera, iniciar_camera, ler_frame
from serial_comm import abrir_serial, enviar_velocidades_diferenciais, fechar_serial, parar
from stream import ServidorStream
from vision import ConfiguracaoVisao, EstadoVisao, analisar_quadro


ESTADO_INICIANDO = "INICIANDO"
ESTADO_SEGUINDO = "SEGUINDO_LINHA"
ESTADO_SEM_LINHA = "SEM_LINHA"


@dataclass
class EstadoControle:
    estado_atual: str = ESTADO_INICIANDO
    tempo_entrada_estado: float = field(default_factory=time.monotonic)
    assinatura_ultima_acao: tuple | None = None
    instante_ultimo_envio: float = 0.0
    velocidade_esquerda_anterior: int = 0
    velocidade_direita_anterior: int = 0


class ControladorPID:
    def __init__(self, kP, kI, kD, limite_integral, dt_minimo, alpha_derivada):
        self.kP = float(kP)
        self.kI = float(kI)
        self.kD = float(kD)
        self.limite_integral = float(abs(limite_integral))
        self.dt_minimo = float(max(1e-4, dt_minimo))
        self.alpha_derivada = float(_limitar(alpha_derivada, 0.0, 1.0))

        self.erro_proporcional = 0.0
        self.erro_integral = 0.0
        self.erro_derivativo = 0.0
        self.erro_anterior = 0.0
        self.tempo_anterior = None
        self._derivada_filtrada = 0.0

    def reiniciar(self, suave=False):
        self.erro_proporcional = 0.0
        self.erro_derivativo = 0.0
        if suave:
            self.erro_integral *= 0.25
        else:
            self.erro_integral = 0.0
            self._derivada_filtrada = 0.0
            self.erro_anterior = 0.0
            self.tempo_anterior = None

    def calcular(self, erro_atual, tempo_atual):
        self.erro_proporcional = float(erro_atual)

        if self.tempo_anterior is None:
            dt = self.dt_minimo
        else:
            dt = max(self.dt_minimo, float(tempo_atual - self.tempo_anterior))

        self.erro_integral += self.erro_proporcional * dt
        self.erro_integral = _limitar(self.erro_integral, -self.limite_integral, self.limite_integral)

        if dt <= (self.dt_minimo * 1.01):
            derivada_bruta = 0.0
        else:
            derivada_bruta = (self.erro_proporcional - self.erro_anterior) / dt

        self._derivada_filtrada = (
            self.alpha_derivada * self._derivada_filtrada
            + (1.0 - self.alpha_derivada) * derivada_bruta
        )
        self.erro_derivativo = self._derivada_filtrada

        saida = (
            self.kP * self.erro_proporcional
            + self.kI * self.erro_integral
            + self.kD * self.erro_derivativo
        )

        self.erro_anterior = self.erro_proporcional
        self.tempo_anterior = float(tempo_atual)
        return float(saida)


def _limitar(valor, minimo, maximo):
    return max(minimo, min(maximo, valor))


def _limitar_pwm(valor):
    return int(_limitar(int(round(valor)), 0, 255))


def _limitar_pwm_assinado(valor):
    return int(_limitar(int(round(valor)), -255, 255))


def _aplicar_piso_assinado(valor, piso):
    if valor > 0:
        return max(valor, piso)
    if valor < 0:
        return min(valor, -piso)
    return 0


def _criar_acao_parar():
    return {"tipo": "S"}


def _criar_acao_diferencial(velocidade_esquerda, velocidade_direita):
    return {
        "tipo": "D",
        "velocidade_esquerda": _limitar_pwm_assinado(velocidade_esquerda),
        "velocidade_direita": _limitar_pwm_assinado(velocidade_direita),
    }


def _assinatura_acao(acao):
    if acao["tipo"] == "D":
        return ("D", acao["velocidade_esquerda"], acao["velocidade_direita"])
    return ("S",)


def _deve_enviar(estado_controle, acao, agora, intervalo_minimo):
    assinatura = _assinatura_acao(acao)
    if assinatura != estado_controle.assinatura_ultima_acao:
        return True
    return (agora - estado_controle.instante_ultimo_envio) >= intervalo_minimo


def _enviar_acao_serial(ser, acao):
    if ser is None:
        return

    if acao["tipo"] == "D":
        enviar_velocidades_diferenciais(
            ser,
            acao["velocidade_esquerda"],
            acao["velocidade_direita"],
        )
        return

    parar(ser)


def _calcular_acao_pid(dados_visao, pid, parametros, tempo_atual):
    erro_linha = dados_visao["erro_linha"]
    if parametros.inverter_correcao:
        erro_linha = -erro_linha
    correcao_pid = pid.calcular(erro_linha, tempo_atual)
    correcao_pid = _limitar(correcao_pid, -parametros.correcao_maxima, parametros.correcao_maxima)
    erro_abs = abs(erro_linha)

    if erro_abs >= parametros.limiar_erro_pivo:
        velocidade_pivo = _limitar(
            parametros.velocidade_pivo,
            parametros.velocidade_minima,
            parametros.velocidade_maxima,
        )
        if erro_linha >= 0.0:
            velocidade_esquerda = velocidade_pivo
            velocidade_direita = -velocidade_pivo
        else:
            velocidade_esquerda = -velocidade_pivo
            velocidade_direita = velocidade_pivo
        return _criar_acao_diferencial(velocidade_esquerda, velocidade_direita), correcao_pid

    if erro_abs >= parametros.limiar_erro_curva:
        velocidade_base = parametros.velocidade_curva
    else:
        velocidade_base = parametros.velocidade_base

    velocidade_esquerda = _limitar(
        _aplicar_piso_assinado(velocidade_base + correcao_pid, parametros.velocidade_minima),
        -parametros.velocidade_maxima,
        parametros.velocidade_maxima,
    )
    velocidade_direita = _limitar(
        _aplicar_piso_assinado(velocidade_base - correcao_pid, parametros.velocidade_minima),
        -parametros.velocidade_maxima,
        parametros.velocidade_maxima,
    )

    return _criar_acao_diferencial(velocidade_esquerda, velocidade_direita), correcao_pid


def _atualizar_controle(estado_controle, dados_visao, pid, parametros, agora):
    motivo = ""
    correcao_pid = 0.0

    if estado_controle.estado_atual == ESTADO_INICIANDO:
        if (agora - estado_controle.tempo_entrada_estado) < parametros.tempo_inicial:
            return _criar_acao_parar(), "espera de seguranca na partida", correcao_pid
        estado_controle.estado_atual = ESTADO_SEM_LINHA
        estado_controle.tempo_entrada_estado = agora

    linha_valida = bool(
        dados_visao["linha_encontrada"]
        and dados_visao["confianca_linha"] >= parametros.limiar_confianca
    )

    if linha_valida:
        if estado_controle.estado_atual != ESTADO_SEGUINDO:
            estado_controle.estado_atual = ESTADO_SEGUINDO
            estado_controle.tempo_entrada_estado = agora
            pid.reiniciar(suave=True)

        acao, correcao_pid = _calcular_acao_pid(dados_visao, pid, parametros, agora)
        estado_controle.velocidade_esquerda_anterior = acao["velocidade_esquerda"]
        estado_controle.velocidade_direita_anterior = acao["velocidade_direita"]
        motivo = "seguindo linha com PID"
        return acao, motivo, correcao_pid

    if estado_controle.estado_atual != ESTADO_SEM_LINHA:
        estado_controle.estado_atual = ESTADO_SEM_LINHA
        estado_controle.tempo_entrada_estado = agora

    pid.reiniciar(suave=True)
    estado_controle.velocidade_esquerda_anterior = 0
    estado_controle.velocidade_direita_anterior = 0
    motivo = "linha ausente ou confianca baixa"
    return _criar_acao_parar(), motivo, correcao_pid


def _desenhar_info_controle(quadro_debug, estado_controle, dados_visao, acao, correcao_pid, pid, motivo):
    if acao["tipo"] == "D":
        texto_acao = f"D,{acao['velocidade_esquerda']},{acao['velocidade_direita']}"
    else:
        texto_acao = "S"

    textos = [
        f"estado={estado_controle.estado_atual}",
        f"acao={texto_acao}",
        f"motivo={motivo}",
        f"erro={dados_visao['erro_linha']:+.3f} conf={dados_visao['confianca_linha']:.2f}",
        f"pid_p={pid.erro_proporcional:+.3f} pid_i={pid.erro_integral:+.3f} pid_d={pid.erro_derivativo:+.3f}",
        f"correcao_pid={correcao_pid:+.2f}",
        f"tempo_sem_linha={dados_visao['tempo_sem_linha']:.2f}s",
    ]

    for indice, texto in enumerate(textos):
        y_texto = 24 + indice * 21
        cv2.putText(
            quadro_debug,
            texto,
            (12, y_texto),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )


def _imprimir_status(estado_controle, dados_visao, acao, motivo, correcao_pid):
    if acao["tipo"] == "D":
        texto_acao = f"D,{acao['velocidade_esquerda']},{acao['velocidade_direita']}"
    else:
        texto_acao = "S"

    print(
        " | ".join(
            [
                f"estado={estado_controle.estado_atual}",
                f"acao={texto_acao}",
                f"erro={dados_visao['erro_linha']:+.3f}",
                f"conf={dados_visao['confianca_linha']:.2f}",
                f"pid={correcao_pid:+.2f}",
                f"motivo={motivo}",
            ]
        ),
        flush=True,
    )


def analisar_argumentos():
    analisador = argparse.ArgumentParser(
        description="Modo controle: seguidor de linha puro com PID e comando diferencial",
    )

    analisador.add_argument("--device", type=int, default=None, help="Indice da camera USB.")
    analisador.add_argument("--width", type=int, default=640, help="Largura da imagem.")
    analisador.add_argument("--height", type=int, default=480, help="Altura da imagem.")
    analisador.add_argument("--fps", type=int, default=30, help="Taxa de quadros desejada.")

    analisador.add_argument("--show", action="store_true", help="Mostra janela de debug local.")
    analisador.add_argument("--no-show", action="store_true", help="Nao mostra janela local.")
    analisador.add_argument("--debug-path", default=None, help="Salva continuamente o ultimo quadro de debug.")
    analisador.add_argument("--stream", action="store_true", help="Publica stream MJPEG no navegador.")
    analisador.add_argument("--stream-host", default="0.0.0.0", help="Host do stream HTTP.")
    analisador.add_argument("--stream-port", type=int, default=8080, help="Porta do stream HTTP.")
    analisador.add_argument("--print-every", type=float, default=0.20, help="Intervalo minimo para logs.")

    analisador.add_argument("--roi", type=float, default=0.45)
    analisador.add_argument("--limiar-binario", type=int, default=None)
    analisador.add_argument("--invert", action="store_true", help="Inverte polaridade da linha.")
    analisador.add_argument("--area-minima-contorno", type=int, default=180)
    analisador.add_argument("--area-minima-linha", type=int, default=320)
    analisador.add_argument("--suavizacao-erro", type=float, default=0.40)
    analisador.add_argument("--limiar-confianca", type=float, default=0.10)

    analisador.add_argument("--kp", type=float, default=145.0)
    analisador.add_argument("--ki", type=float, default=10.0)
    analisador.add_argument("--kd", type=float, default=42.0)
    analisador.add_argument("--integral-max", type=float, default=0.85)
    analisador.add_argument("--dt-minimo", type=float, default=0.01)
    analisador.add_argument("--alpha-derivada", type=float, default=0.25)
    analisador.add_argument("--correcao-maxima", type=float, default=200.0)

    analisador.add_argument("--velocidade-base", type=int, default=75)
    analisador.add_argument("--velocidade-curva", type=int, default=55)
    analisador.add_argument("--velocidade-minima", type=int, default=45)
    analisador.add_argument("--velocidade-maxima", type=int, default=120)
    analisador.add_argument("--limiar-erro-curva", type=float, default=0.12)
    analisador.add_argument("--limiar-erro-pivo", type=float, default=0.52)
    analisador.add_argument("--velocidade-pivo", type=int, default=90)
    analisador.add_argument(
        "--inverter-correcao",
        action="store_true",
        help="Inverte o sentido da correcao lateral (use quando corrige para o lado errado).",
    )

    analisador.add_argument("--tempo-inicial", type=float, default=0.35)

    analisador.add_argument("--port", type=str, default=None, help="Porta serial (ex: /dev/ttyACM0).")
    analisador.add_argument("--baud", type=int, default=115200)
    analisador.add_argument("--comando-intervalo", type=float, default=0.04)

    return analisador.parse_args()


def criar_configuracao_visao(parametros):
    return ConfiguracaoVisao(
        roi=parametros.roi,
        limiar_binario=parametros.limiar_binario,
        inverter_linha=parametros.invert,
        area_minima_contorno=parametros.area_minima_contorno,
        area_minima_linha=parametros.area_minima_linha,
        suavizacao_erro=parametros.suavizacao_erro,
        limiar_confianca=parametros.limiar_confianca,
    )


def principal():
    parametros = analisar_argumentos()

    parametros.velocidade_base = _limitar_pwm(parametros.velocidade_base)
    parametros.velocidade_curva = _limitar_pwm(parametros.velocidade_curva)
    parametros.velocidade_minima = _limitar_pwm(parametros.velocidade_minima)
    parametros.velocidade_maxima = _limitar_pwm(parametros.velocidade_maxima)

    if parametros.velocidade_minima > parametros.velocidade_maxima:
        parametros.velocidade_minima, parametros.velocidade_maxima = (
            parametros.velocidade_maxima,
            parametros.velocidade_minima,
        )

    configuracao_visao = criar_configuracao_visao(parametros)
    estado_visao = EstadoVisao()
    estado_controle = EstadoControle()
    pid = ControladorPID(
        kP=parametros.kp,
        kI=parametros.ki,
        kD=parametros.kd,
        limite_integral=parametros.integral_max,
        dt_minimo=parametros.dt_minimo,
        alpha_derivada=parametros.alpha_derivada,
    )

    estado_controle.tempo_entrada_estado = time.monotonic()

    tem_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    exibir_janela = parametros.show or (tem_display and not parametros.no_show)
    servidor_stream = None
    ser = None

    try:
        priorizar_picamera2 = parametros.device is None
        camera = iniciar_camera(
            device=parametros.device,
            width=parametros.width,
            height=parametros.height,
            framerate=parametros.fps,
            prefer_picamera2=priorizar_picamera2,
            fallback_picamera2=True,
        )
    except RuntimeError as excecao:
        print(f"Erro ao abrir camera: {excecao}", file=sys.stderr)
        return 1

    if parametros.port:
        try:
            ser = abrir_serial(porta=parametros.port, baud=parametros.baud)
        except Exception as excecao:
            ser = None
            print(
                f"Aviso: nao foi possivel abrir serial em {parametros.port}: {excecao}. Rodando apenas visao.",
                file=sys.stderr,
            )

    if ser is not None:
        parar(ser)

    print(
        f"Camera pronta via {camera['backend']} ({camera['device']}) em {parametros.width}x{parametros.height}@{parametros.fps}."
    )
    if ser is None:
        print("Controle em modo visao-only (serial indisponivel).")
    else:
        print(f"Serial ativa em {parametros.port} @ {parametros.baud}.")

    if exibir_janela:
        cv2.namedWindow("controle_debug", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("controle_debug", max(640, parametros.width), max(480, parametros.height))

    if parametros.stream:
        servidor_stream = ServidorStream(host=parametros.stream_host, port=parametros.stream_port)
        servidor_stream.iniciar()
        print(
            f"Stream ativo em http://127.0.0.1:{parametros.stream_port} (local) e http://IP_DA_RASPBERRY:{parametros.stream_port}."
        )

    instante_ultimo_log = 0.0

    try:
        while True:
            quadro_bgr = ler_frame(camera)
            if quadro_bgr is None:
                print("Falha ao ler quadro da camera.", file=sys.stderr)
                break

            dados_visao = analisar_quadro(quadro_bgr, configuracao_visao, estado_visao)
            agora = time.monotonic()

            acao, motivo, correcao_pid = _atualizar_controle(
                estado_controle,
                dados_visao,
                pid,
                parametros,
                agora,
            )

            if _deve_enviar(estado_controle, acao, agora, parametros.comando_intervalo):
                _enviar_acao_serial(ser, acao)
                estado_controle.assinatura_ultima_acao = _assinatura_acao(acao)
                estado_controle.instante_ultimo_envio = agora

            quadro_debug = dados_visao["quadro_debug"].copy()
            _desenhar_info_controle(
                quadro_debug,
                estado_controle,
                dados_visao,
                acao,
                correcao_pid,
                pid,
                motivo,
            )

            if parametros.debug_path:
                cv2.imwrite(parametros.debug_path, quadro_debug)

            if servidor_stream is not None:
                servidor_stream.atualizar_quadro(quadro_debug)

            if exibir_janela:
                cv2.imshow("controle_debug", quadro_debug)
                tecla = cv2.waitKey(1) & 0xFF
                if tecla in (27, ord("q")):
                    break

            if parametros.print_every <= 0 or (agora - instante_ultimo_log) >= parametros.print_every:
                _imprimir_status(estado_controle, dados_visao, acao, motivo, correcao_pid)
                instante_ultimo_log = agora

    except KeyboardInterrupt:
        pass
    finally:
        if ser is not None:
            try:
                parar(ser)
            finally:
                fechar_serial(ser)
        fechar_camera(camera)
        if servidor_stream is not None:
            servidor_stream.parar()
        if exibir_janela:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(principal())

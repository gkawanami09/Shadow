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
from serial_comm import (
    abrir_serial,
    enviar_serial,
    enviar_velocidades_diferenciais,
    fechar_serial,
    parar,
)
from stream import ServidorStream
from vision import ConfiguracaoVisao, EstadoVisao, analisar_quadro


ESTADO_INICIANDO = "INICIANDO"
ESTADO_SEGUINDO = "SEGUINDO_LINHA"
ESTADO_GAP = "GAP"
ESTADO_INTERSECAO_SEM = "INTERSECAO_SEM_MARCACAO"
ESTADO_INTERSECAO_VERDE_ESQ = "INTERSECAO_COM_VERDE_ESQUERDA"
ESTADO_INTERSECAO_VERDE_DIR = "INTERSECAO_COM_VERDE_DIREITA"
ESTADO_BECO = "BECO_SEM_SAIDA"
ESTADO_VERMELHO = "PARADO_VERMELHO"
ESTADO_RECUPERANDO = "RECUPERANDO_LINHA"
ESTADO_SEM_LINHA = "SEM_LINHA"


@dataclass
class EstadoControle:
    estado_atual: str = ESTADO_INICIANDO
    tempo_entrada_estado: float = field(default_factory=time.monotonic)
    tempo_saida_estado: float = 0.0
    tempo_ultima_intersecao: float = 0.0
    tempo_ultimo_verde: float = 0.0
    tempo_ultimo_beco: float = 0.0
    lado_recuperacao: int = 1
    tendencia_gap: float = 0.0
    assinatura_ultima_acao: tuple | None = None
    instante_ultimo_envio: float = 0.0
    finalizado: bool = False


class ControladorPID:
    def __init__(
        self,
        kP,
        kI,
        kD,
        limite_integral,
        dt_minimo,
        alpha_derivada,
    ):
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
            self.erro_integral *= 0.35
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
        self.erro_integral = _limitar(
            self.erro_integral,
            -self.limite_integral,
            self.limite_integral,
        )

        if dt <= self.dt_minimo * 1.01:
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


def _entrar_estado(estado_controle, novo_estado, agora, duracao=0.0):
    estado_controle.estado_atual = novo_estado
    estado_controle.tempo_entrada_estado = agora
    estado_controle.tempo_saida_estado = agora + max(0.0, float(duracao))


def _criar_acao_diferencial(velocidade_esquerda, velocidade_direita):
    return {
        "tipo": "D",
        "velocidade_esquerda": _limitar_pwm(velocidade_esquerda),
        "velocidade_direita": _limitar_pwm(velocidade_direita),
    }


def _criar_acao_comando(comando, velocidade=None):
    acao = {"tipo": "C", "comando": comando}
    if velocidade is not None:
        acao["velocidade"] = _limitar_pwm(velocidade)
    return acao


def _assinatura_acao(acao):
    if acao["tipo"] == "D":
        return ("D", acao["velocidade_esquerda"], acao["velocidade_direita"])
    return ("C", acao["comando"], acao.get("velocidade", -1))


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

    enviar_serial(ser, acao["comando"], acao.get("velocidade"))


def _deve_enviar(estado_controle, acao, agora, intervalo_minimo):
    assinatura = _assinatura_acao(acao)
    if assinatura != estado_controle.assinatura_ultima_acao:
        return True
    return (agora - estado_controle.instante_ultimo_envio) >= intervalo_minimo


def _acao_pid(dados_visao, pid, parametros, agora):
    erro = dados_visao["erro_linha"]
    correcao_pid = pid.calcular(erro, agora)
    correcao_pid = _limitar(correcao_pid, -parametros.correcao_maxima, parametros.correcao_maxima)

    if abs(erro) >= parametros.limiar_erro_curva:
        velocidade_base = parametros.velocidade_curva
    else:
        velocidade_base = parametros.velocidade_base

    velocidade_esquerda = velocidade_base + correcao_pid
    velocidade_direita = velocidade_base - correcao_pid

    acao = _criar_acao_diferencial(velocidade_esquerda, velocidade_direita)
    return acao, correcao_pid


def _acao_intersecao_verde_esquerda(parametros):
    velocidade_esquerda = parametros.velocidade_giro * 0.50
    velocidade_direita = parametros.velocidade_giro
    return _criar_acao_diferencial(velocidade_esquerda, velocidade_direita)


def _acao_intersecao_verde_direita(parametros):
    velocidade_esquerda = parametros.velocidade_giro
    velocidade_direita = parametros.velocidade_giro * 0.50
    return _criar_acao_diferencial(velocidade_esquerda, velocidade_direita)


def _acao_gap(estado_controle, parametros, agora):
    tempo_restante = max(0.0, estado_controle.tempo_saida_estado - agora)
    if parametros.tempo_gap > 1e-6:
        fator = _limitar(tempo_restante / parametros.tempo_gap, 0.25, 1.0)
    else:
        fator = 1.0

    correcao = _limitar(
        estado_controle.tendencia_gap * fator,
        -parametros.correcao_gap_maxima,
        parametros.correcao_gap_maxima,
    )
    return _criar_acao_diferencial(parametros.velocidade_gap + correcao, parametros.velocidade_gap - correcao)


def _acao_recuperacao(estado_controle, parametros, agora):
    duracao_total = max(1e-3, estado_controle.tempo_saida_estado - estado_controle.tempo_entrada_estado)
    progresso = _limitar((agora - estado_controle.tempo_entrada_estado) / duracao_total, 0.0, 1.0)
    lado = estado_controle.lado_recuperacao

    if progresso < 0.45:
        if lado <= 0:
            return _criar_acao_diferencial(parametros.velocidade_recuperacao_leve * 0.70, parametros.velocidade_recuperacao_leve)
        return _criar_acao_diferencial(parametros.velocidade_recuperacao_leve, parametros.velocidade_recuperacao_leve * 0.70)

    if progresso < 0.80:
        if lado <= 0:
            return _criar_acao_diferencial(parametros.velocidade_recuperacao_forte * 0.35, parametros.velocidade_recuperacao_forte)
        return _criar_acao_diferencial(parametros.velocidade_recuperacao_forte, parametros.velocidade_recuperacao_forte * 0.35)

    if progresso < 0.95:
        if lado <= 0:
            return _criar_acao_comando("L", parametros.velocidade_giro)
        return _criar_acao_comando("R", parametros.velocidade_giro)

    return _criar_acao_comando("S")


def _atualizar_estado_controle(estado_controle, dados_visao, pid, parametros, agora):
    motivo = ""
    correcao_pid = 0.0

    if estado_controle.finalizado:
        return _criar_acao_comando("S"), "FINALIZADO", correcao_pid

    if dados_visao["vermelho_confirmado"] and estado_controle.estado_atual != ESTADO_VERMELHO:
        _entrar_estado(estado_controle, ESTADO_VERMELHO, agora, max(5.0, parametros.tempo_vermelho_parado))
        pid.reiniciar(suave=False)

    if estado_controle.estado_atual == ESTADO_VERMELHO:
        motivo = "parada obrigatoria no vermelho"
        if agora >= estado_controle.tempo_saida_estado:
            if parametros.encerrar_apos_vermelho:
                estado_controle.finalizado = True
                return _criar_acao_comando("S"), motivo, correcao_pid
            if not dados_visao["vermelho_confirmado"]:
                _entrar_estado(estado_controle, ESTADO_RECUPERANDO, agora, parametros.tempo_recuperacao)
                pid.reiniciar(suave=True)
        return _criar_acao_comando("S"), motivo, correcao_pid

    if estado_controle.estado_atual == ESTADO_BECO:
        motivo = "giro de 180 em beco sem saida"
        if agora >= estado_controle.tempo_saida_estado:
            _entrar_estado(estado_controle, ESTADO_RECUPERANDO, agora, parametros.tempo_recuperacao)
            pid.reiniciar(suave=False)
        return _criar_acao_comando("U", parametros.velocidade_giro_180), motivo, correcao_pid

    if estado_controle.estado_atual == ESTADO_INTERSECAO_VERDE_ESQ:
        motivo = "intersecao com verde para esquerda"
        if agora >= estado_controle.tempo_saida_estado:
            _entrar_estado(estado_controle, ESTADO_RECUPERANDO, agora, parametros.tempo_recuperacao * 0.7)
        return _acao_intersecao_verde_esquerda(parametros), motivo, correcao_pid

    if estado_controle.estado_atual == ESTADO_INTERSECAO_VERDE_DIR:
        motivo = "intersecao com verde para direita"
        if agora >= estado_controle.tempo_saida_estado:
            _entrar_estado(estado_controle, ESTADO_RECUPERANDO, agora, parametros.tempo_recuperacao * 0.7)
        return _acao_intersecao_verde_direita(parametros), motivo, correcao_pid

    if estado_controle.estado_atual == ESTADO_INTERSECAO_SEM:
        motivo = "intersecao sem marcacao, seguir reto"
        if agora >= estado_controle.tempo_saida_estado:
            if dados_visao["linha_encontrada"]:
                _entrar_estado(estado_controle, ESTADO_SEGUINDO, agora)
            else:
                _entrar_estado(estado_controle, ESTADO_RECUPERANDO, agora, parametros.tempo_recuperacao)
        return _criar_acao_diferencial(parametros.velocidade_curva, parametros.velocidade_curva), motivo, correcao_pid

    if estado_controle.estado_atual == ESTADO_INICIANDO:
        motivo = "partida segura"
        if agora >= estado_controle.tempo_saida_estado:
            _entrar_estado(estado_controle, ESTADO_SEM_LINHA, agora)
        return _criar_acao_comando("S"), motivo, correcao_pid

    # Eventos prioritarios fora dos estados travados.
    passou_cooldown_intersecao = (agora - estado_controle.tempo_ultima_intersecao) >= parametros.cooldown_intersecao
    passou_cooldown_verde = (agora - estado_controle.tempo_ultimo_verde) >= parametros.cooldown_verde
    passou_cooldown_beco = (agora - estado_controle.tempo_ultimo_beco) >= parametros.cooldown_beco

    if (
        dados_visao["intersecao_detectada"]
        and dados_visao["confianca_linha"] >= parametros.limiar_confianca
        and passou_cooldown_intersecao
    ):
        tipo_verde = dados_visao["tipo_verde"]

        if tipo_verde == "VERDE_DUPLO" and passou_cooldown_beco:
            estado_controle.tempo_ultimo_beco = agora
            estado_controle.tempo_ultima_intersecao = agora
            _entrar_estado(estado_controle, ESTADO_BECO, agora, parametros.tempo_giro_180)
            pid.reiniciar(suave=False)
            motivo = "duplo verde antes da intersecao"
            return _criar_acao_comando("U", parametros.velocidade_giro_180), motivo, correcao_pid

        if tipo_verde == "VERDE_ESQUERDA" and passou_cooldown_verde:
            estado_controle.tempo_ultimo_verde = agora
            estado_controle.tempo_ultima_intersecao = agora
            _entrar_estado(estado_controle, ESTADO_INTERSECAO_VERDE_ESQ, agora, parametros.tempo_intersecao_verde)
            pid.reiniciar(suave=True)
            motivo = "verde valido para esquerda"
            return _acao_intersecao_verde_esquerda(parametros), motivo, correcao_pid

        if tipo_verde == "VERDE_DIREITA" and passou_cooldown_verde:
            estado_controle.tempo_ultimo_verde = agora
            estado_controle.tempo_ultima_intersecao = agora
            _entrar_estado(estado_controle, ESTADO_INTERSECAO_VERDE_DIR, agora, parametros.tempo_intersecao_verde)
            pid.reiniciar(suave=True)
            motivo = "verde valido para direita"
            return _acao_intersecao_verde_direita(parametros), motivo, correcao_pid

        estado_controle.tempo_ultima_intersecao = agora
        _entrar_estado(estado_controle, ESTADO_INTERSECAO_SEM, agora, parametros.tempo_intersecao_sem_marcacao)
        pid.reiniciar(suave=True)
        motivo = "intersecao sem verde valido"
        return _criar_acao_diferencial(parametros.velocidade_curva, parametros.velocidade_curva), motivo, correcao_pid

    if not dados_visao["linha_encontrada"]:
        if dados_visao["gap_provavel"] and dados_visao["tempo_sem_linha"] <= parametros.tempo_gap:
            if estado_controle.estado_atual != ESTADO_GAP:
                _entrar_estado(estado_controle, ESTADO_GAP, agora, parametros.tempo_gap)
                estado_controle.tendencia_gap = _limitar(
                    pid.erro_anterior * parametros.ganho_tendencia_gap,
                    -parametros.correcao_gap_maxima,
                    parametros.correcao_gap_maxima,
                )
                pid.reiniciar(suave=True)

            if agora >= estado_controle.tempo_saida_estado:
                _entrar_estado(estado_controle, ESTADO_RECUPERANDO, agora, parametros.tempo_recuperacao)
                estado_controle.lado_recuperacao = dados_visao["lado_ultimo_erro"] or estado_controle.lado_recuperacao
                pid.reiniciar(suave=True)
                motivo = "gap sem reacquisicao"
                return _acao_recuperacao(estado_controle, parametros, agora), motivo, correcao_pid

            motivo = "travessia de gap em linha reta"
            return _acao_gap(estado_controle, parametros, agora), motivo, correcao_pid

        if estado_controle.estado_atual != ESTADO_RECUPERANDO:
            _entrar_estado(estado_controle, ESTADO_RECUPERANDO, agora, parametros.tempo_recuperacao)
            estado_controle.lado_recuperacao = dados_visao["lado_ultimo_erro"] or estado_controle.lado_recuperacao
            pid.reiniciar(suave=True)

        if agora >= estado_controle.tempo_saida_estado:
            _entrar_estado(estado_controle, ESTADO_SEM_LINHA, agora)
            pid.reiniciar(suave=False)
            motivo = "tempo de recuperacao esgotado"
            return _criar_acao_comando("S"), motivo, correcao_pid

        motivo = "recuperacao de linha"
        return _acao_recuperacao(estado_controle, parametros, agora), motivo, correcao_pid

    # Linha encontrada
    if dados_visao["confianca_linha"] < parametros.limiar_confianca:
        _entrar_estado(estado_controle, ESTADO_RECUPERANDO, agora, parametros.tempo_recuperacao * 0.6)
        estado_controle.lado_recuperacao = dados_visao["lado_ultimo_erro"] or estado_controle.lado_recuperacao
        pid.reiniciar(suave=True)
        motivo = "linha com baixa confianca"
        return _acao_recuperacao(estado_controle, parametros, agora), motivo, correcao_pid

    if estado_controle.estado_atual in {ESTADO_GAP, ESTADO_RECUPERANDO, ESTADO_SEM_LINHA}:
        pid.reiniciar(suave=True)

    _entrar_estado(estado_controle, ESTADO_SEGUINDO, agora)
    acao_pid, correcao_pid = _acao_pid(dados_visao, pid, parametros, agora)
    motivo = "seguindo linha com PID"
    return acao_pid, motivo, correcao_pid


def _desenhar_info_controle(quadro_debug, estado_controle, dados_visao, acao, correcao_pid, pid, motivo):
    if acao["tipo"] == "D":
        descricao_acao = f"D,{acao['velocidade_esquerda']},{acao['velocidade_direita']}"
    else:
        if "velocidade" in acao:
            descricao_acao = f"{acao['comando']},{acao['velocidade']}"
        else:
            descricao_acao = acao["comando"]

    linhas = [
        f"estado={estado_controle.estado_atual}",
        f"acao={descricao_acao}",
        f"motivo={motivo}",
        f"erro={dados_visao['erro_linha']:+.3f} conf={dados_visao['confianca_linha']:.2f}",
        f"pid_p={pid.erro_proporcional:+.3f} pid_i={pid.erro_integral:+.3f} pid_d={pid.erro_derivativo:+.3f}",
        f"correcao_pid={correcao_pid:+.2f}",
        f"intersecao={dados_visao['intersecao_detectada']} verde={dados_visao['tipo_verde']}",
        f"vermelho={dados_visao['tipo_vermelho']} confirmado={dados_visao['vermelho_confirmado']}",
        f"gap_provavel={dados_visao['gap_provavel']} tempo_sem_linha={dados_visao['tempo_sem_linha']:.2f}s",
    ]

    for indice, texto in enumerate(linhas):
        y_texto = 25 + indice * 21
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
        acao_texto = f"D,{acao['velocidade_esquerda']},{acao['velocidade_direita']}"
    else:
        if "velocidade" in acao:
            acao_texto = f"{acao['comando']},{acao['velocidade']}"
        else:
            acao_texto = acao["comando"]

    print(
        " | ".join(
            [
                f"estado={estado_controle.estado_atual}",
                f"acao={acao_texto}",
                f"erro={dados_visao['erro_linha']:+.3f}",
                f"conf={dados_visao['confianca_linha']:.2f}",
                f"intersecao={'SIM' if dados_visao['intersecao_detectada'] else 'NAO'}",
                f"verde={dados_visao['tipo_verde']}",
                f"vermelho={dados_visao['tipo_vermelho']}",
                f"pid={correcao_pid:+.2f}",
                f"motivo={motivo}",
            ]
        ),
        flush=True,
    )


def analisar_argumentos():
    analisador = argparse.ArgumentParser(
        description="Modo controle: visao + PID + maquina de estados OBR + serial",
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
    analisador.add_argument("--limiar-confianca", type=float, default=0.12)
    analisador.add_argument("--limiar-intersecao", type=float, default=0.52)
    analisador.add_argument("--limiar-intersecao-lado", type=float, default=0.18)

    analisador.add_argument("--verde-hmin", type=int, default=35)
    analisador.add_argument("--verde-hmax", type=int, default=95)
    analisador.add_argument("--verde-smin", type=int, default=65)
    analisador.add_argument("--verde-vmin", type=int, default=65)
    analisador.add_argument("--verde-area-min", type=int, default=230)
    analisador.add_argument("--verde-area-falsa", type=int, default=90)
    analisador.add_argument("--verde-zona-min", type=float, default=0.45)
    analisador.add_argument("--verde-zona-max", type=float, default=0.95)

    analisador.add_argument("--vermelho-hmin1", type=int, default=0)
    analisador.add_argument("--vermelho-hmax1", type=int, default=12)
    analisador.add_argument("--vermelho-hmin2", type=int, default=168)
    analisador.add_argument("--vermelho-hmax2", type=int, default=180)
    analisador.add_argument("--vermelho-smin", type=int, default=70)
    analisador.add_argument("--vermelho-vmin", type=int, default=60)
    analisador.add_argument("--vermelho-area-min", type=int, default=260)
    analisador.add_argument("--vermelho-area-falsa", type=int, default=110)
    analisador.add_argument("--vermelho-zona-min", type=float, default=0.56)
    analisador.add_argument("--vermelho-zona-max", type=float, default=0.98)
    analisador.add_argument("--vermelho-frames-confirmacao", type=int, default=3)
    analisador.add_argument("--vermelho-frames-liberacao", type=int, default=5)
    analisador.add_argument("--tempo-vermelho-parado", type=float, default=5.0)
    analisador.add_argument(
        "--nao-encerrar-apos-vermelho",
        action="store_true",
        help="Se informado, volta a controlar apos a parada no vermelho.",
    )

    analisador.add_argument("--kp", type=float, default=70.0)
    analisador.add_argument("--ki", type=float, default=7.0)
    analisador.add_argument("--kd", type=float, default=18.0)
    analisador.add_argument("--integral-max", type=float, default=0.85)
    analisador.add_argument("--dt-minimo", type=float, default=0.01)
    analisador.add_argument("--alpha-derivada", type=float, default=0.25)
    analisador.add_argument("--correcao-maxima", type=float, default=65.0)
    analisador.add_argument("--limiar-erro-curva", type=float, default=0.28)

    analisador.add_argument("--velocidade-base", type=int, default=150)
    analisador.add_argument("--velocidade-curva", type=int, default=132)
    analisador.add_argument("--velocidade-giro", type=int, default=124)
    analisador.add_argument("--velocidade-giro-180", type=int, default=132)
    analisador.add_argument("--velocidade-gap", type=int, default=145)
    analisador.add_argument("--velocidade-recuperacao-leve", type=int, default=118)
    analisador.add_argument("--velocidade-recuperacao-forte", type=int, default=132)

    analisador.add_argument("--tempo-inicial", type=float, default=0.40)
    analisador.add_argument("--tempo-gap", type=float, default=0.70)
    analisador.add_argument("--tempo-recuperacao", type=float, default=1.60)
    analisador.add_argument("--tempo-intersecao-sem-marcacao", type=float, default=0.32)
    analisador.add_argument("--tempo-intersecao-verde", type=float, default=0.48)
    analisador.add_argument("--tempo-giro-180", type=float, default=1.15)
    analisador.add_argument("--cooldown-intersecao", type=float, default=0.55)
    analisador.add_argument("--cooldown-verde", type=float, default=0.65)
    analisador.add_argument("--cooldown-beco", type=float, default=1.60)
    analisador.add_argument("--ganho-tendencia-gap", type=float, default=55.0)
    analisador.add_argument("--correcao-gap-maxima", type=float, default=28.0)

    analisador.add_argument("--port", type=str, default=None, help="Porta serial (ex: /dev/ttyACM0).")
    analisador.add_argument("--baud", type=int, default=115200)
    analisador.add_argument("--comando-intervalo", type=float, default=0.10)

    return analisador.parse_args()


def criar_configuracao_visao(parametros):
    return ConfiguracaoVisao(
        roi=parametros.roi,
        limiar_binario=parametros.limiar_binario,
        inverter_linha=parametros.invert,
        limiar_confianca=parametros.limiar_confianca,
        limiar_intersecao=parametros.limiar_intersecao,
        limiar_lado_intersecao=parametros.limiar_intersecao_lado,
        verde_hmin=parametros.verde_hmin,
        verde_hmax=parametros.verde_hmax,
        verde_smin=parametros.verde_smin,
        verde_vmin=parametros.verde_vmin,
        verde_area_minima=parametros.verde_area_min,
        verde_area_falsa=parametros.verde_area_falsa,
        verde_zona_min=parametros.verde_zona_min,
        verde_zona_max=parametros.verde_zona_max,
        vermelho_hmin1=parametros.vermelho_hmin1,
        vermelho_hmax1=parametros.vermelho_hmax1,
        vermelho_hmin2=parametros.vermelho_hmin2,
        vermelho_hmax2=parametros.vermelho_hmax2,
        vermelho_smin=parametros.vermelho_smin,
        vermelho_vmin=parametros.vermelho_vmin,
        vermelho_area_minima=parametros.vermelho_area_min,
        vermelho_area_falsa=parametros.vermelho_area_falsa,
        vermelho_zona_min=parametros.vermelho_zona_min,
        vermelho_zona_max=parametros.vermelho_zona_max,
        vermelho_frames_confirmacao=parametros.vermelho_frames_confirmacao,
        vermelho_frames_liberacao=parametros.vermelho_frames_liberacao,
    )


def principal():
    parametros = analisar_argumentos()
    parametros.encerrar_apos_vermelho = not parametros.nao_encerrar_apos_vermelho

    parametros.velocidade_base = _limitar_pwm(parametros.velocidade_base)
    parametros.velocidade_curva = _limitar_pwm(parametros.velocidade_curva)
    parametros.velocidade_giro = _limitar_pwm(parametros.velocidade_giro)
    parametros.velocidade_giro_180 = _limitar_pwm(parametros.velocidade_giro_180)
    parametros.velocidade_gap = _limitar_pwm(parametros.velocidade_gap)
    parametros.velocidade_recuperacao_leve = _limitar_pwm(parametros.velocidade_recuperacao_leve)
    parametros.velocidade_recuperacao_forte = _limitar_pwm(parametros.velocidade_recuperacao_forte)

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

    _entrar_estado(estado_controle, ESTADO_INICIANDO, time.monotonic(), parametros.tempo_inicial)

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

            acao, motivo, correcao_pid = _atualizar_estado_controle(
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

            if estado_controle.finalizado:
                if ser is not None:
                    parar(ser)
                break

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


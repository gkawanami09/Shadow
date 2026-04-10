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
    enviar_velocidades_diferenciais,
    fechar_serial,
    giro_90_direita,
    giro_90_esquerda,
    parar,
)
from stream import ServidorStream
from vision import ConfiguracaoVisao, EstadoVisao, analisar_quadro


ESTADO_INICIANDO = "INICIANDO"
ESTADO_SEGUINDO = "SEGUINDO_LINHA"
ESTADO_SEM_LINHA = "SEM_LINHA"
ESTADO_RECUPERANDO = "RECUPERANDO_LINHA"
ESTADO_MANOBRA_90 = "EXECUTANDO_CURVA_90"


@dataclass
class EstadoControle:
    estado_atual: str = ESTADO_INICIANDO
    tempo_entrada_estado: float = field(default_factory=time.monotonic)
    assinatura_ultima_acao: tuple | None = None
    instante_ultimo_envio: float = 0.0
    velocidade_esquerda_anterior: int = 0
    velocidade_direita_anterior: int = 0
    erro_linha_anterior: float = 0.0
    lado_preferencial_recuperacao: int = 0
    manobra_ativa: dict | None = None
    manobra_ativa_ate: float = 0.0
    instante_ultimo_giro_90: float = -999.0


class   ControladorPID:
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


def _interpolar(valor_inicial, valor_final, proporcao):
    proporcao = _limitar(float(proporcao), 0.0, 1.0)
    return float(valor_inicial) + (float(valor_final) - float(valor_inicial)) * proporcao


def _aplicar_piso_assinado(valor, piso):
    if valor > 0:
        return max(valor, piso)
    if valor < 0:
        return min(valor, -piso)
    return 0


def _obter_risco_lateral(erro_linha, estado_controle, parametros):
    erro_abs = abs(erro_linha)
    direcao_erro = 0
    if erro_linha > 0.0:
        direcao_erro = 1
    elif erro_linha < 0.0:
        direcao_erro = -1

    delta_erro = erro_linha - estado_controle.erro_linha_anterior
    afastando = direcao_erro != 0 and (delta_erro * direcao_erro) > parametros.delta_erro_antecipacao

    perto_borda = erro_abs >= parametros.limiar_erro_borda
    risco_alto = erro_abs >= parametros.limiar_erro_risco
    risco_medio = erro_abs >= parametros.limiar_erro_antecipacao and afastando

    return {
        "direcao": direcao_erro,
        "delta": delta_erro,
        "perto_borda": perto_borda,
        "afastando": afastando,
        "risco_alto": risco_alto,
        "risco_medio": risco_medio,
    }


def _calcular_acao_recuperacao(estado_controle, parametros):
    lado = estado_controle.lado_preferencial_recuperacao
    if lado == 0:
        lado = 1 if estado_controle.erro_linha_anterior >= 0.0 else -1
        if lado == 0:
            lado = 1

    velocidade_externa = _limitar(
        parametros.velocidade_recuperacao,
        parametros.velocidade_minima,
        parametros.velocidade_maxima,
    )
    velocidade_interna = _limitar(
        parametros.velocidade_recuperacao_reversa,
        parametros.velocidade_minima,
        parametros.velocidade_maxima,
    )

    if lado > 0:
        return _criar_acao_diferencial(velocidade_externa, -velocidade_interna)
    return _criar_acao_diferencial(-velocidade_interna, velocidade_externa)


def _ganho_lookahead_dinamico(erro_linha, erro_lookahead, confianca_lookahead, parametros):
    erro_referencia = max(abs(float(erro_linha)), abs(float(erro_lookahead)))
    faixa = max(1e-6, parametros.lookahead_erro_maximo - parametros.lookahead_erro_minimo)
    proporcao = _limitar(
        (erro_referencia - parametros.lookahead_erro_minimo) / faixa,
        0.0,
        1.0,
    )
    ganho_base = _interpolar(
        parametros.ganho_lookahead_suave,
        parametros.ganho_lookahead_forte,
        proporcao,
    )
    return float(ganho_base * _limitar(confianca_lookahead, 0.0, 1.0))


def _criar_acao_parar():
    return {"tipo": "S"}


def _criar_acao_diferencial(velocidade_esquerda, velocidade_direita):
    return {
        "tipo": "D",
        "velocidade_esquerda": _limitar_pwm_assinado(velocidade_esquerda),
        "velocidade_direita": _limitar_pwm_assinado(velocidade_direita),
    }


def _criar_acao_giro_90(direcao, velocidade):
    return {
        "tipo": "L90" if direcao == "esquerda" else "R90",
        "velocidade": _limitar_pwm(velocidade),
    }


def _assinatura_acao(acao):
    if acao["tipo"] == "D":
        return ("D", acao["velocidade_esquerda"], acao["velocidade_direita"])
    if acao["tipo"] in {"L90", "R90"}:
        return (acao["tipo"], acao["velocidade"])
    return ("S",)


def _deve_enviar(estado_controle, acao, agora, intervalo_minimo):
    assinatura = _assinatura_acao(acao)
    if assinatura != estado_controle.assinatura_ultima_acao:
        return True
    if acao["tipo"] in {"L90", "R90"}:
        return False
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

    if acao["tipo"] == "L90":
        giro_90_esquerda(ser, acao["velocidade"])
        return

    if acao["tipo"] == "R90":
        giro_90_direita(ser, acao["velocidade"])
        return

    parar(ser)


def _deve_executar_curva_90(estado_controle, dados_visao, parametros, agora):
    if (agora - estado_controle.instante_ultimo_giro_90) < parametros.cooldown_giro_90:
        return None

    if dados_visao["confianca_linha"] < parametros.limiar_confianca_curva_90_execucao:
        return None

    if dados_visao.get("confianca_curva_90", 0.0) < parametros.limiar_confianca_curva_90_execucao:
        return None

    if dados_visao.get("curva_90_esquerda"):
        return "esquerda"
    if dados_visao.get("curva_90_direita"):
        return "direita"
    return None


def _calcular_acao_pid(dados_visao, pid, parametros, tempo_atual):
    erro_linha = dados_visao["erro_linha"]
    erro_lookahead = float(dados_visao.get("erro_lookahead", erro_linha))
    confianca_lookahead = float(dados_visao.get("confianca_lookahead", 0.0))
    if parametros.inverter_correcao:
        erro_linha = -erro_linha
        erro_lookahead = -erro_lookahead

    ganho_lookahead = _ganho_lookahead_dinamico(
        erro_linha,
        erro_lookahead,
        confianca_lookahead,
        parametros,
    )
    erro_controle = _limitar(
        erro_linha + (erro_lookahead * ganho_lookahead),
        -1.0,
        1.0,
    )

    correcao_pid = pid.calcular(erro_controle, tempo_atual)
    correcao_pid = _limitar(correcao_pid, -parametros.correcao_maxima, parametros.correcao_maxima)
    erro_abs = abs(erro_controle)
    risco_lateral = _obter_risco_lateral(erro_controle, dados_visao["estado_controle"], parametros)
    confianca_baixa = dados_visao["confianca_linha"] <= parametros.limiar_confianca_pivo

    usar_pivo = (
        erro_abs >= parametros.limiar_erro_pivo
        and (
            erro_abs >= parametros.limiar_erro_pivo_critico
            or confianca_baixa
            or risco_lateral["perto_borda"]
        )
    )

    if usar_pivo:
        velocidade_pivo = _limitar(
            parametros.velocidade_pivo,
            parametros.velocidade_minima,
            parametros.velocidade_maxima,
        )
        velocidade_pivo = _limitar(
            velocidade_pivo + parametros.bonus_velocidade_pivo,
            parametros.velocidade_minima,
            parametros.velocidade_maxima,
        )
        if erro_controle >= 0.0:
            velocidade_esquerda = velocidade_pivo
            velocidade_direita = -velocidade_pivo
        else:
            velocidade_esquerda = -velocidade_pivo
            velocidade_direita = velocidade_pivo
        return _criar_acao_diferencial(velocidade_esquerda, velocidade_direita), correcao_pid

    faixa_curva = max(1e-6, parametros.limiar_erro_pivo - parametros.limiar_erro_curva)
    proporcao_curva = _limitar((erro_abs - parametros.limiar_erro_curva) / faixa_curva, 0.0, 1.0)
    velocidade_base = _interpolar(
        parametros.velocidade_base,
        parametros.velocidade_curva,
        proporcao_curva,
    )

    if risco_lateral["risco_alto"]:
        velocidade_base = min(velocidade_base, float(parametros.velocidade_risco))
        correcao_pid += risco_lateral["direcao"] * parametros.bonus_correcao_risco
    elif risco_lateral["risco_medio"]:
        velocidade_base = min(velocidade_base, float(parametros.velocidade_antecipacao))
        correcao_pid += risco_lateral["direcao"] * parametros.bonus_correcao_antecipacao

    if confianca_baixa:
        velocidade_base = min(velocidade_base, float(parametros.velocidade_confianca_baixa))
        correcao_pid += risco_lateral["direcao"] * parametros.bonus_correcao_confianca

    velocidade_esquerda_bruta = velocidade_base + correcao_pid
    velocidade_direita_bruta = velocidade_base - correcao_pid

    if erro_abs >= parametros.limiar_erro_motor_forte:
        bonus_motor = _interpolar(
            0.0,
            parametros.bonus_velocidade_motor_forte,
            (erro_abs - parametros.limiar_erro_motor_forte)
            / max(1e-6, 1.0 - parametros.limiar_erro_motor_forte),
        )
        if erro_controle >= 0.0:
            velocidade_esquerda_bruta += bonus_motor
            velocidade_direita_bruta -= bonus_motor * parametros.fator_freio_motor_interno
        else:
            velocidade_esquerda_bruta -= bonus_motor * parametros.fator_freio_motor_interno
            velocidade_direita_bruta += bonus_motor

    if erro_abs < parametros.limiar_erro_reversao:
        velocidade_esquerda = _limitar(
            max(0.0, velocidade_esquerda_bruta),
            0.0,
            float(parametros.velocidade_maxima),
        )
        velocidade_direita = _limitar(
            max(0.0, velocidade_direita_bruta),
            0.0,
            float(parametros.velocidade_maxima),
        )

        if velocidade_esquerda > 0.0:
            velocidade_esquerda = max(velocidade_esquerda, float(parametros.velocidade_minima))
        if velocidade_direita > 0.0:
            velocidade_direita = max(velocidade_direita, float(parametros.velocidade_minima))
    else:
        velocidade_esquerda = _limitar(
            _aplicar_piso_assinado(velocidade_esquerda_bruta, parametros.velocidade_minima),
            -parametros.velocidade_maxima,
            parametros.velocidade_maxima,
        )
        velocidade_direita = _limitar(
            _aplicar_piso_assinado(velocidade_direita_bruta, parametros.velocidade_minima),
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

    if estado_controle.manobra_ativa is not None and agora < estado_controle.manobra_ativa_ate:
        return estado_controle.manobra_ativa, "executando curva de 90 graus", correcao_pid

    if estado_controle.manobra_ativa is not None and agora >= estado_controle.manobra_ativa_ate:
        estado_controle.manobra_ativa = None
        estado_controle.manobra_ativa_ate = 0.0

    linha_valida = bool(
        dados_visao["linha_encontrada"]
        and dados_visao["confianca_linha"] >= parametros.limiar_confianca
    )
    linha_fraca = bool(
        dados_visao["linha_encontrada"]
        and dados_visao["confianca_linha"] >= parametros.limiar_confianca_minima_recuperacao
    )

    if dados_visao["linha_encontrada"]:
        if dados_visao["erro_linha"] > 0.03:
            estado_controle.lado_preferencial_recuperacao = 1
        elif dados_visao["erro_linha"] < -0.03:
            estado_controle.lado_preferencial_recuperacao = -1

    if linha_valida:
        direcao_curva_90 = _deve_executar_curva_90(estado_controle, dados_visao, parametros, agora)
        if direcao_curva_90 is not None:
            estado_controle.estado_atual = ESTADO_MANOBRA_90
            estado_controle.tempo_entrada_estado = agora
            estado_controle.instante_ultimo_giro_90 = agora
            estado_controle.lado_preferencial_recuperacao = 1 if direcao_curva_90 == "direita" else -1
            estado_controle.manobra_ativa = _criar_acao_giro_90(
                direcao_curva_90,
                parametros.velocidade_giro_90,
            )
            estado_controle.manobra_ativa_ate = agora + parametros.tempo_giro_90
            pid.reiniciar(suave=False)
            return estado_controle.manobra_ativa, f"curva de 90 graus {direcao_curva_90}", correcao_pid

        if estado_controle.estado_atual != ESTADO_SEGUINDO:
            estado_controle.estado_atual = ESTADO_SEGUINDO
            estado_controle.tempo_entrada_estado = agora
            pid.reiniciar(suave=True)

        dados_visao_controle = dict(dados_visao)
        dados_visao_controle["estado_controle"] = estado_controle

        acao, correcao_pid = _calcular_acao_pid(dados_visao_controle, pid, parametros, agora)
        estado_controle.velocidade_esquerda_anterior = acao["velocidade_esquerda"]
        estado_controle.velocidade_direita_anterior = acao["velocidade_direita"]
        estado_controle.erro_linha_anterior = dados_visao["erro_linha"]
        erro_referencia = max(
            abs(dados_visao["erro_linha"]),
            abs(float(dados_visao.get("erro_lookahead", 0.0))),
        )

        if acao["tipo"] == "D" and (
            acao["velocidade_esquerda"] < 0 or acao["velocidade_direita"] < 0
        ):
            motivo = "curva critica com reversao controlada"
        elif erro_referencia >= parametros.limiar_erro_risco:
            motivo = "correcao imediata por risco lateral"
        elif erro_referencia >= parametros.limiar_erro_antecipacao:
            motivo = "antecipando fuga lateral"
        else:
            motivo = "seguindo linha com PID suave"
        return acao, motivo, correcao_pid

    if linha_fraca:
        if estado_controle.estado_atual != ESTADO_RECUPERANDO:
            estado_controle.estado_atual = ESTADO_RECUPERANDO
            estado_controle.tempo_entrada_estado = agora
            pid.reiniciar(suave=False)

        estado_controle.erro_linha_anterior = dados_visao["erro_linha"]
        acao = _calcular_acao_recuperacao(estado_controle, parametros)
        estado_controle.velocidade_esquerda_anterior = acao["velocidade_esquerda"]
        estado_controle.velocidade_direita_anterior = acao["velocidade_direita"]
        return acao, "recuperando linha com leitura fraca", correcao_pid

    if (
        dados_visao["tempo_sem_linha"] <= parametros.tempo_recuperacao_linha
        and estado_controle.lado_preferencial_recuperacao != 0
    ):
        if estado_controle.estado_atual != ESTADO_RECUPERANDO:
            estado_controle.estado_atual = ESTADO_RECUPERANDO
            estado_controle.tempo_entrada_estado = agora
            pid.reiniciar(suave=False)

        acao = _calcular_acao_recuperacao(estado_controle, parametros)
        estado_controle.velocidade_esquerda_anterior = acao["velocidade_esquerda"]
        estado_controle.velocidade_direita_anterior = acao["velocidade_direita"]
        return acao, "linha perdida, buscando ultimo lado conhecido", correcao_pid

    if estado_controle.estado_atual != ESTADO_SEM_LINHA:
        estado_controle.estado_atual = ESTADO_SEM_LINHA
        estado_controle.tempo_entrada_estado = agora

    pid.reiniciar(suave=True)
    estado_controle.velocidade_esquerda_anterior = 0
    estado_controle.velocidade_direita_anterior = 0
    estado_controle.erro_linha_anterior = 0.0
    motivo = "linha ausente ou confianca baixa"
    return _criar_acao_parar(), motivo, correcao_pid


def _desenhar_info_controle(quadro_debug, estado_controle, dados_visao, acao, correcao_pid, pid, motivo):
    if acao["tipo"] == "D":
        texto_acao = f"D,{acao['velocidade_esquerda']},{acao['velocidade_direita']}"
    elif acao["tipo"] in {"L90", "R90"}:
        texto_acao = f"{acao['tipo']},{acao['velocidade']}"
    else:
        texto_acao = "S"

    textos = [
        f"estado={estado_controle.estado_atual}",
        f"acao={texto_acao}",
        f"motivo={motivo}",
        (
            f"erro={dados_visao['erro_linha']:+.3f} "
            f"la={dados_visao.get('erro_lookahead', 0.0):+.3f} "
            f"conf={dados_visao['confianca_linha']:.2f} "
            f"c90={dados_visao.get('confianca_curva_90', 0.0):.2f}"
        ),
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
    elif acao["tipo"] in {"L90", "R90"}:
        texto_acao = f"{acao['tipo']},{acao['velocidade']}"
    else:
        texto_acao = "S"

    print(
        " | ".join(
            [
                f"estado={estado_controle.estado_atual}",
                f"acao={texto_acao}",
                f"erro={dados_visao['erro_linha']:+.3f}",
                f"lookahead={dados_visao.get('erro_lookahead', 0.0):+.3f}",
                f"curva90={dados_visao.get('confianca_curva_90', 0.0):.2f}",
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
    analisador.add_argument("--lookahead-fracao", type=float, default=0.42)
    analisador.add_argument("--lookahead-minimo-pixels", type=int, default=18)
    analisador.add_argument("--limiar-confianca-curva-90", type=float, default=0.22)
    analisador.add_argument("--limiar-confianca-lookahead-curva-90", type=float, default=0.20)
    analisador.add_argument("--limiar-erro-lookahead-curva-90", type=float, default=0.48)
    analisador.add_argument("--limiar-delta-erro-curva-90", type=float, default=0.18)
    analisador.add_argument("--limiar-erro-base-curva-90", type=float, default=0.22)
    analisador.add_argument("--faixa-superior-curva-90", type=float, default=0.40)
    analisador.add_argument("--faixa-inferior-curva-90", type=float, default=0.24)
    analisador.add_argument("--densidade-lateral-curva-90", type=float, default=0.16)
    analisador.add_argument("--densidade-oposta-max-curva-90", type=float, default=0.05)
    analisador.add_argument("--densidade-base-centro-curva-90", type=float, default=0.10)
    analisador.add_argument("--limiar-confianca-minima-recuperacao", type=float, default=0.03)

    analisador.add_argument("--kp", type=float, default=145.0)
    analisador.add_argument("--ki", type=float, default=10.0)
    analisador.add_argument("--kd", type=float, default=42.0)
    analisador.add_argument("--integral-max", type=float, default=0.85)
    analisador.add_argument("--dt-minimo", type=float, default=0.01)
    analisador.add_argument("--alpha-derivada", type=float, default=0.25)
    analisador.add_argument("--correcao-maxima", type=float, default=200.0)
    analisador.add_argument("--ganho-lookahead-suave", type=float, default=0.20)
    analisador.add_argument("--ganho-lookahead-forte", type=float, default=0.95)
    analisador.add_argument("--lookahead-erro-minimo", type=float, default=0.10)
    analisador.add_argument("--lookahead-erro-maximo", type=float, default=0.42)

    analisador.add_argument("--velocidade-base", type=int, default=82)
    analisador.add_argument("--velocidade-curva", type=int, default=64)
    analisador.add_argument("--velocidade-minima", type=int, default=45)
    analisador.add_argument("--velocidade-maxima", type=int, default=135)
    analisador.add_argument("--limiar-erro-curva", type=float, default=0.12)
    analisador.add_argument("--limiar-erro-reversao", type=float, default=0.58)
    analisador.add_argument("--limiar-erro-pivo", type=float, default=0.72)
    analisador.add_argument("--limiar-erro-pivo-critico", type=float, default=0.88)
    analisador.add_argument("--limiar-confianca-pivo", type=float, default=0.16)
    analisador.add_argument("--limiar-erro-antecipacao", type=float, default=0.22)
    analisador.add_argument("--limiar-erro-risco", type=float, default=0.34)
    analisador.add_argument("--limiar-erro-borda", type=float, default=0.58)
    analisador.add_argument("--delta-erro-antecipacao", type=float, default=0.035)
    analisador.add_argument("--velocidade-antecipacao", type=int, default=56)
    analisador.add_argument("--velocidade-risco", type=int, default=48)
    analisador.add_argument("--velocidade-confianca-baixa", type=int, default=44)
    analisador.add_argument("--velocidade-recuperacao", type=int, default=62)
    analisador.add_argument("--velocidade-recuperacao-reversa", type=int, default=56)
    analisador.add_argument("--bonus-correcao-antecipacao", type=float, default=18.0)
    analisador.add_argument("--bonus-correcao-risco", type=float, default=34.0)
    analisador.add_argument("--bonus-correcao-confianca", type=float, default=22.0)
    analisador.add_argument("--limiar-erro-motor-forte", type=float, default=0.34)
    analisador.add_argument("--bonus-velocidade-motor-forte", type=float, default=22.0)
    analisador.add_argument("--fator-freio-motor-interno", type=float, default=0.65)
    analisador.add_argument("--velocidade-pivo", type=int, default=96)
    analisador.add_argument("--bonus-velocidade-pivo", type=int, default=14)
    analisador.add_argument("--velocidade-giro-90", type=int, default=120)
    analisador.add_argument("--tempo-giro-90", type=float, default=0.42)
    analisador.add_argument("--cooldown-giro-90", type=float, default=0.90)
    analisador.add_argument("--limiar-confianca-curva-90-execucao", type=float, default=0.26)
    analisador.add_argument(
        "--inverter-correcao",
        action="store_true",
        help="Inverte o sentido da correcao lateral (use quando corrige para o lado errado).",
    )

    analisador.add_argument("--tempo-inicial", type=float, default=0.35)
    analisador.add_argument("--tempo-recuperacao-linha", type=float, default=0.28)

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
        lookahead_fracao=parametros.lookahead_fracao,
        lookahead_minimo_pixels=parametros.lookahead_minimo_pixels,
        limiar_confianca_curva_90=parametros.limiar_confianca_curva_90,
        limiar_confianca_lookahead_curva_90=parametros.limiar_confianca_lookahead_curva_90,
        limiar_erro_lookahead_curva_90=parametros.limiar_erro_lookahead_curva_90,
        limiar_delta_erro_curva_90=parametros.limiar_delta_erro_curva_90,
        limiar_erro_base_curva_90=parametros.limiar_erro_base_curva_90,
        faixa_superior_curva_90=parametros.faixa_superior_curva_90,
        faixa_inferior_curva_90=parametros.faixa_inferior_curva_90,
        densidade_lateral_curva_90=parametros.densidade_lateral_curva_90,
        densidade_oposta_max_curva_90=parametros.densidade_oposta_max_curva_90,
        densidade_base_centro_curva_90=parametros.densidade_base_centro_curva_90,
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

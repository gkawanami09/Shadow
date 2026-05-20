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
ESTADO_CONTENCAO = "CONTENDO_LINHA"
ESTADO_BUSCA = "BUSCANDO_LINHA"
ESTADO_MANOBRA_90 = "EXECUTANDO_CURVA_90"
ESTADO_ASSISTENCIA_CURVA = "ASSISTINDO_CURVA"


@dataclass
class EstadoControle:
    estado_atual: str = ESTADO_INICIANDO
    tempo_entrada_estado: float = field(default_factory=time.monotonic)
    assinatura_ultima_acao: tuple | None = None
    instante_ultimo_envio: float = 0.0
    manobra_ativa: dict | None = None
    manobra_ativa_ate: float = 0.0
    instante_ultimo_giro_90: float = -999.0
    instante_ultima_assistencia_curva: float = -999.0
    ultimo_erro_confiavel: float = 0.0
    ultima_direcao_confiavel: str | None = None
    instante_ultima_linha_confiavel: float = -999.0
    quadros_confiaveis_consecutivos: int = 0


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

    def estabilizar_centro(self):
        self.erro_proporcional = 0.0
        self.erro_integral *= 0.5
        if abs(self.erro_integral) < 1e-4:
            self.erro_integral = 0.0
        self.erro_derivativo = 0.0
        self.erro_anterior = 0.0
        self.tempo_anterior = None
        self._derivada_filtrada = 0.0

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


def _aplicar_pisos_frente(velocidade_esquerda, velocidade_direita, erro_controle, parametros):
    velocidade_esquerda = _limitar(
        max(0.0, float(velocidade_esquerda)),
        0.0,
        float(parametros.velocidade_maxima),
    )
    velocidade_direita = _limitar(
        max(0.0, float(velocidade_direita)),
        0.0,
        float(parametros.velocidade_maxima),
    )

    piso_externo = float(parametros.velocidade_minima)
    piso_interno = float(
        _limitar(
            parametros.velocidade_minima_interna,
            0,
            parametros.velocidade_minima,
        )
    )

    if erro_controle >= 0.0:
        if velocidade_esquerda > 0.0:
            velocidade_esquerda = max(velocidade_esquerda, piso_externo)
        if velocidade_direita > 0.0:
            velocidade_direita = max(velocidade_direita, piso_interno)
    else:
        if velocidade_esquerda > 0.0:
            velocidade_esquerda = max(velocidade_esquerda, piso_interno)
        if velocidade_direita > 0.0:
            velocidade_direita = max(velocidade_direita, piso_externo)

    return velocidade_esquerda, velocidade_direita


def _compensar_re_esquerda(velocidade_esquerda, velocidade_direita, parametros):
    velocidade_esquerda = float(velocidade_esquerda)
    velocidade_direita = float(velocidade_direita)

    # Compensa apenas a curva para a esquerda: lado esquerdo em re enquanto o
    # lado direito esta parado ou puxando para frente.
    if velocidade_esquerda >= 0.0 or velocidade_direita < 0.0:
        return velocidade_esquerda, velocidade_direita

    bonus_re_esquerda = max(0.0, float(parametros.bonus_re_esquerda))
    piso_re_esquerda = float(
        _limitar(
            parametros.piso_re_esquerda,
            0,
            parametros.velocidade_maxima,
        )
    )

    velocidade_esquerda -= bonus_re_esquerda
    if piso_re_esquerda > 0.0:
        velocidade_esquerda = min(velocidade_esquerda, -piso_re_esquerda)

    velocidade_esquerda = _limitar(
        velocidade_esquerda,
        -float(parametros.velocidade_maxima),
        float(parametros.velocidade_maxima),
    )
    return velocidade_esquerda, velocidade_direita


def _obter_pivo_esquerda_fixo(parametros):
    velocidade_frente = _limitar_pwm(parametros.velocidade_pivo_esquerda_frente)
    velocidade_reversa = _limitar_pwm(parametros.velocidade_pivo_esquerda_reversa)

    if velocidade_frente <= 0:
        velocidade_frente = _limitar_pwm(parametros.velocidade_pivo)
    if velocidade_reversa <= 0:
        velocidade_reversa = _limitar_pwm(parametros.velocidade_pivo)

    return float(-velocidade_reversa), float(velocidade_frente)


def _obter_pivo_direita_fixo(parametros):
    velocidade_frente = _limitar_pwm(parametros.velocidade_pivo_direita_frente)
    velocidade_reversa = _limitar_pwm(parametros.velocidade_pivo_direita_reversa)

    if velocidade_frente <= 0:
        velocidade_frente = _limitar_pwm(parametros.velocidade_pivo)
    if velocidade_reversa <= 0:
        velocidade_reversa = _limitar_pwm(parametros.velocidade_pivo)

    return float(velocidade_frente), float(-velocidade_reversa)


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


def _ganho_correcao_dinamico(erro_abs, parametros):
    faixa = max(1e-6, parametros.limiar_erro_pivo - parametros.limiar_erro_centralizado)
    proporcao = _limitar(
        (float(erro_abs) - parametros.limiar_erro_centralizado) / faixa,
        0.0,
        1.0,
    )
    return float(_interpolar(1.0, parametros.fator_correcao_forte, proporcao)), proporcao


def _erro_referencia_curva(erro_controle, erro_lookahead, confianca_lookahead, parametros):
    erro_referencia = abs(float(erro_controle))
    if confianca_lookahead >= parametros.limiar_confianca_lookahead_velocidade:
        erro_referencia = max(
            erro_referencia,
            abs(float(erro_lookahead)) * float(parametros.fator_antecipacao_velocidade),
    )
    return float(_limitar(erro_referencia, 0.0, 1.0))


def _deve_forcar_pivo_curva_fechada(
    erro_controle,
    erro_lookahead,
    confianca_lookahead,
    dados_visao,
    parametros,
):
    if abs(float(erro_controle)) >= float(parametros.limiar_erro_pivo_forcado):
        return True

    if (
        confianca_lookahead >= float(parametros.limiar_confianca_lookahead_pivo_forcado)
        and abs(float(erro_lookahead)) >= float(parametros.limiar_erro_lookahead_pivo_forcado)
    ):
        return True

    if (
        (
            dados_visao.get("linha_toca_borda_esquerda")
            or dados_visao.get("linha_toca_borda_direita")
        )
        and abs(float(erro_lookahead)) >= float(parametros.limiar_erro_lookahead_borda_pivo)
    ):
        return True

    return False


def _direcao_oposta(direcao):
    if direcao == "esquerda":
        return "direita"
    if direcao == "direita":
        return "esquerda"
    return None


def _direcao_correcao_por_erro(erro_visao, parametros, fallback=None):
    if erro_visao > 0.0:
        return _ajustar_direcao_para_chassi("direita", parametros)
    if erro_visao < 0.0:
        return _ajustar_direcao_para_chassi("esquerda", parametros)
    return fallback


def _criar_acao_busca_linha(direcao, velocidade_frente, velocidade_reversa, parametros):
    direcao = direcao if direcao in {"esquerda", "direita"} else "direita"
    velocidade_frente = _limitar_pwm(velocidade_frente)
    velocidade_reversa = _limitar_pwm(velocidade_reversa)
    if direcao == "esquerda":
        velocidade_esquerda, velocidade_direita = _compensar_re_esquerda(
            -velocidade_reversa,
            velocidade_frente,
            parametros,
        )
    else:
        velocidade_esquerda = velocidade_frente
        velocidade_direita = -velocidade_reversa

    acao = _criar_acao_diferencial(velocidade_esquerda, velocidade_direita)
    acao["modo"] = "busca_linha"
    return acao


def _calcular_erro_contencao(dados_visao, parametros):
    erro_linha = float(dados_visao["erro_linha"])
    erro_lookahead = float(dados_visao.get("erro_lookahead", erro_linha))
    confianca_lookahead = _limitar(float(dados_visao.get("confianca_lookahead", 0.0)), 0.0, 1.0)
    ganho_lookahead = float(parametros.ganho_lookahead_contencao) * confianca_lookahead
    erro_contencao = _limitar(erro_linha + (erro_lookahead * ganho_lookahead), -1.0, 1.0)
    if parametros.inverter_correcao:
        erro_contencao = -erro_contencao
    return float(erro_contencao)


def _linha_minimamente_valida(dados_visao, parametros):
    return bool(
        dados_visao["linha_encontrada"]
        and dados_visao["confianca_linha"] >= parametros.limiar_confianca
    )


def _linha_em_risco(dados_visao, parametros):
    if not dados_visao["linha_encontrada"]:
        return True
    if dados_visao["confianca_linha"] < parametros.limiar_confianca_cautela:
        return True
    if abs(float(dados_visao["erro_linha"])) >= parametros.limiar_erro_contencao:
        return True
    if dados_visao.get("linha_toca_borda_esquerda") or dados_visao.get("linha_toca_borda_direita"):
        return True
    return False


def _linha_confiavel_para_retomada(dados_visao, parametros):
    return bool(
        dados_visao["linha_encontrada"]
        and dados_visao["confianca_linha"] >= parametros.limiar_confianca_retomada
        and not dados_visao.get("linha_toca_borda_esquerda")
        and not dados_visao.get("linha_toca_borda_direita")
    )


def _atualizar_memoria_linha(estado_controle, dados_visao, parametros, agora):
    if not _linha_minimamente_valida(dados_visao, parametros):
        estado_controle.quadros_confiaveis_consecutivos = 0
        return

    erro_visao = float(dados_visao["erro_linha"])
    if erro_visao == 0.0:
        if dados_visao.get("linha_toca_borda_esquerda"):
            erro_visao = -1.0
        elif dados_visao.get("linha_toca_borda_direita"):
            erro_visao = 1.0

    direcao = _direcao_correcao_por_erro(
        erro_visao,
        parametros,
        fallback=estado_controle.ultima_direcao_confiavel,
    )
    if direcao is not None:
        estado_controle.ultima_direcao_confiavel = direcao
    estado_controle.ultimo_erro_confiavel = erro_visao

    if _linha_confiavel_para_retomada(dados_visao, parametros):
        estado_controle.instante_ultima_linha_confiavel = agora
        estado_controle.quadros_confiaveis_consecutivos += 1
    else:
        estado_controle.quadros_confiaveis_consecutivos = 0


def _obter_direcao_busca(estado_controle, parametros, agora):
    direcao_base = estado_controle.ultima_direcao_confiavel
    if direcao_base not in {"esquerda", "direita"}:
        return None

    decorrido = max(0.0, agora - estado_controle.tempo_entrada_estado)
    if decorrido <= parametros.tempo_busca_mesmo_lado:
        return direcao_base

    fase = int((decorrido - parametros.tempo_busca_mesmo_lado) / max(0.05, parametros.tempo_varredura_busca))
    if fase % 2 == 0:
        return _direcao_oposta(direcao_base)
    return direcao_base


def _calcular_acao_busca_sem_linha(estado_controle, dados_visao, parametros, agora):
    tempo_desde_linha_confiavel = agora - estado_controle.instante_ultima_linha_confiavel
    if tempo_desde_linha_confiavel > parametros.tempo_memoria_busca:
        return _criar_acao_parar(), "linha perdida alem da memoria segura"
    if dados_visao["tempo_sem_linha"] > parametros.tempo_maximo_busca_sem_linha:
        return _criar_acao_parar(), "busca expirou sem reencontrar a linha"

    direcao_busca = _obter_direcao_busca(estado_controle, parametros, agora)
    if direcao_busca is None:
        return _criar_acao_parar(), "sem direcao confiavel para buscar a linha"

    return (
        _criar_acao_busca_linha(
            direcao_busca,
            parametros.velocidade_busca_linha,
            parametros.velocidade_reversa_busca_linha,
            parametros,
        ),
        f"busca lenta da linha para {direcao_busca}",
    )


def _calcular_acao_contencao(dados_visao, estado_controle, parametros):
    erro_contencao = _calcular_erro_contencao(dados_visao, parametros)
    direcao_borda = None
    if dados_visao.get("linha_toca_borda_esquerda"):
        direcao_borda = _ajustar_direcao_para_chassi("esquerda", parametros)
    elif dados_visao.get("linha_toca_borda_direita"):
        direcao_borda = _ajustar_direcao_para_chassi("direita", parametros)

    if direcao_borda is not None or abs(erro_contencao) >= parametros.limiar_erro_contencao_pivo:
        direcao_pivo = direcao_borda or _direcao_correcao_por_erro(
            float(dados_visao["erro_linha"]),
            parametros,
            fallback=estado_controle.ultima_direcao_confiavel,
        )
        return (
            _criar_acao_busca_linha(
                direcao_pivo,
                parametros.velocidade_contencao,
                parametros.velocidade_reversa_contencao,
                parametros,
            ),
            "contencao em pivo para manter a linha no quadro",
            erro_contencao,
        )

    velocidade_base = float(min(parametros.velocidade_contencao, parametros.velocidade_maxima_contencao))
    ajuste = min(
        float(parametros.velocidade_maxima_contencao),
        abs(erro_contencao) * float(parametros.ganho_correcao_contencao),
    )
    velocidade_esquerda = velocidade_base + ajuste if erro_contencao >= 0.0 else velocidade_base - ajuste
    velocidade_direita = velocidade_base - ajuste if erro_contencao >= 0.0 else velocidade_base + ajuste

    velocidade_esquerda = _limitar(velocidade_esquerda, 0.0, float(parametros.velocidade_maxima_contencao))
    velocidade_direita = _limitar(velocidade_direita, 0.0, float(parametros.velocidade_maxima_contencao))

    return (
        _criar_acao_diferencial(velocidade_esquerda, velocidade_direita),
        "contencao com avanco reduzido",
        erro_contencao,
    )


def _criar_acao_parar():
    return {"tipo": "S"}


def _criar_acao_diferencial(velocidade_esquerda, velocidade_direita):
    return {
        "tipo": "D",
        "velocidade_esquerda": _limitar_pwm_assinado(velocidade_esquerda),
        "velocidade_direita": _limitar_pwm_assinado(velocidade_direita),
    }


def _criar_acao_giro_90(direcao, velocidade_frente, velocidade_reversa, parametros):
    velocidade_frente = _limitar_pwm(velocidade_frente)
    velocidade_reversa = _limitar_pwm(velocidade_reversa)

    if direcao == "esquerda":
        velocidade_esquerda, velocidade_direita = _compensar_re_esquerda(
            -velocidade_reversa,
            velocidade_frente,
            parametros,
        )
    else:
        velocidade_esquerda = velocidade_frente
        velocidade_direita = -velocidade_reversa

    acao = _criar_acao_diferencial(velocidade_esquerda, velocidade_direita)
    acao["modo"] = "giro_90"
    return acao


def _criar_acao_assistencia_curva(direcao, velocidade_frente, velocidade_reversa, parametros):
    velocidade_frente = _limitar_pwm(velocidade_frente)
    velocidade_reversa = _limitar_pwm(velocidade_reversa)
    if direcao == "esquerda":
        velocidade_esquerda, velocidade_direita = _compensar_re_esquerda(
            -velocidade_reversa,
            velocidade_frente,
            parametros,
        )
    else:
        velocidade_esquerda = velocidade_frente
        velocidade_direita = -velocidade_reversa

    acao = _criar_acao_diferencial(velocidade_esquerda, velocidade_direita)
    acao["modo"] = "assistencia_curva"
    return acao


def _ajustar_direcao_para_chassi(direcao, parametros):
    if direcao not in {"esquerda", "direita"}:
        return direcao
    if not getattr(parametros, "inverter_correcao", False):
        return direcao
    return "direita" if direcao == "esquerda" else "esquerda"


def _obter_parametros_giro_90(direcao, parametros):
    if direcao == "esquerda":
        return (
            parametros.velocidade_giro_90_esquerda,
            parametros.velocidade_reversa_giro_90_esquerda,
            parametros.tempo_giro_90_esquerda,
        )
    return (
        parametros.velocidade_giro_90,
        parametros.velocidade_reversa_giro_90,
        parametros.tempo_giro_90,
    )


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
    if dados_visao.get("curva_90_literal_esquerda"):
        return _ajustar_direcao_para_chassi("esquerda", parametros)
    if dados_visao.get("curva_90_literal_direita"):
        return _ajustar_direcao_para_chassi("direita", parametros)

    memoria_esquerda = bool(dados_visao.get("curva_90_memoria_esquerda"))
    memoria_direita = bool(dados_visao.get("curva_90_memoria_direita"))
    memoria_ativa = memoria_esquerda or memoria_direita
    pista_de_risco = bool(
        dados_visao.get("linha_toca_borda_esquerda")
        or dados_visao.get("linha_toca_borda_direita")
        or dados_visao.get("tempo_sem_linha", 0.0) > 0.0
        or dados_visao["confianca_linha"] < (parametros.limiar_confianca_curva_90_execucao + 0.05)
    )

    if dados_visao["confianca_linha"] < parametros.limiar_confianca_curva_90_execucao:
        if memoria_ativa and pista_de_risco:
            if memoria_esquerda:
                return _ajustar_direcao_para_chassi("esquerda", parametros)
            if memoria_direita:
                return _ajustar_direcao_para_chassi("direita", parametros)
        return None

    confianca_curva_90 = max(
        float(dados_visao.get("confianca_curva_90", 0.0)),
        float(dados_visao.get("confianca_curva_90_memoria", 0.0)) if pista_de_risco else 0.0,
    )
    if confianca_curva_90 < parametros.limiar_confianca_curva_90_execucao:
        return None
    if dados_visao.get("curva_90_esquerda"):
        return _ajustar_direcao_para_chassi("esquerda", parametros)
    if dados_visao.get("curva_90_direita"):
        return _ajustar_direcao_para_chassi("direita", parametros)
    if memoria_esquerda and pista_de_risco:
        return _ajustar_direcao_para_chassi("esquerda", parametros)
    if memoria_direita and pista_de_risco:
        return _ajustar_direcao_para_chassi("direita", parametros)
    return None


def _deve_executar_assistencia_curva(estado_controle, dados_visao, parametros, agora):
    if (agora - estado_controle.instante_ultima_assistencia_curva) < parametros.cooldown_assistencia_curva:
        return None
    if dados_visao["confianca_linha"] < parametros.limiar_confianca_assistencia_curva:
        return None

    confianca_lookahead = float(dados_visao.get("confianca_lookahead", 0.0))
    if confianca_lookahead < parametros.limiar_confianca_lookahead_assistencia_curva:
        return None

    erro_linha = float(dados_visao["erro_linha"])
    erro_lookahead = float(dados_visao.get("erro_lookahead", erro_linha))
    delta_antecipacao = abs(erro_lookahead - erro_linha)
    sinal_lookahead = 0
    if erro_lookahead > 0.0:
        sinal_lookahead = 1
    elif erro_lookahead < 0.0:
        sinal_lookahead = -1

    if sinal_lookahead == 0:
        return None

    if abs(erro_lookahead) < parametros.limiar_erro_lookahead_assistencia_curva:
        return None
    if delta_antecipacao < parametros.limiar_delta_erro_assistencia_curva:
        return None

    mesma_tendencia = (erro_linha == 0.0) or ((erro_linha * erro_lookahead) >= 0.0)
    if not mesma_tendencia and abs(erro_lookahead) < parametros.limiar_erro_lookahead_assistencia_curva_oposta:
        return None

    direcao = "direita" if sinal_lookahead > 0 else "esquerda"
    return _ajustar_direcao_para_chassi(direcao, parametros)


def _deve_encerrar_curva_90(estado_controle, dados_visao, parametros, agora):
    manobra = estado_controle.manobra_ativa
    if manobra is None or manobra.get("modo") != "giro_90":
        return False
    if (agora - estado_controle.tempo_entrada_estado) < parametros.tempo_minimo_giro_90:
        return False
    if not dados_visao["linha_encontrada"]:
        return False
    if dados_visao["confianca_linha"] < parametros.limiar_confianca_retomada_giro_90:
        return False

    erro_linha = abs(float(dados_visao["erro_linha"]))
    erro_lookahead = abs(float(dados_visao.get("erro_lookahead", dados_visao["erro_linha"])))
    if erro_linha > parametros.limiar_erro_retomada_giro_90:
        return False
    if erro_lookahead > parametros.limiar_erro_lookahead_retomada_giro_90:
        return False
    return True


def _deve_encerrar_assistencia_curva(estado_controle, dados_visao, parametros, agora):
    manobra = estado_controle.manobra_ativa
    if manobra is None or manobra.get("modo") != "assistencia_curva":
        return False

    if (agora - estado_controle.tempo_entrada_estado) < parametros.tempo_minimo_assistencia_curva:
        return False
    if not dados_visao["linha_encontrada"]:
        return False
    if dados_visao["confianca_linha"] < parametros.limiar_confianca_retomada_assistencia_curva:
        return False

    erro_linha = abs(float(dados_visao["erro_linha"]))
    erro_lookahead = abs(float(dados_visao.get("erro_lookahead", dados_visao["erro_linha"])))
    if erro_linha > parametros.limiar_erro_retomada_assistencia_curva:
        return False
    if erro_lookahead > parametros.limiar_erro_lookahead_retomada_assistencia_curva:
        return False
    return True


def _calcular_acao_pid(dados_visao, pid, parametros, tempo_atual):
    erro_linha = float(dados_visao["erro_linha"])
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
    erro_controle = _limitar(erro_linha + (erro_lookahead * ganho_lookahead), -1.0, 1.0)
    erro_abs = abs(erro_controle)
    erro_referencia_curva = _erro_referencia_curva(
        erro_controle,
        erro_lookahead,
        confianca_lookahead,
        parametros,
    )
    forcar_pivo = _deve_forcar_pivo_curva_fechada(
        erro_controle,
        erro_lookahead,
        confianca_lookahead,
        dados_visao,
        parametros,
    )
    ganho_correcao, proporcao_curva = _ganho_correcao_dinamico(erro_referencia_curva, parametros)

    if erro_abs <= parametros.limiar_erro_centralizado:
        pid.estabilizar_centro()
        velocidade_reta = _limitar(
            parametros.velocidade_base,
            parametros.velocidade_minima,
            parametros.velocidade_maxima,
        )
        return _criar_acao_diferencial(velocidade_reta, velocidade_reta), 0.0, erro_controle

    correcao_pid = pid.calcular(erro_controle, tempo_atual)
    correcao_pid = _limitar(correcao_pid, -parametros.correcao_maxima, parametros.correcao_maxima)
    correcao_pid *= ganho_correcao

    if forcar_pivo or erro_referencia_curva >= parametros.limiar_erro_pivo:
        if erro_controle >= 0.0:
            velocidade_esquerda, velocidade_direita = _obter_pivo_direita_fixo(parametros)
            return _criar_acao_diferencial(velocidade_esquerda, velocidade_direita), correcao_pid, erro_controle
        velocidade_esquerda, velocidade_direita = _obter_pivo_esquerda_fixo(parametros)
        return _criar_acao_diferencial(velocidade_esquerda, velocidade_direita), correcao_pid, erro_controle

    velocidade_cruzeiro = _interpolar(
        parametros.velocidade_base,
        parametros.velocidade_curva,
        proporcao_curva,
    )
    bonus_tracao_externa = _interpolar(0.0, parametros.bonus_tracao_externa, proporcao_curva)
    bonus_freio_interno = _interpolar(0.0, parametros.bonus_freio_interno, proporcao_curva)

    velocidade_esquerda_bruta = velocidade_cruzeiro + correcao_pid
    velocidade_direita_bruta = velocidade_cruzeiro - correcao_pid

    if erro_controle >= 0.0:
        velocidade_esquerda_bruta += bonus_tracao_externa
        velocidade_direita_bruta -= bonus_freio_interno
    else:
        velocidade_esquerda_bruta -= bonus_freio_interno
        velocidade_direita_bruta += bonus_tracao_externa

    limiar_reversao = float(parametros.limiar_erro_reversao)
    if erro_controle < 0.0:
        limiar_reversao = min(
            limiar_reversao,
            float(parametros.limiar_erro_reversao_esquerda),
        )

    if erro_referencia_curva < limiar_reversao:
        velocidade_esquerda, velocidade_direita = _aplicar_pisos_frente(
            velocidade_esquerda_bruta,
            velocidade_direita_bruta,
            erro_controle,
            parametros,
        )
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

    velocidade_esquerda, velocidade_direita = _compensar_re_esquerda(
        velocidade_esquerda,
        velocidade_direita,
        parametros,
    )
    return _criar_acao_diferencial(velocidade_esquerda, velocidade_direita), correcao_pid, erro_controle


def _atualizar_controle(estado_controle, dados_visao, pid, parametros, agora):
    correcao_pid = 0.0

    if estado_controle.estado_atual == ESTADO_INICIANDO:
        if (agora - estado_controle.tempo_entrada_estado) < parametros.tempo_inicial:
            return _criar_acao_parar(), "espera de seguranca na partida", correcao_pid
        estado_controle.estado_atual = ESTADO_SEM_LINHA
        estado_controle.tempo_entrada_estado = agora

    if estado_controle.manobra_ativa is not None and agora < estado_controle.manobra_ativa_ate:
        modo_manobra = estado_controle.manobra_ativa.get("modo")
        if modo_manobra == "giro_90":
            if _deve_encerrar_curva_90(estado_controle, dados_visao, parametros, agora):
                estado_controle.manobra_ativa = None
                estado_controle.manobra_ativa_ate = 0.0
                pid.reiniciar(suave=True)
            else:
                return estado_controle.manobra_ativa, "executando curva de 90 graus", correcao_pid
        elif modo_manobra == "assistencia_curva":
            if _deve_encerrar_assistencia_curva(estado_controle, dados_visao, parametros, agora):
                estado_controle.manobra_ativa = None
                estado_controle.manobra_ativa_ate = 0.0
                pid.reiniciar(suave=True)
            else:
                return estado_controle.manobra_ativa, "assistencia temporizada de curva", correcao_pid
        else:
            return estado_controle.manobra_ativa, "manobra temporizada", correcao_pid

    if estado_controle.manobra_ativa is not None and agora >= estado_controle.manobra_ativa_ate:
        estado_controle.manobra_ativa = None
        estado_controle.manobra_ativa_ate = 0.0

    _atualizar_memoria_linha(estado_controle, dados_visao, parametros, agora)

    linha_minima = _linha_minimamente_valida(dados_visao, parametros)
    linha_em_risco = _linha_em_risco(dados_visao, parametros)
    linha_confiavel = _linha_confiavel_para_retomada(dados_visao, parametros)
    retomada_confirmada = (
        estado_controle.quadros_confiaveis_consecutivos >= parametros.quadros_confirmacao_retomada
    )
    em_recuperacao = estado_controle.estado_atual in {ESTADO_SEM_LINHA, ESTADO_BUSCA, ESTADO_CONTENCAO}

    if not linha_minima:
        if estado_controle.estado_atual != ESTADO_BUSCA:
            estado_controle.estado_atual = ESTADO_BUSCA
            estado_controle.tempo_entrada_estado = agora
        pid.reiniciar(suave=True)
        acao_busca, motivo_busca = _calcular_acao_busca_sem_linha(
            estado_controle,
            dados_visao,
            parametros,
            agora,
        )
        return acao_busca, motivo_busca, correcao_pid

    if linha_em_risco or (em_recuperacao and (not linha_confiavel or not retomada_confirmada)):
        if estado_controle.estado_atual != ESTADO_CONTENCAO:
            estado_controle.estado_atual = ESTADO_CONTENCAO
            estado_controle.tempo_entrada_estado = agora
        pid.reiniciar(suave=True)
        acao_contencao, motivo_contencao, correcao_pid = _calcular_acao_contencao(
            dados_visao,
            estado_controle,
            parametros,
        )
        if em_recuperacao and linha_confiavel and not retomada_confirmada:
            motivo_contencao = "confirmando retomada antes de liberar velocidade normal"
        return acao_contencao, motivo_contencao, correcao_pid

    direcao_curva_90 = _deve_executar_curva_90(estado_controle, dados_visao, parametros, agora)
    if direcao_curva_90 is not None:
        estado_controle.estado_atual = ESTADO_MANOBRA_90
        estado_controle.tempo_entrada_estado = agora
        estado_controle.instante_ultimo_giro_90 = agora
        velocidade_frente_giro_90, velocidade_reversa_giro_90, tempo_giro_90 = _obter_parametros_giro_90(
            direcao_curva_90,
            parametros,
        )
        estado_controle.manobra_ativa = _criar_acao_giro_90(
            direcao_curva_90,
            velocidade_frente_giro_90,
            velocidade_reversa_giro_90,
            parametros,
        )
        estado_controle.manobra_ativa_ate = agora + tempo_giro_90
        pid.reiniciar(suave=False)
        return estado_controle.manobra_ativa, f"curva de 90 graus {direcao_curva_90}", correcao_pid

    direcao_assistencia_curva = _deve_executar_assistencia_curva(
        estado_controle,
        dados_visao,
        parametros,
        agora,
    )
    if direcao_assistencia_curva is not None:
        estado_controle.estado_atual = ESTADO_ASSISTENCIA_CURVA
        estado_controle.tempo_entrada_estado = agora
        estado_controle.instante_ultima_assistencia_curva = agora
        estado_controle.manobra_ativa = _criar_acao_assistencia_curva(
            direcao_assistencia_curva,
            parametros.velocidade_assistencia_curva,
            parametros.velocidade_reversa_assistencia_curva,
            parametros,
        )
        estado_controle.manobra_ativa_ate = agora + parametros.tempo_assistencia_curva
        pid.reiniciar(suave=False)
        return (
            estado_controle.manobra_ativa,
            f"assistencia de curva para {direcao_assistencia_curva}",
            correcao_pid,
        )

    if estado_controle.estado_atual != ESTADO_SEGUINDO:
        estado_controle.estado_atual = ESTADO_SEGUINDO
        estado_controle.tempo_entrada_estado = agora
        pid.reiniciar(suave=True)

    acao, correcao_pid, erro_controle = _calcular_acao_pid(
        dados_visao,
        pid,
        parametros,
        agora,
    )

    if acao["tipo"] == "D" and acao["velocidade_esquerda"] == acao["velocidade_direita"]:
        motivo = "linha centralizada, mantendo reto"
    elif abs(erro_controle) >= parametros.limiar_erro_pivo:
        motivo = "correcao agressiva para reenquadrar"
    else:
        motivo = "correcao de linha com PID"

    return acao, motivo, correcao_pid


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
        (
            "borda="
            f"{'E' if dados_visao.get('linha_toca_borda_esquerda') else ('D' if dados_visao.get('linha_toca_borda_direita') else 'NAO')} "
            f"sem_linha={dados_visao.get('tempo_sem_linha', 0.0):.2f}s "
            f"ret={estado_controle.quadros_confiaveis_consecutivos}"
        ),
        f"verde={'SIM' if dados_visao.get('verde_detectado') else 'NAO'} confV={dados_visao.get('confianca_verde', 0.0):.2f}",
        f"pid_p={pid.erro_proporcional:+.3f} pid_i={pid.erro_integral:+.3f} pid_d={pid.erro_derivativo:+.3f}",
        f"correcao_pid={correcao_pid:+.2f}",
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
                f"verde={'SIM' if dados_visao.get('verde_detectado') else 'NAO'}",
                f"conf={dados_visao['confianca_linha']:.2f}",
                (
                    "borda="
                    f"{'E' if dados_visao.get('linha_toca_borda_esquerda') else ('D' if dados_visao.get('linha_toca_borda_direita') else 'NAO')}"
                ),
                f"sem_linha={dados_visao.get('tempo_sem_linha', 0.0):.2f}s",
                f"ret={estado_controle.quadros_confiaveis_consecutivos}",
                f"pid={correcao_pid:+.2f}",
                f"motivo={motivo}",
            ]
        ),
        flush=True,
    )


def analisar_argumentos():
    analisador = argparse.ArgumentParser(
        description="Modo controle simplificado: correcao de linha, curva de 90 e verde.",
    )

    analisador.add_argument("--device", type=int, default=None, help="Indice da camera USB.")
    analisador.add_argument("--width", type=int, default=640, help="Largura da imagem.")
    analisador.add_argument("--height", type=int, default=480, help="Altura da imagem.")
    analisador.add_argument("--fps", type=int, default=30, help="Taxa de quadros desejada.")
    analisador.add_argument(
        "--autofocus",
        dest="autofocus",
        action="store_true",
        default=True,
        help="Liga foco automatico continuo quando a camera suportar.",
    )
    analisador.add_argument(
        "--no-autofocus",
        dest="autofocus",
        action="store_false",
        help="Desliga foco automatico e mantem o foco manual/padrao.",
    )
    analisador.add_argument(
        "--focus-value",
        type=float,
        default=None,
        help="Foco manual. Webcam USB costuma usar 0-255; camera CSI usa LensPosition.",
    )
    analisador.add_argument(
        "--camera-buffer-drop",
        type=int,
        default=2,
        help="Quantidade de frames antigos descartados por leitura para reduzir atraso.",
    )

    analisador.add_argument("--show", action="store_true", help="Mostra janela de debug local.")
    analisador.add_argument("--no-show", action="store_true", help="Nao mostra janela local.")
    analisador.add_argument("--debug-path", default=None, help="Salva continuamente o ultimo quadro de debug.")
    analisador.add_argument(
        "--debug-write-interval",
        type=float,
        default=0.25,
        help="Intervalo minimo entre escritas em --debug-path.",
    )
    analisador.add_argument("--stream", action="store_true", help="Publica stream MJPEG no navegador.")
    analisador.add_argument("--stream-host", default="0.0.0.0", help="Host do stream HTTP.")
    analisador.add_argument("--stream-port", type=int, default=8080, help="Porta do stream HTTP.")
    analisador.add_argument(
        "--stream-jpeg-quality",
        type=int,
        default=70,
        help="Qualidade JPEG do stream (menor = menos latencia).",
    )
    analisador.add_argument("--print-every", type=float, default=0.20, help="Intervalo minimo para logs.")

    analisador.add_argument("--roi", type=float, default=0.48)
    analisador.add_argument("--limiar-binario", type=int, default=None)
    analisador.add_argument(
        "--invert",
        action="store_true",
        help="Use apenas quando a linha for clara em fundo escuro.",
    )
    analisador.add_argument(
        "--sem-limiar-adaptativo",
        action="store_true",
        help="Desliga o limiar adaptativo de apoio contra brilho.",
    )
    analisador.add_argument("--bloco-limiar-adaptativo", type=int, default=41)
    analisador.add_argument("--constante-limiar-adaptativo", type=int, default=9)
    analisador.add_argument("--area-minima-contorno", type=int, default=180)
    analisador.add_argument("--area-minima-linha", type=int, default=320)
    analisador.add_argument("--suavizacao-erro", type=float, default=0.40)
    analisador.add_argument("--limiar-confianca", type=float, default=0.10)
    analisador.add_argument(
        "--limiar-confianca-cautela",
        type=float,
        default=0.22,
        help="Abaixo deste valor o robo entra em contencao e reduz velocidade antes de perder a linha.",
    )
    analisador.add_argument(
        "--limiar-confianca-retomada",
        type=float,
        default=0.30,
        help="Confianca minima para considerar que a linha voltou de forma segura.",
    )
    analisador.add_argument("--faixa-base-contorno", type=float, default=0.16)
    analisador.add_argument("--margem-lateral-descarte", type=float, default=0.10)
    analisador.add_argument("--lookahead-fracao", type=float, default=0.42)
    analisador.add_argument("--lookahead-minimo-pixels", type=int, default=18)
    analisador.add_argument("--limiar-confianca-curva-90", type=float, default=0.18)
    analisador.add_argument("--limiar-confianca-lookahead-curva-90", type=float, default=0.14)
    analisador.add_argument("--limiar-erro-lookahead-curva-90", type=float, default=0.36)
    analisador.add_argument("--limiar-delta-erro-curva-90", type=float, default=0.10)
    analisador.add_argument("--limiar-erro-base-curva-90", type=float, default=0.30)
    analisador.add_argument("--faixa-superior-curva-90", type=float, default=0.48)
    analisador.add_argument("--faixa-inferior-curva-90", type=float, default=0.24)
    analisador.add_argument("--densidade-lateral-curva-90", type=float, default=0.16)
    analisador.add_argument("--densidade-oposta-max-curva-90", type=float, default=0.08)
    analisador.add_argument("--densidade-base-centro-curva-90", type=float, default=0.10)
    analisador.add_argument("--largura-janela-branca-curva-90", type=float, default=0.16)
    analisador.add_argument("--largura-janela-lateral-curva-90", type=float, default=0.28)
    analisador.add_argument("--densidade-frontal-max-curva-90", type=float, default=0.08)
    analisador.add_argument("--roi-verde", type=float, default=0.75)
    analisador.add_argument("--verde-h-min", type=int, default=35)
    analisador.add_argument("--verde-h-max", type=int, default=95)
    analisador.add_argument("--verde-s-min", type=int, default=60)
    analisador.add_argument("--verde-v-min", type=int, default=45)
    analisador.add_argument("--area-minima-verde", type=int, default=180)

    analisador.add_argument("--kp", type=float, default=160.0)
    analisador.add_argument("--ki", type=float, default=10.0)
    analisador.add_argument("--kd", type=float, default=42.0)
    analisador.add_argument("--integral-max", type=float, default=0.85)
    analisador.add_argument("--dt-minimo", type=float, default=0.01)
    analisador.add_argument("--alpha-derivada", type=float, default=0.25)
    analisador.add_argument("--correcao-maxima", type=float, default=235.0)
    analisador.add_argument("--ganho-lookahead-suave", type=float, default=0.32)
    analisador.add_argument("--ganho-lookahead-forte", type=float, default=1.22)
    analisador.add_argument("--lookahead-erro-minimo", type=float, default=0.10)
    analisador.add_argument("--lookahead-erro-maximo", type=float, default=0.48)
    analisador.add_argument("--fator-correcao-forte", type=float, default=1.36)
    analisador.add_argument("--fator-antecipacao-velocidade", type=float, default=1.35)
    analisador.add_argument("--limiar-confianca-lookahead-velocidade", type=float, default=0.18)
    analisador.add_argument("--bonus-tracao-externa", type=float, default=8.0)
    analisador.add_argument("--bonus-freio-interno", type=float, default=26.0)

    analisador.add_argument("--velocidade-base", type=int, default=84)
    analisador.add_argument("--velocidade-curva", type=int, default=60)
    analisador.add_argument("--velocidade-minima", type=int, default=50)
    analisador.add_argument("--velocidade-minima-interna", type=int, default=12)
    analisador.add_argument("--velocidade-maxima", type=int, default=135)
    analisador.add_argument("--limiar-erro-centralizado", type=float, default=0.035)
    analisador.add_argument("--limiar-erro-reversao", type=float, default=0.46)
    analisador.add_argument(
        "--limiar-erro-reversao-esquerda",
        type=float,
        default=0.28,
        help="Permite que a esquerda entre em re mais cedo para fechar curva quando esse lado tem mais dificuldade mecanica.",
    )
    analisador.add_argument(
        "--bonus-re-esquerda",
        type=float,
        default=34.0,
        help="Empurrao extra de PWM quando o lado esquerdo precisa entrar em re para fechar a curva.",
    )
    analisador.add_argument(
        "--piso-re-esquerda",
        type=int,
        default=122,
        help="PWM minimo absoluto para a esquerda quando esse lado estiver em re numa curva para a esquerda.",
    )
    analisador.add_argument(
        "--limiar-erro-contencao",
        type=float,
        default=0.18,
        help="Erro lateral que ativa a contencao preventiva antes da perda total da linha.",
    )
    analisador.add_argument(
        "--limiar-erro-contencao-pivo",
        type=float,
        default=0.30,
        help="Erro lateral que bloqueia avanco e força reenquadramento em pivo.",
    )
    analisador.add_argument(
        "--ganho-lookahead-contencao",
        type=float,
        default=0.35,
        help="Peso do lookahead durante a contencao preventiva.",
    )
    analisador.add_argument("--ganho-correcao-contencao", type=float, default=118.0)
    analisador.add_argument("--velocidade-contencao", type=int, default=92)
    analisador.add_argument("--velocidade-maxima-contencao", type=int, default=128)
    analisador.add_argument("--velocidade-reversa-contencao", type=int, default=90)
    analisador.add_argument("--velocidade-busca-linha", type=int, default=86)
    analisador.add_argument("--velocidade-reversa-busca-linha", type=int, default=82)
    analisador.add_argument(
        "--tempo-memoria-busca",
        type=float,
        default=1.10,
        help="Tempo maximo desde a ultima linha confiavel em que ainda vale a pena buscar sem avancar.",
    )
    analisador.add_argument(
        "--tempo-maximo-busca-sem-linha",
        type=float,
        default=1.40,
        help="Limite total de busca controlada antes de parar completamente.",
    )
    analisador.add_argument("--tempo-busca-mesmo-lado", type=float, default=0.35)
    analisador.add_argument("--tempo-varredura-busca", type=float, default=0.28)
    analisador.add_argument(
        "--quadros-confirmacao-retomada",
        type=int,
        default=4,
        help="Quantidade de quadros confiaveis seguidos antes de liberar a volta ao modo normal.",
    )
    analisador.add_argument("--limiar-erro-pivo", type=float, default=0.62)
    analisador.add_argument("--velocidade-pivo", type=int, default=150)
    analisador.add_argument("--bonus-velocidade-pivo", type=int, default=0)
    analisador.add_argument(
        "--limiar-erro-pivo-forcado",
        type=float,
        default=0.42,
        help="Erro de controle a partir do qual a curva entra direto em pivo seco.",
    )
    analisador.add_argument(
        "--limiar-confianca-lookahead-pivo-forcado",
        type=float,
        default=0.12,
        help="Confianca minima do lookahead para acionar pivo forcado.",
    )
    analisador.add_argument(
        "--limiar-erro-lookahead-pivo-forcado",
        type=float,
        default=0.28,
        help="Erro do lookahead que aciona pivo forcado em curva muito fechada.",
    )
    analisador.add_argument(
        "--limiar-erro-lookahead-borda-pivo",
        type=float,
        default=0.18,
        help="Erro minimo do lookahead para forcar pivo quando a linha encostar na borda.",
    )
    analisador.add_argument(
        "--velocidade-pivo-esquerda-frente",
        type=int,
        default=150,
        help="PWM da direita para pivo no proprio eixo quando a curva for para a esquerda.",
    )
    analisador.add_argument(
        "--velocidade-pivo-esquerda-reversa",
        type=int,
        default=210,
        help="PWM de re da esquerda para pivo no proprio eixo quando a curva for para a esquerda.",
    )
    analisador.add_argument(
        "--velocidade-pivo-direita-frente",
        type=int,
        default=150,
        help="PWM da esquerda para pivo no proprio eixo quando a curva for para a direita.",
    )
    analisador.add_argument(
        "--velocidade-pivo-direita-reversa",
        type=int,
        default=150,
        help="PWM de re da direita para pivo no proprio eixo quando a curva for para a direita.",
    )
    analisador.add_argument("--velocidade-giro-90", type=int, default=144)
    analisador.add_argument("--velocidade-reversa-giro-90", type=int, default=136)
    analisador.add_argument("--velocidade-giro-90-esquerda", type=int, default=158)
    analisador.add_argument("--velocidade-reversa-giro-90-esquerda", type=int, default=176)
    analisador.add_argument("--tempo-giro-90", type=float, default=0.56)
    analisador.add_argument("--tempo-giro-90-esquerda", type=float, default=0.62)
    analisador.add_argument("--tempo-minimo-giro-90", type=float, default=0.16)
    analisador.add_argument("--cooldown-giro-90", type=float, default=0.70)
    analisador.add_argument("--limiar-confianca-curva-90-execucao", type=float, default=0.22)
    analisador.add_argument("--limiar-confianca-retomada-giro-90", type=float, default=0.26)
    analisador.add_argument("--limiar-erro-retomada-giro-90", type=float, default=0.22)
    analisador.add_argument("--limiar-erro-lookahead-retomada-giro-90", type=float, default=0.34)
    analisador.add_argument("--velocidade-assistencia-curva", type=int, default=112)
    analisador.add_argument("--velocidade-reversa-assistencia-curva", type=int, default=102)
    analisador.add_argument("--tempo-assistencia-curva", type=float, default=0.18)
    analisador.add_argument("--tempo-minimo-assistencia-curva", type=float, default=0.06)
    analisador.add_argument("--cooldown-assistencia-curva", type=float, default=0.16)
    analisador.add_argument("--limiar-confianca-assistencia-curva", type=float, default=0.22)
    analisador.add_argument("--limiar-confianca-lookahead-assistencia-curva", type=float, default=0.14)
    analisador.add_argument("--limiar-erro-lookahead-assistencia-curva", type=float, default=0.12)
    analisador.add_argument("--limiar-erro-lookahead-assistencia-curva-oposta", type=float, default=0.22)
    analisador.add_argument("--limiar-delta-erro-assistencia-curva", type=float, default=0.07)
    analisador.add_argument("--limiar-confianca-retomada-assistencia-curva", type=float, default=0.20)
    analisador.add_argument("--limiar-erro-retomada-assistencia-curva", type=float, default=0.14)
    analisador.add_argument("--limiar-erro-lookahead-retomada-assistencia-curva", type=float, default=0.22)
    analisador.set_defaults(inverter_correcao=False)
    analisador.add_argument(
        "--inverter-correcao",
        dest="inverter_correcao",
        action="store_true",
        help="Inverte o sentido da correcao lateral quando o robo corrigir para o lado errado.",
    )
    analisador.add_argument(
        "--nao-inverter-correcao",
        dest="inverter_correcao",
        action="store_false",
        help="Usa o sentido padrao de correcao lateral sem inverter esquerda/direita.",
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
        usar_limiar_adaptativo=not parametros.sem_limiar_adaptativo,
        bloco_limiar_adaptativo=parametros.bloco_limiar_adaptativo,
        constante_limiar_adaptativo=parametros.constante_limiar_adaptativo,
        area_minima_contorno=parametros.area_minima_contorno,
        area_minima_linha=parametros.area_minima_linha,
        suavizacao_erro=parametros.suavizacao_erro,
        limiar_confianca=parametros.limiar_confianca,
        faixa_base_contorno=parametros.faixa_base_contorno,
        margem_lateral_descarte=parametros.margem_lateral_descarte,
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
        largura_janela_branca_curva_90=parametros.largura_janela_branca_curva_90,
        largura_janela_lateral_curva_90=parametros.largura_janela_lateral_curva_90,
        densidade_frontal_max_curva_90=parametros.densidade_frontal_max_curva_90,
        roi_verde=parametros.roi_verde,
        verde_h_min=parametros.verde_h_min,
        verde_h_max=parametros.verde_h_max,
        verde_s_min=parametros.verde_s_min,
        verde_v_min=parametros.verde_v_min,
        area_minima_verde=parametros.area_minima_verde,
    )


def principal():
    parametros = analisar_argumentos()

    parametros.velocidade_base = _limitar_pwm(parametros.velocidade_base)
    parametros.velocidade_curva = _limitar_pwm(parametros.velocidade_curva)
    parametros.velocidade_minima = _limitar_pwm(parametros.velocidade_minima)
    parametros.velocidade_minima_interna = _limitar_pwm(parametros.velocidade_minima_interna)
    parametros.velocidade_maxima = _limitar_pwm(parametros.velocidade_maxima)
    parametros.velocidade_pivo = _limitar_pwm(parametros.velocidade_pivo)
    parametros.velocidade_pivo_esquerda_frente = _limitar_pwm(parametros.velocidade_pivo_esquerda_frente)
    parametros.velocidade_pivo_esquerda_reversa = _limitar_pwm(parametros.velocidade_pivo_esquerda_reversa)
    parametros.velocidade_pivo_direita_frente = _limitar_pwm(parametros.velocidade_pivo_direita_frente)
    parametros.velocidade_pivo_direita_reversa = _limitar_pwm(parametros.velocidade_pivo_direita_reversa)
    parametros.velocidade_giro_90 = _limitar_pwm(parametros.velocidade_giro_90)
    parametros.velocidade_reversa_giro_90 = _limitar_pwm(parametros.velocidade_reversa_giro_90)
    parametros.velocidade_giro_90_esquerda = _limitar_pwm(parametros.velocidade_giro_90_esquerda)
    parametros.velocidade_reversa_giro_90_esquerda = _limitar_pwm(
        parametros.velocidade_reversa_giro_90_esquerda
    )
    parametros.velocidade_assistencia_curva = _limitar_pwm(parametros.velocidade_assistencia_curva)
    parametros.velocidade_reversa_assistencia_curva = _limitar_pwm(parametros.velocidade_reversa_assistencia_curva)
    parametros.velocidade_contencao = _limitar_pwm(parametros.velocidade_contencao)
    parametros.velocidade_maxima_contencao = _limitar_pwm(parametros.velocidade_maxima_contencao)
    parametros.velocidade_reversa_contencao = _limitar_pwm(parametros.velocidade_reversa_contencao)
    parametros.velocidade_busca_linha = _limitar_pwm(parametros.velocidade_busca_linha)
    parametros.velocidade_reversa_busca_linha = _limitar_pwm(parametros.velocidade_reversa_busca_linha)
    parametros.piso_re_esquerda = _limitar_pwm(parametros.piso_re_esquerda)
    parametros.bonus_re_esquerda = max(0.0, float(parametros.bonus_re_esquerda))
    parametros.quadros_confirmacao_retomada = max(1, int(parametros.quadros_confirmacao_retomada))
    parametros.tempo_memoria_busca = max(0.0, float(parametros.tempo_memoria_busca))
    parametros.tempo_maximo_busca_sem_linha = max(0.0, float(parametros.tempo_maximo_busca_sem_linha))
    parametros.tempo_busca_mesmo_lado = max(0.0, float(parametros.tempo_busca_mesmo_lado))
    parametros.tempo_varredura_busca = max(0.05, float(parametros.tempo_varredura_busca))
    parametros.ganho_lookahead_contencao = _limitar(float(parametros.ganho_lookahead_contencao), 0.0, 1.0)
    parametros.ganho_correcao_contencao = max(0.0, float(parametros.ganho_correcao_contencao))
    parametros.limiar_confianca_cautela = _limitar(float(parametros.limiar_confianca_cautela), 0.0, 1.0)
    parametros.limiar_confianca_retomada = _limitar(float(parametros.limiar_confianca_retomada), 0.0, 1.0)
    parametros.limiar_erro_contencao = _limitar(float(parametros.limiar_erro_contencao), 0.0, 1.0)
    parametros.limiar_erro_contencao_pivo = _limitar(float(parametros.limiar_erro_contencao_pivo), 0.0, 1.0)
    parametros.limiar_erro_pivo_forcado = _limitar(float(parametros.limiar_erro_pivo_forcado), 0.0, 1.0)
    parametros.limiar_confianca_lookahead_pivo_forcado = _limitar(
        float(parametros.limiar_confianca_lookahead_pivo_forcado),
        0.0,
        1.0,
    )
    parametros.limiar_erro_lookahead_pivo_forcado = _limitar(
        float(parametros.limiar_erro_lookahead_pivo_forcado),
        0.0,
        1.0,
    )
    parametros.limiar_erro_lookahead_borda_pivo = _limitar(
        float(parametros.limiar_erro_lookahead_borda_pivo),
        0.0,
        1.0,
    )

    if parametros.velocidade_minima > parametros.velocidade_maxima:
        parametros.velocidade_minima, parametros.velocidade_maxima = (
            parametros.velocidade_maxima,
            parametros.velocidade_minima,
        )
    parametros.velocidade_minima_interna = int(
        _limitar(
            parametros.velocidade_minima_interna,
            0,
            parametros.velocidade_minima,
        )
    )
    parametros.piso_re_esquerda = int(
        _limitar(
            parametros.piso_re_esquerda,
            0,
            parametros.velocidade_maxima,
        )
    )
    parametros.velocidade_maxima_contencao = int(
        _limitar(
            parametros.velocidade_maxima_contencao,
            0,
            parametros.velocidade_maxima,
        )
    )
    parametros.velocidade_contencao = int(
        _limitar(
            parametros.velocidade_contencao,
            0,
            max(parametros.velocidade_maxima_contencao, 1),
        )
    )
    parametros.velocidade_reversa_contencao = int(
        _limitar(
            parametros.velocidade_reversa_contencao,
            0,
            parametros.velocidade_maxima_contencao,
        )
    )
    parametros.velocidade_busca_linha = int(
        _limitar(
            parametros.velocidade_busca_linha,
            0,
            parametros.velocidade_maxima_contencao,
        )
    )
    parametros.velocidade_reversa_busca_linha = int(
        _limitar(
            parametros.velocidade_reversa_busca_linha,
            0,
            parametros.velocidade_maxima_contencao,
        )
    )
    parametros.limiar_confianca_cautela = max(
        parametros.limiar_confianca,
        parametros.limiar_confianca_cautela,
    )
    parametros.limiar_confianca_retomada = max(
        parametros.limiar_confianca_cautela,
        parametros.limiar_confianca_retomada,
    )
    parametros.limiar_erro_contencao_pivo = max(
        parametros.limiar_erro_contencao,
        parametros.limiar_erro_contencao_pivo,
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
            autofocus=parametros.autofocus,
            focus_value=parametros.focus_value,
            quadros_descartados_por_leitura=parametros.camera_buffer_drop,
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
        f"Camera pronta via {camera['backend']} ({camera['device']}) em "
        f"{parametros.width}x{parametros.height}@{parametros.fps} "
        f"| foco={camera.get('descricao_foco', 'padrao')} "
        f"| drop={camera.get('quadros_descartados_por_leitura', 0)}."
    )
    if ser is None:
        print("Controle em modo visao-only (serial indisponivel).")
    else:
        print(f"Serial ativa em {parametros.port} @ {parametros.baud}.")

    if exibir_janela:
        cv2.namedWindow("controle_debug", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("controle_debug", max(640, parametros.width), max(480, parametros.height))

    if parametros.stream:
        servidor_stream = ServidorStream(
            host=parametros.stream_host,
            port=parametros.stream_port,
            qualidade_jpeg=parametros.stream_jpeg_quality,
        )
        servidor_stream.iniciar()
        print(
            f"Stream ativo em http://127.0.0.1:{parametros.stream_port} (local) e http://IP_DA_RASPBERRY:{parametros.stream_port}."
        )

    instante_ultimo_log = 0.0
    instante_ultima_gravacao_debug = 0.0

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

            if parametros.debug_path and (
                parametros.debug_write_interval <= 0
                or (agora - instante_ultima_gravacao_debug) >= parametros.debug_write_interval
            ):
                cv2.imwrite(parametros.debug_path, quadro_debug)
                instante_ultima_gravacao_debug = agora

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

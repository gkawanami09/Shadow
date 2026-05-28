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
ESTADO_INTERSECAO = "NAVEGANDO_INTERSECAO"


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
    instante_inicio_similaridade_alta: float = -999.0
    instante_ultimo_destravamento: float = -999.0
    indicios_curva_90_esquerda: int = 0
    indicios_curva_90_direita: int = 0
    indicios_intersecao: int = 0
    indicios_verde_esquerda: int = 0
    indicios_verde_direita: int = 0
    instante_ultima_intersecao: float = -999.0


class ControladorPID:
    def __init__(
        self,
        kP,
        kI,
        kD,
        limite_integral,
        dt_minimo,
        alpha_derivada,
        alpha_correcao_saida,
        delta_correcao_maxima,
    ):
        self.kP = float(kP)
        self.kI = float(kI)
        self.kD = float(kD)
        self.limite_integral = float(abs(limite_integral))
        self.dt_minimo = float(max(1e-4, dt_minimo))
        self.alpha_derivada = float(_limitar(alpha_derivada, 0.0, 1.0))
        self.alpha_correcao_saida = float(_limitar(alpha_correcao_saida, 0.0, 0.95))
        self.delta_correcao_maxima = float(max(0.0, delta_correcao_maxima))

        self.erro_proporcional = 0.0
        self.erro_integral = 0.0
        self.erro_derivativo = 0.0
        self.erro_anterior = 0.0
        self.tempo_anterior = None
        self._derivada_filtrada = 0.0
        self._correcao_suavizada = 0.0

    def reiniciar(self, suave=False):
        self.erro_proporcional = 0.0
        self.erro_derivativo = 0.0
        if suave:
            self.erro_integral *= 0.25
            self._correcao_suavizada *= 0.25
        else:
            self.erro_integral = 0.0
            self._derivada_filtrada = 0.0
            self.erro_anterior = 0.0
            self.tempo_anterior = None
            self._correcao_suavizada = 0.0

    def estabilizar_centro(self):
        self.erro_proporcional = 0.0
        self.erro_integral *= 0.5
        if abs(self.erro_integral) < 1e-4:
            self.erro_integral = 0.0
        self.erro_derivativo = 0.0
        self.erro_anterior = 0.0
        self.tempo_anterior = None
        self._derivada_filtrada = 0.0
        self._correcao_suavizada *= 0.35
        if abs(self._correcao_suavizada) < 1e-3:
            self._correcao_suavizada = 0.0

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

    def suavizar_correcao(self, correcao_alvo):
        correcao_alvo = float(correcao_alvo)

        if self.delta_correcao_maxima > 0.0:
            delta = _limitar(
                correcao_alvo - self._correcao_suavizada,
                -self.delta_correcao_maxima,
                self.delta_correcao_maxima,
            )
            correcao_alvo = self._correcao_suavizada + delta

        self._correcao_suavizada = (
            self.alpha_correcao_saida * self._correcao_suavizada
            + (1.0 - self.alpha_correcao_saida) * correcao_alvo
        )
        if abs(self._correcao_suavizada) < 1e-4:
            self._correcao_suavizada = 0.0
        return float(self._correcao_suavizada)


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


def _deve_destravar_por_similaridade(estado_controle, dados_visao, parametros, agora):
    if not dados_visao["linha_encontrada"]:
        estado_controle.instante_inicio_similaridade_alta = -999.0
        return False
    if bool(dados_visao.get("intersecao_detectada")):
        estado_controle.instante_inicio_similaridade_alta = -999.0
        return False
    if float(dados_visao.get("confianca_curva_90", 0.0)) >= float(parametros.limiar_confianca_curva_90_execucao):
        estado_controle.instante_inicio_similaridade_alta = -999.0
        return False
    if dados_visao["confianca_linha"] < float(parametros.limiar_confianca_cautela):
        estado_controle.instante_inicio_similaridade_alta = -999.0
        return False

    if abs(float(dados_visao["erro_linha"])) > float(parametros.limiar_erro_similaridade_stuck):
        estado_controle.instante_inicio_similaridade_alta = -999.0
        return False

    similaridade_linha = float(dados_visao.get("similaridade_linha", 0.0))
    if similaridade_linha < float(parametros.limiar_similaridade_stuck):
        estado_controle.instante_inicio_similaridade_alta = -999.0
        return False

    if (agora - estado_controle.instante_ultimo_destravamento) < float(parametros.cooldown_similaridade_stuck):
        return False

    if estado_controle.instante_inicio_similaridade_alta < 0.0:
        estado_controle.instante_inicio_similaridade_alta = agora
        return False

    return (agora - estado_controle.instante_inicio_similaridade_alta) >= float(
        parametros.tempo_similaridade_stuck
    )


def _criar_acao_destravamento(estado_controle, dados_visao, parametros):
    erro_referencia = float(dados_visao.get("erro_lookahead", dados_visao["erro_linha"]))
    direcao_destravar = _direcao_correcao_por_erro(
        erro_referencia,
        parametros,
        fallback=estado_controle.ultima_direcao_confiavel,
    )
    if direcao_destravar not in {"esquerda", "direita"}:
        direcao_destravar = "direita"

    acao = _criar_acao_busca_linha(
        direcao_destravar,
        parametros.velocidade_destravamento_frente,
        parametros.velocidade_destravamento_re,
        parametros,
    )
    acao["modo"] = "destravamento"
    return acao, direcao_destravar


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
    acao["direcao"] = direcao
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


def _zerar_indicios_curva_90(estado_controle):
    estado_controle.indicios_curva_90_esquerda = 0
    estado_controle.indicios_curva_90_direita = 0


def _registrar_indicio_curva_90(estado_controle, direcao):
    if direcao == "esquerda":
        estado_controle.indicios_curva_90_esquerda = min(9, estado_controle.indicios_curva_90_esquerda + 1)
        estado_controle.indicios_curva_90_direita = max(0, estado_controle.indicios_curva_90_direita - 1)
    elif direcao == "direita":
        estado_controle.indicios_curva_90_direita = min(9, estado_controle.indicios_curva_90_direita + 1)
        estado_controle.indicios_curva_90_esquerda = max(0, estado_controle.indicios_curva_90_esquerda - 1)
    else:
        estado_controle.indicios_curva_90_esquerda = max(0, estado_controle.indicios_curva_90_esquerda - 1)
        estado_controle.indicios_curva_90_direita = max(0, estado_controle.indicios_curva_90_direita - 1)


def _criar_acao_retomada_pos_90(direcao, parametros):
    direcao = direcao if direcao in {"esquerda", "direita"} else "direita"
    acao = _criar_acao_busca_linha(
        direcao,
        parametros.velocidade_retomada_pos_90_frente,
        parametros.velocidade_retomada_pos_90_re,
        parametros,
    )
    acao["modo"] = "retomada_pos_90"
    acao["direcao"] = direcao
    return acao


def _atualizar_indicios_intersecao(estado_controle, dados_visao, parametros):
    intersecao_detectada = bool(dados_visao.get("intersecao_detectada"))
    confianca_intersecao = float(dados_visao.get("confianca_intersecao", 0.0))
    limiar_intersecao = float(parametros.limiar_confianca_intersecao_confirmacao)
    intersecao_candidata = bool(intersecao_detectada and confianca_intersecao >= limiar_intersecao)

    if intersecao_candidata:
        estado_controle.indicios_intersecao = min(12, estado_controle.indicios_intersecao + 1)
    else:
        estado_controle.indicios_intersecao = max(0, estado_controle.indicios_intersecao - 1)

    limiar_verde = float(parametros.limiar_confianca_verde_confirmacao)
    direcao_verde = _obter_direcao_verde_relativa_linha(dados_visao, parametros, limiar_verde)
    verde_esquerda = bool(direcao_verde == "esquerda")
    verde_direita = bool(direcao_verde == "direita")
    verde_duplo = bool(direcao_verde == "retorno")

    if intersecao_candidata and (verde_esquerda or verde_duplo):
        estado_controle.indicios_verde_esquerda = min(12, estado_controle.indicios_verde_esquerda + 1)
    else:
        estado_controle.indicios_verde_esquerda = max(0, estado_controle.indicios_verde_esquerda - 2)

    if intersecao_candidata and (verde_direita or verde_duplo):
        estado_controle.indicios_verde_direita = min(12, estado_controle.indicios_verde_direita + 1)
    else:
        estado_controle.indicios_verde_direita = max(0, estado_controle.indicios_verde_direita - 2)


def _limpar_indicios_intersecao(estado_controle):
    estado_controle.indicios_intersecao = 0
    estado_controle.indicios_verde_esquerda = 0
    estado_controle.indicios_verde_direita = 0


def _decidir_direcao_intersecao(estado_controle, dados_visao, parametros):
    direcao_verde_atual = _obter_direcao_verde_relativa_linha(dados_visao, parametros)
    if direcao_verde_atual in {"esquerda", "direita", "retorno"}:
        return direcao_verde_atual

    limiar_verde = max(1, int(parametros.quadros_confirmacao_verde_intersecao))
    esquerda = estado_controle.indicios_verde_esquerda >= limiar_verde
    direita = estado_controle.indicios_verde_direita >= limiar_verde

    if esquerda and direita:
        return "retorno"
    if esquerda:
        return "esquerda"
    if direita:
        return "direita"
    return "reto"


def _criar_acao_reto_temporizada(velocidade, modo):
    velocidade_pwm = _limitar_pwm(velocidade)
    acao = _criar_acao_diferencial(velocidade_pwm, velocidade_pwm)
    acao["modo"] = modo
    return acao


def _ajustar_direcao_verde(direcao, parametros):
    if direcao not in {"esquerda", "direita"}:
        return direcao
    if not getattr(parametros, "inverter_lado_verde", False):
        return direcao
    return "direita" if direcao == "esquerda" else "esquerda"


def _obter_direcao_verde_relativa_linha(dados_visao, parametros, limiar_confianca=None):
    if limiar_confianca is None:
        limiar_confianca = float(parametros.limiar_confianca_verde_confirmacao)

    if not bool(dados_visao.get("verde_detectado")):
        return None
    if float(dados_visao.get("confianca_verde", 0.0)) < float(limiar_confianca):
        return None
    if bool(dados_visao.get("verde_duplo_detectado")):
        return "retorno"

    deslocamento = float(dados_visao.get("deslocamento_verde_relativo_linha", 0.0))
    limiar_deslocamento = float(getattr(parametros, "limiar_deslocamento_verde_linha", 0.0))
    if deslocamento >= limiar_deslocamento:
        return _ajustar_direcao_verde("direita", parametros)
    if deslocamento <= -limiar_deslocamento:
        return _ajustar_direcao_verde("esquerda", parametros)

    verde_esquerda = bool(dados_visao.get("verde_esquerda_detectado"))
    verde_direita = bool(dados_visao.get("verde_direita_detectado"))
    if verde_esquerda ^ verde_direita:
        return _ajustar_direcao_verde(
            "esquerda" if verde_esquerda else "direita",
            parametros,
        )
    return None


def _criar_acao_giro_verde_curto(direcao, parametros):
    acao = _criar_acao_assistencia_curva(
        direcao,
        parametros.velocidade_giro_verde_frente,
        parametros.velocidade_giro_verde_re,
        parametros,
    )
    acao["modo"] = "intersecao_verde_giro"
    acao["direcao"] = direcao
    return acao


def _criar_acao_avanco_antes_giro_verde(direcao, parametros):
    acao = _criar_acao_reto_temporizada(
        parametros.velocidade_avanco_antes_verde,
        "intersecao_verde_pre_giro",
    )
    acao["direcao"] = direcao
    return acao


def _criar_acao_retorno_intersecao(parametros):
    velocidade = _limitar_pwm(parametros.velocidade_retorno_intersecao)
    acao = _criar_acao_diferencial(velocidade, -velocidade)
    acao["modo"] = "intersecao_retorno"
    acao["direcao"] = "retorno"
    return acao


def _iniciar_fluxo_intersecao(estado_controle, dados_visao, parametros, agora):
    estado_controle.estado_atual = ESTADO_INTERSECAO
    estado_controle.tempo_entrada_estado = agora

    direcao_intersecao = _decidir_direcao_intersecao(estado_controle, dados_visao, parametros)
    if direcao_intersecao in {"esquerda", "direita"}:
        direcao_intersecao = _ajustar_direcao_para_chassi(direcao_intersecao, parametros)
    if direcao_intersecao in {"esquerda", "direita"}:
        estado_controle.manobra_ativa = _criar_acao_avanco_antes_giro_verde(
            direcao_intersecao,
            parametros,
        )
        estado_controle.manobra_ativa_ate = agora + float(parametros.tempo_avanco_antes_verde)
        motivo = f"verde detectado ({direcao_intersecao}) - avancando antes de virar"
    else:
        estado_controle.manobra_ativa = _criar_acao_reto_temporizada(
            parametros.velocidade_avanco_intersecao,
            "intersecao_avanco",
        )
        estado_controle.manobra_ativa_ate = agora + float(parametros.tempo_avanco_intersecao)
        motivo = f"intersecao confirmada ({direcao_intersecao}) - avancando para centralizar"

    estado_controle.manobra_ativa["direcao_intersecao"] = direcao_intersecao
    estado_controle.manobra_ativa["verdeE"] = int(estado_controle.indicios_verde_esquerda)
    estado_controle.manobra_ativa["verdeD"] = int(estado_controle.indicios_verde_direita)
    _limpar_indicios_intersecao(estado_controle)
    return (
        estado_controle.manobra_ativa,
        motivo,
    )


def _deve_executar_intersecao(estado_controle, dados_visao, parametros, agora):
    if (agora - estado_controle.instante_ultima_intersecao) < float(parametros.cooldown_intersecao):
        return False
    if not dados_visao.get("linha_encontrada"):
        return False
    if float(dados_visao.get("confianca_linha", 0.0)) < float(parametros.limiar_confianca_intersecao_execucao):
        return False
    if estado_controle.indicios_intersecao < max(1, int(parametros.quadros_confirmacao_intersecao)):
        return False

    # Se a leitura parece apenas curva de 90 sem confirmacao de intersecao, deixa
    # o fluxo de curva tomar conta.
    if (
        float(dados_visao.get("confianca_curva_90", 0.0)) >= float(parametros.limiar_confianca_curva_90_execucao)
        and not bool(dados_visao.get("intersecao_detectada"))
    ):
        return False

    return True


def _deve_executar_curva_90(estado_controle, dados_visao, parametros, agora):
    if (agora - estado_controle.instante_ultimo_giro_90) < parametros.cooldown_giro_90:
        _registrar_indicio_curva_90(estado_controle, None)
        return None

    literal_esquerda = bool(dados_visao.get("curva_90_literal_esquerda"))
    literal_direita = bool(dados_visao.get("curva_90_literal_direita"))
    literal_ativo = bool(literal_esquerda ^ literal_direita)

    memoria_esquerda = bool(dados_visao.get("curva_90_memoria_esquerda"))
    memoria_direita = bool(dados_visao.get("curva_90_memoria_direita"))
    memoria_ativa = memoria_esquerda or memoria_direita
    intersecao_detectada = bool(dados_visao.get("intersecao_detectada"))
    confianca_intersecao = float(dados_visao.get("confianca_intersecao", 0.0))
    verde_detectado = bool(dados_visao.get("verde_detectado"))
    confianca_verde = float(dados_visao.get("confianca_verde", 0.0))

    pista_de_risco = bool(
        dados_visao.get("linha_toca_borda_esquerda")
        or dados_visao.get("linha_toca_borda_direita")
        or dados_visao.get("tempo_sem_linha", 0.0) > 0.0
        or dados_visao["confianca_linha"] < (parametros.limiar_confianca_curva_90_execucao + 0.05)
    )

    bloquear_por_intersecao = bool(
        intersecao_detectada
        and confianca_intersecao >= float(parametros.limiar_confianca_intersecao_bloqueio_90)
        and not literal_ativo
    )
    bloquear_por_verde = bool(
        verde_detectado
        and confianca_verde >= float(parametros.limiar_confianca_verde_bloqueio_90)
        and intersecao_detectada
        and not literal_ativo
    )
    if bloquear_por_intersecao or bloquear_por_verde:
        _registrar_indicio_curva_90(estado_controle, None)
        return None

    direcao_candidata = None
    if literal_ativo:
        direcao_candidata = "esquerda" if literal_esquerda else "direita"
    else:
        confianca_curva_90 = max(
            float(dados_visao.get("confianca_curva_90", 0.0)),
            float(dados_visao.get("confianca_curva_90_memoria", 0.0)) if pista_de_risco else 0.0,
        )
        confianca_linha = float(dados_visao.get("confianca_linha", 0.0))

        if confianca_linha < float(parametros.limiar_confianca_curva_90_execucao):
            if memoria_ativa and pista_de_risco:
                if memoria_esquerda:
                    direcao_candidata = "esquerda"
                elif memoria_direita:
                    direcao_candidata = "direita"
        elif confianca_curva_90 >= float(parametros.limiar_confianca_curva_90_execucao):
            if dados_visao.get("curva_90_esquerda"):
                direcao_candidata = "esquerda"
            elif dados_visao.get("curva_90_direita"):
                direcao_candidata = "direita"
            elif memoria_esquerda and pista_de_risco:
                direcao_candidata = "esquerda"
            elif memoria_direita and pista_de_risco:
                direcao_candidata = "direita"

    if direcao_candidata is None:
        _registrar_indicio_curva_90(estado_controle, None)
        return None

    direcao_candidata = _ajustar_direcao_para_chassi(direcao_candidata, parametros)
    _registrar_indicio_curva_90(estado_controle, direcao_candidata)

    quadros_necessarios = 1 if literal_ativo else max(1, int(parametros.quadros_confirmacao_curva_90))
    if direcao_candidata == "esquerda":
        if estado_controle.indicios_curva_90_esquerda < quadros_necessarios:
            return None
    elif direcao_candidata == "direita":
        if estado_controle.indicios_curva_90_direita < quadros_necessarios:
            return None
    else:
        return None

    _zerar_indicios_curva_90(estado_controle)
    return direcao_candidata


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
    correcao_pid = pid.suavizar_correcao(correcao_pid)
    correcao_pid = _limitar(correcao_pid, -parametros.correcao_maxima, parametros.correcao_maxima)

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
        _zerar_indicios_curva_90(estado_controle)

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
        elif modo_manobra == "retomada_pos_90":
            if _linha_confiavel_para_retomada(dados_visao, parametros):
                estado_controle.manobra_ativa = None
                estado_controle.manobra_ativa_ate = 0.0
                pid.reiniciar(suave=True)
            else:
                return estado_controle.manobra_ativa, "retomando linha apos curva de 90", correcao_pid
        elif modo_manobra == "retomada_pos_intersecao":
            if _linha_confiavel_para_retomada(dados_visao, parametros):
                estado_controle.manobra_ativa = None
                estado_controle.manobra_ativa_ate = 0.0
                pid.reiniciar(suave=True)
                _limpar_indicios_intersecao(estado_controle)
            else:
                return estado_controle.manobra_ativa, "retomando linha apos intersecao", correcao_pid
        elif modo_manobra in {
            "intersecao_avanco",
            "intersecao_verde_pre_giro",
            "intersecao_verde_giro",
            "intersecao_verde_avanco",
            "intersecao_giro",
            "intersecao_reto",
            "intersecao_retorno",
        }:
            return estado_controle.manobra_ativa, "executando manobra de intersecao", correcao_pid
        else:
            return estado_controle.manobra_ativa, "manobra temporizada", correcao_pid

    if estado_controle.manobra_ativa is not None and agora >= estado_controle.manobra_ativa_ate:
        manobra_encerrada = estado_controle.manobra_ativa
        estado_controle.manobra_ativa = None
        estado_controle.manobra_ativa_ate = 0.0

        if manobra_encerrada is not None and manobra_encerrada.get("modo") == "intersecao_avanco":
            direcao_intersecao = manobra_encerrada.get("direcao_intersecao", "reto")
            if direcao_intersecao == "retorno":
                estado_controle.manobra_ativa = _criar_acao_retorno_intersecao(parametros)
                estado_controle.manobra_ativa_ate = agora + float(parametros.tempo_retorno_intersecao)
            elif direcao_intersecao in {"esquerda", "direita"}:
                estado_controle.manobra_ativa = _criar_acao_giro_verde_curto(
                    direcao_intersecao,
                    parametros,
                )
                estado_controle.manobra_ativa_ate = agora + float(parametros.tempo_giro_verde)
            else:
                estado_controle.manobra_ativa = _criar_acao_reto_temporizada(
                    parametros.velocidade_reto_intersecao,
                    "intersecao_reto",
                )
                estado_controle.manobra_ativa["direcao"] = "reto"
                estado_controle.manobra_ativa_ate = agora + float(parametros.tempo_reto_intersecao)

            pid.reiniciar(suave=False)
            return estado_controle.manobra_ativa, f"intersecao: decisao {direcao_intersecao}", correcao_pid

        if manobra_encerrada is not None and manobra_encerrada.get("modo") == "intersecao_verde_pre_giro":
            direcao_intersecao = manobra_encerrada.get("direcao", "reto")
            estado_controle.manobra_ativa = _criar_acao_giro_verde_curto(
                direcao_intersecao,
                parametros,
            )
            estado_controle.manobra_ativa_ate = agora + float(parametros.tempo_giro_verde)
            pid.reiniciar(suave=False)
            return (
                estado_controle.manobra_ativa,
                f"intersecao: iniciando giro pelo verde em {direcao_intersecao}",
                correcao_pid,
            )

        if manobra_encerrada is not None and manobra_encerrada.get("modo") == "intersecao_verde_giro":
            direcao_intersecao = manobra_encerrada.get("direcao", "reto")
            estado_controle.manobra_ativa = _criar_acao_reto_temporizada(
                parametros.velocidade_avanco_verde,
                "intersecao_verde_avanco",
            )
            estado_controle.manobra_ativa["direcao"] = direcao_intersecao
            estado_controle.manobra_ativa_ate = agora + float(parametros.tempo_avanco_verde)
            pid.reiniciar(suave=True)
            return (
                estado_controle.manobra_ativa,
                f"intersecao: avancando para buscar a linha pelo verde em {direcao_intersecao}",
                correcao_pid,
            )

        if manobra_encerrada is not None and manobra_encerrada.get("modo") in {
            "intersecao_verde_pre_giro",
            "intersecao_verde_avanco",
            "intersecao_giro",
            "intersecao_reto",
            "intersecao_retorno",
        }:
            _limpar_indicios_intersecao(estado_controle)
            direcao_retomada = manobra_encerrada.get("direcao")
            if direcao_retomada not in {"esquerda", "direita"}:
                direcao_retomada = estado_controle.ultima_direcao_confiavel
            if direcao_retomada not in {"esquerda", "direita"}:
                direcao_retomada = "direita"

            if float(parametros.tempo_retomada_pos_intersecao) > 0.0:
                estado_controle.estado_atual = ESTADO_CONTENCAO
                estado_controle.tempo_entrada_estado = agora
                estado_controle.manobra_ativa = _criar_acao_busca_linha(
                    direcao_retomada,
                    parametros.velocidade_retomada_pos_intersecao_frente,
                    parametros.velocidade_retomada_pos_intersecao_re,
                    parametros,
                )
                estado_controle.manobra_ativa["modo"] = "retomada_pos_intersecao"
                estado_controle.manobra_ativa_ate = agora + float(parametros.tempo_retomada_pos_intersecao)
                pid.reiniciar(suave=True)
                return (
                    estado_controle.manobra_ativa,
                    f"retomando linha apos intersecao para {direcao_retomada}",
                    correcao_pid,
                )

        if (
            manobra_encerrada is not None
            and manobra_encerrada.get("modo") == "giro_90"
            and not _linha_confiavel_para_retomada(dados_visao, parametros)
            and float(parametros.tempo_retomada_pos_90) > 0.0
        ):
            direcao_retomada = manobra_encerrada.get("direcao")
            if direcao_retomada not in {"esquerda", "direita"}:
                direcao_retomada = estado_controle.ultima_direcao_confiavel
            if direcao_retomada not in {"esquerda", "direita"}:
                direcao_retomada = "direita"

            estado_controle.estado_atual = ESTADO_CONTENCAO
            estado_controle.tempo_entrada_estado = agora
            estado_controle.manobra_ativa = _criar_acao_retomada_pos_90(
                direcao_retomada,
                parametros,
            )
            estado_controle.manobra_ativa_ate = agora + float(parametros.tempo_retomada_pos_90)
            pid.reiniciar(suave=True)
            return (
                estado_controle.manobra_ativa,
                f"retomando linha apos curva de 90 para {direcao_retomada}",
                correcao_pid,
            )

    _atualizar_memoria_linha(estado_controle, dados_visao, parametros, agora)
    _atualizar_indicios_intersecao(estado_controle, dados_visao, parametros)

    linha_minima = _linha_minimamente_valida(dados_visao, parametros)
    linha_em_risco = _linha_em_risco(dados_visao, parametros)
    linha_confiavel = _linha_confiavel_para_retomada(dados_visao, parametros)
    retomada_confirmada = (
        estado_controle.quadros_confiaveis_consecutivos >= parametros.quadros_confirmacao_retomada
    )
    em_recuperacao = estado_controle.estado_atual in {ESTADO_SEM_LINHA, ESTADO_BUSCA, ESTADO_CONTENCAO}

    if not linha_minima:
        _zerar_indicios_curva_90(estado_controle)
        _limpar_indicios_intersecao(estado_controle)
        estado_controle.instante_inicio_similaridade_alta = -999.0
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

    if _deve_executar_intersecao(estado_controle, dados_visao, parametros, agora):
        estado_controle.instante_ultima_intersecao = agora
        _zerar_indicios_curva_90(estado_controle)
        acao_intersecao, motivo_intersecao = _iniciar_fluxo_intersecao(
            estado_controle,
            dados_visao,
            parametros,
            agora,
        )
        pid.reiniciar(suave=False)
        return acao_intersecao, motivo_intersecao, correcao_pid

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
        _zerar_indicios_curva_90(estado_controle)
        return estado_controle.manobra_ativa, f"curva de 90 graus {direcao_curva_90}", correcao_pid

    if _deve_destravar_por_similaridade(estado_controle, dados_visao, parametros, agora):
        estado_controle.estado_atual = ESTADO_CONTENCAO
        estado_controle.tempo_entrada_estado = agora
        estado_controle.instante_ultimo_destravamento = agora
        estado_controle.instante_inicio_similaridade_alta = -999.0
        estado_controle.manobra_ativa, direcao_destravamento = _criar_acao_destravamento(
            estado_controle,
            dados_visao,
            parametros,
        )
        estado_controle.manobra_ativa_ate = agora + float(parametros.tempo_destravamento)
        pid.reiniciar(suave=False)
        return (
            estado_controle.manobra_ativa,
            f"destravando por similaridade alta para {direcao_destravamento}",
            correcao_pid,
        )

    if linha_em_risco or (em_recuperacao and (not linha_confiavel or not retomada_confirmada)):
        _zerar_indicios_curva_90(estado_controle)
        if not bool(dados_visao.get("intersecao_detectada")):
            _limpar_indicios_intersecao(estado_controle)
        estado_controle.instante_inicio_similaridade_alta = -999.0
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
            f"sim={dados_visao.get('similaridade_linha', 0.0):.3f} "
            f"ret={estado_controle.quadros_confiaveis_consecutivos}"
        ),
        (
            "int="
            f"{'SIM' if dados_visao.get('intersecao_detectada') else 'NAO'} "
            f"confI={dados_visao.get('confianca_intersecao', 0.0):.2f}"
        ),
        (
            "verde="
            f"E={('SIM' if dados_visao.get('verde_esquerda_detectado') else 'NAO')} "
            f"D={('SIM' if dados_visao.get('verde_direita_detectado') else 'NAO')} "
            f"confV={dados_visao.get('confianca_verde', 0.0):.2f} "
            f"vl={dados_visao.get('direcao_verde_relativa_linha') or 'CENTRO'} "
            f"dv={dados_visao.get('deslocamento_verde_relativo_linha', 0.0):+.2f}"
        ),
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
                (
                    "verde="
                    f"E={('SIM' if dados_visao.get('verde_esquerda_detectado') else 'NAO')} "
                    f"D={('SIM' if dados_visao.get('verde_direita_detectado') else 'NAO')} "
                    f"vl={dados_visao.get('direcao_verde_relativa_linha') or 'CENTRO'} "
                    f"dv={dados_visao.get('deslocamento_verde_relativo_linha', 0.0):+.2f}"
                ),
                f"conf={dados_visao['confianca_linha']:.2f}",
                (
                    "borda="
                    f"{'E' if dados_visao.get('linha_toca_borda_esquerda') else ('D' if dados_visao.get('linha_toca_borda_direita') else 'NAO')}"
                ),
                f"int={dados_visao.get('confianca_intersecao', 0.0):.2f}",
                f"sem_linha={dados_visao.get('tempo_sem_linha', 0.0):.2f}s",
                f"sim={dados_visao.get('similaridade_linha', 0.0):.3f}",
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
    analisador.add_argument(
        "--suavizacao-erro",
        type=float,
        default=0.32,
        help="Peso do erro novo na suavizacao visual. Menor = mais suave; maior = mais reativo.",
    )
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
    analisador.add_argument(
        "--desativar-limiar-contextual",
        action="store_true",
        help="Desliga limiares diferentes para topo/base da ROI.",
    )
    analisador.add_argument("--fracao-topo-contextual", type=float, default=0.40)
    analisador.add_argument("--ajuste-limiar-topo-contextual", type=int, default=18)
    analisador.add_argument("--ajuste-limiar-base-contextual", type=int, default=8)
    analisador.add_argument("--ajuste-limiar-topo-escuro", type=int, default=22)
    analisador.add_argument("--limiar-densidade-topo-escuro", type=float, default=0.38)
    analisador.add_argument("--margem-melhoria-densidade-topo", type=float, default=0.08)
    analisador.add_argument("--faixa-contato-base-temporal", type=float, default=0.26)
    analisador.add_argument("--peso-distancia-x-temporal", type=float, default=1.0)
    analisador.add_argument("--peso-distancia-y-temporal", type=float, default=0.30)
    analisador.add_argument("--intervalo-similaridade-quadros", type=int, default=6)
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
    analisador.add_argument("--largura-minima-intersecao-relativa", type=float, default=0.42)
    analisador.add_argument("--densidade-lateral-minima-intersecao", type=float, default=0.14)
    analisador.add_argument("--densidade-centro-minima-intersecao", type=float, default=0.08)
    analisador.add_argument("--densidade-frontal-minima-intersecao", type=float, default=0.06)
    analisador.add_argument("--limiar-confianca-intersecao", type=float, default=0.55)
    analisador.add_argument("--quadros-confirmacao-intersecao", type=int, default=2)
    analisador.add_argument("--limiar-confianca-intersecao-confirmacao", type=float, default=0.58)
    analisador.add_argument("--limiar-confianca-intersecao-execucao", type=float, default=0.14)
    analisador.add_argument("--quadros-confirmacao-verde-intersecao", type=int, default=2)
    analisador.add_argument("--limiar-confianca-verde-confirmacao", type=float, default=0.14)
    analisador.add_argument("--cooldown-intersecao", type=float, default=0.80)
    analisador.add_argument("--velocidade-avanco-intersecao", type=int, default=92)
    analisador.add_argument("--tempo-avanco-intersecao", type=float, default=0.14)
    analisador.add_argument("--velocidade-giro-intersecao-frente", type=int, default=140)
    analisador.add_argument("--velocidade-giro-intersecao-re", type=int, default=132)
    analisador.add_argument("--tempo-giro-intersecao", type=float, default=0.54)
    analisador.add_argument("--velocidade-reto-intersecao", type=int, default=96)
    analisador.add_argument("--tempo-reto-intersecao", type=float, default=0.16)
    analisador.add_argument("--velocidade-retorno-intersecao", type=int, default=148)
    analisador.add_argument("--tempo-retorno-intersecao", type=float, default=1.05)
    analisador.add_argument("--tempo-retomada-pos-intersecao", type=float, default=0.22)
    analisador.add_argument("--velocidade-retomada-pos-intersecao-frente", type=int, default=112)
    analisador.add_argument("--velocidade-retomada-pos-intersecao-re", type=int, default=96)
    analisador.add_argument("--velocidade-avanco-antes-verde", type=int, default=92)
    analisador.add_argument("--tempo-avanco-antes-verde", type=float, default=0.28)
    analisador.add_argument("--velocidade-giro-verde-frente", type=int, default=112)
    analisador.add_argument("--velocidade-giro-verde-re", type=int, default=102)
    analisador.add_argument("--tempo-giro-verde", type=float, default=0.20)
    analisador.add_argument("--velocidade-avanco-verde", type=int, default=92)
    analisador.add_argument("--tempo-avanco-verde", type=float, default=0.18)
    analisador.add_argument(
        "--limiar-deslocamento-verde-linha",
        type=float,
        default=0.04,
        help="Deslocamento minimo normalizado entre centro do verde e centro da linha para decidir esquerda/direita.",
    )
    analisador.add_argument("--roi-verde", type=float, default=0.75)
    analisador.add_argument("--verde-h-min", type=int, default=35)
    analisador.add_argument("--verde-h-max", type=int, default=95)
    analisador.add_argument("--verde-s-min", type=int, default=60)
    analisador.add_argument("--verde-v-min", type=int, default=45)
    analisador.add_argument("--area-minima-verde", type=int, default=180)
    analisador.add_argument("--area-minima-verde-lateral", type=int, default=110)
    analisador.add_argument("--margem-central-verde", type=float, default=0.10)
    analisador.add_argument("--limiar-confianca-verde-lateral", type=float, default=0.10)
    analisador.set_defaults(inverter_lado_verde=False)
    analisador.add_argument(
        "--inverter-lado-verde",
        dest="inverter_lado_verde",
        action="store_true",
        help="Inverte apenas a interpretacao esquerda/direita do verde, sem afetar a correcao da linha.",
    )
    analisador.add_argument(
        "--nao-inverter-lado-verde",
        dest="inverter_lado_verde",
        action="store_false",
        help="Mantem o sentido padrao de esquerda/direita para o verde.",
    )
    analisador.add_argument(
        "--detectar-verde",
        action="store_true",
        default=False,
        help="Ativa a deteccao de verde no modo controle (desligado por padrao para reduzir custo de CPU).",
    )
    analisador.add_argument("--kp", type=float, default=145.0)
    analisador.add_argument("--ki", type=float, default=8.0)
    analisador.add_argument("--kd", type=float, default=30.0)
    analisador.add_argument("--integral-max", type=float, default=0.85)
    analisador.add_argument("--dt-minimo", type=float, default=0.01)
    analisador.add_argument("--alpha-derivada", type=float, default=0.45)
    analisador.add_argument(
        "--alpha-correcao-saida",
        type=float,
        default=0.40,
        help="Suavizacao adicional na saida final da correcao. Maior = transicao mais amortecida.",
    )
    analisador.add_argument(
        "--delta-correcao-maxima",
        type=float,
        default=24.0,
        help="Limita quanto a correcao pode variar entre quadros, reduzindo trancos.",
    )
    analisador.add_argument("--correcao-maxima", type=float, default=185.0)
    analisador.add_argument("--ganho-lookahead-suave", type=float, default=0.32)
    analisador.add_argument("--ganho-lookahead-forte", type=float, default=1.02)
    analisador.add_argument("--lookahead-erro-minimo", type=float, default=0.10)
    analisador.add_argument("--lookahead-erro-maximo", type=float, default=0.48)
    analisador.add_argument("--fator-correcao-forte", type=float, default=1.18)
    analisador.add_argument("--fator-antecipacao-velocidade", type=float, default=1.35)
    analisador.add_argument("--limiar-confianca-lookahead-velocidade", type=float, default=0.18)
    analisador.add_argument("--bonus-tracao-externa", type=float, default=4.0)
    analisador.add_argument("--bonus-freio-interno", type=float, default=16.0)

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
        "--limiar-similaridade-stuck",
        type=float,
        default=0.985,
        help="Similaridade minima da mascara entre quadros para suspeitar travamento.",
    )
    analisador.add_argument(
        "--tempo-similaridade-stuck",
        type=float,
        default=0.75,
        help="Tempo com similaridade alta antes de acionar destravamento.",
    )
    analisador.add_argument(
        "--cooldown-similaridade-stuck",
        type=float,
        default=2.40,
        help="Tempo minimo entre duas manobras de destravamento.",
    )
    analisador.add_argument(
        "--limiar-erro-similaridade-stuck",
        type=float,
        default=0.20,
        help="Nao considera travamento por similaridade quando o erro lateral esta alto.",
    )
    analisador.add_argument("--tempo-destravamento", type=float, default=0.28)
    analisador.add_argument("--velocidade-destravamento-frente", type=int, default=118)
    analisador.add_argument("--velocidade-destravamento-re", type=int, default=108)
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
    analisador.add_argument("--quadros-confirmacao-curva-90", type=int, default=2)
    analisador.add_argument("--limiar-confianca-intersecao-bloqueio-90", type=float, default=0.62)
    analisador.add_argument("--limiar-confianca-verde-bloqueio-90", type=float, default=0.35)
    analisador.add_argument("--limiar-confianca-curva-90-execucao", type=float, default=0.22)
    analisador.add_argument("--limiar-confianca-retomada-giro-90", type=float, default=0.26)
    analisador.add_argument("--limiar-erro-retomada-giro-90", type=float, default=0.22)
    analisador.add_argument("--limiar-erro-lookahead-retomada-giro-90", type=float, default=0.34)
    analisador.add_argument("--tempo-retomada-pos-90", type=float, default=0.20)
    analisador.add_argument("--velocidade-retomada-pos-90-frente", type=int, default=104)
    analisador.add_argument("--velocidade-retomada-pos-90-re", type=int, default=92)
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
        usar_limiar_contextual=not parametros.desativar_limiar_contextual,
        fracao_topo_contextual=parametros.fracao_topo_contextual,
        ajuste_limiar_topo_contextual=parametros.ajuste_limiar_topo_contextual,
        ajuste_limiar_base_contextual=parametros.ajuste_limiar_base_contextual,
        ajuste_limiar_topo_escuro=parametros.ajuste_limiar_topo_escuro,
        limiar_densidade_topo_escuro=parametros.limiar_densidade_topo_escuro,
        margem_melhoria_densidade_topo=parametros.margem_melhoria_densidade_topo,
        faixa_contato_base_temporal=parametros.faixa_contato_base_temporal,
        peso_distancia_x_temporal=parametros.peso_distancia_x_temporal,
        peso_distancia_y_temporal=parametros.peso_distancia_y_temporal,
        intervalo_similaridade_quadros=parametros.intervalo_similaridade_quadros,
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
        largura_minima_intersecao_relativa=parametros.largura_minima_intersecao_relativa,
        densidade_lateral_minima_intersecao=parametros.densidade_lateral_minima_intersecao,
        densidade_centro_minima_intersecao=parametros.densidade_centro_minima_intersecao,
        densidade_frontal_minima_intersecao=parametros.densidade_frontal_minima_intersecao,
        limiar_confianca_intersecao=parametros.limiar_confianca_intersecao,
        roi_verde=parametros.roi_verde,
        verde_h_min=parametros.verde_h_min,
        verde_h_max=parametros.verde_h_max,
        verde_s_min=parametros.verde_s_min,
        verde_v_min=parametros.verde_v_min,
        area_minima_verde=parametros.area_minima_verde,
        area_minima_verde_lateral=parametros.area_minima_verde_lateral,
        margem_central_verde=parametros.margem_central_verde,
        limiar_confianca_verde_lateral=parametros.limiar_confianca_verde_lateral,
        detectar_verde=parametros.detectar_verde,
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
    parametros.velocidade_destravamento_frente = _limitar_pwm(parametros.velocidade_destravamento_frente)
    parametros.velocidade_destravamento_re = _limitar_pwm(parametros.velocidade_destravamento_re)
    parametros.velocidade_retomada_pos_90_frente = _limitar_pwm(parametros.velocidade_retomada_pos_90_frente)
    parametros.velocidade_retomada_pos_90_re = _limitar_pwm(parametros.velocidade_retomada_pos_90_re)
    parametros.velocidade_avanco_intersecao = _limitar_pwm(parametros.velocidade_avanco_intersecao)
    parametros.velocidade_giro_intersecao_frente = _limitar_pwm(parametros.velocidade_giro_intersecao_frente)
    parametros.velocidade_giro_intersecao_re = _limitar_pwm(parametros.velocidade_giro_intersecao_re)
    parametros.velocidade_reto_intersecao = _limitar_pwm(parametros.velocidade_reto_intersecao)
    parametros.velocidade_retorno_intersecao = _limitar_pwm(parametros.velocidade_retorno_intersecao)
    parametros.velocidade_retomada_pos_intersecao_frente = _limitar_pwm(
        parametros.velocidade_retomada_pos_intersecao_frente
    )
    parametros.velocidade_retomada_pos_intersecao_re = _limitar_pwm(parametros.velocidade_retomada_pos_intersecao_re)
    parametros.velocidade_avanco_antes_verde = _limitar_pwm(parametros.velocidade_avanco_antes_verde)
    parametros.velocidade_giro_verde_frente = _limitar_pwm(parametros.velocidade_giro_verde_frente)
    parametros.velocidade_giro_verde_re = _limitar_pwm(parametros.velocidade_giro_verde_re)
    parametros.velocidade_avanco_verde = _limitar_pwm(parametros.velocidade_avanco_verde)
    parametros.piso_re_esquerda = _limitar_pwm(parametros.piso_re_esquerda)
    parametros.bonus_re_esquerda = max(0.0, float(parametros.bonus_re_esquerda))
    parametros.suavizacao_erro = _limitar(float(parametros.suavizacao_erro), 0.05, 0.95)
    parametros.alpha_derivada = _limitar(float(parametros.alpha_derivada), 0.0, 0.98)
    parametros.alpha_correcao_saida = _limitar(float(parametros.alpha_correcao_saida), 0.0, 0.95)
    parametros.delta_correcao_maxima = max(0.0, float(parametros.delta_correcao_maxima))
    parametros.correcao_maxima = max(0.0, float(parametros.correcao_maxima))
    parametros.ganho_lookahead_suave = _limitar(float(parametros.ganho_lookahead_suave), 0.0, 2.0)
    parametros.ganho_lookahead_forte = _limitar(float(parametros.ganho_lookahead_forte), 0.0, 2.0)
    parametros.fator_correcao_forte = _limitar(float(parametros.fator_correcao_forte), 1.0, 2.5)
    parametros.bonus_tracao_externa = max(0.0, float(parametros.bonus_tracao_externa))
    parametros.bonus_freio_interno = max(0.0, float(parametros.bonus_freio_interno))
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
    parametros.fracao_topo_contextual = _limitar(float(parametros.fracao_topo_contextual), 0.10, 0.90)
    parametros.ajuste_limiar_topo_contextual = int(_limitar(parametros.ajuste_limiar_topo_contextual, 0, 80))
    parametros.ajuste_limiar_base_contextual = int(_limitar(parametros.ajuste_limiar_base_contextual, 0, 80))
    parametros.ajuste_limiar_topo_escuro = int(_limitar(parametros.ajuste_limiar_topo_escuro, 0, 80))
    parametros.limiar_densidade_topo_escuro = _limitar(float(parametros.limiar_densidade_topo_escuro), 0.0, 1.0)
    parametros.margem_melhoria_densidade_topo = _limitar(
        float(parametros.margem_melhoria_densidade_topo),
        0.0,
        1.0,
    )
    parametros.faixa_contato_base_temporal = _limitar(float(parametros.faixa_contato_base_temporal), 0.05, 0.60)
    parametros.peso_distancia_x_temporal = max(0.0, float(parametros.peso_distancia_x_temporal))
    parametros.peso_distancia_y_temporal = max(0.0, float(parametros.peso_distancia_y_temporal))
    parametros.intervalo_similaridade_quadros = max(1, int(parametros.intervalo_similaridade_quadros))
    parametros.limiar_similaridade_stuck = _limitar(float(parametros.limiar_similaridade_stuck), 0.0, 1.0)
    parametros.tempo_similaridade_stuck = max(0.0, float(parametros.tempo_similaridade_stuck))
    parametros.cooldown_similaridade_stuck = max(0.0, float(parametros.cooldown_similaridade_stuck))
    parametros.limiar_erro_similaridade_stuck = _limitar(
        float(parametros.limiar_erro_similaridade_stuck),
        0.0,
        1.0,
    )
    parametros.tempo_destravamento = max(0.05, float(parametros.tempo_destravamento))
    parametros.tempo_retomada_pos_90 = max(0.0, float(parametros.tempo_retomada_pos_90))
    parametros.quadros_confirmacao_curva_90 = max(1, int(parametros.quadros_confirmacao_curva_90))
    parametros.limiar_confianca_intersecao_bloqueio_90 = _limitar(
        float(parametros.limiar_confianca_intersecao_bloqueio_90),
        0.0,
        1.0,
    )
    parametros.limiar_confianca_verde_bloqueio_90 = _limitar(
        float(parametros.limiar_confianca_verde_bloqueio_90),
        0.0,
        1.0,
    )
    parametros.largura_minima_intersecao_relativa = _limitar(
        float(parametros.largura_minima_intersecao_relativa),
        0.10,
        0.95,
    )
    parametros.densidade_lateral_minima_intersecao = _limitar(
        float(parametros.densidade_lateral_minima_intersecao),
        0.0,
        1.0,
    )
    parametros.densidade_centro_minima_intersecao = _limitar(
        float(parametros.densidade_centro_minima_intersecao),
        0.0,
        1.0,
    )
    parametros.densidade_frontal_minima_intersecao = _limitar(
        float(parametros.densidade_frontal_minima_intersecao),
        0.0,
        1.0,
    )
    parametros.limiar_confianca_intersecao = _limitar(
        float(parametros.limiar_confianca_intersecao),
        0.0,
        1.0,
    )
    parametros.area_minima_verde_lateral = max(1, int(parametros.area_minima_verde_lateral))
    parametros.margem_central_verde = _limitar(float(parametros.margem_central_verde), 0.02, 0.35)
    parametros.limiar_confianca_verde_lateral = _limitar(
        float(parametros.limiar_confianca_verde_lateral),
        0.0,
        1.0,
    )
    parametros.quadros_confirmacao_intersecao = max(1, int(parametros.quadros_confirmacao_intersecao))
    parametros.quadros_confirmacao_verde_intersecao = max(1, int(parametros.quadros_confirmacao_verde_intersecao))
    parametros.limiar_confianca_intersecao_confirmacao = _limitar(
        float(parametros.limiar_confianca_intersecao_confirmacao),
        0.0,
        1.0,
    )
    parametros.limiar_confianca_intersecao_execucao = _limitar(
        float(parametros.limiar_confianca_intersecao_execucao),
        0.0,
        1.0,
    )
    parametros.limiar_confianca_verde_confirmacao = _limitar(
        float(parametros.limiar_confianca_verde_confirmacao),
        0.0,
        1.0,
    )
    parametros.cooldown_intersecao = max(0.0, float(parametros.cooldown_intersecao))
    parametros.tempo_avanco_intersecao = max(0.0, float(parametros.tempo_avanco_intersecao))
    parametros.tempo_giro_intersecao = max(0.0, float(parametros.tempo_giro_intersecao))
    parametros.tempo_reto_intersecao = max(0.0, float(parametros.tempo_reto_intersecao))
    parametros.tempo_retorno_intersecao = max(0.0, float(parametros.tempo_retorno_intersecao))
    parametros.tempo_retomada_pos_intersecao = max(0.0, float(parametros.tempo_retomada_pos_intersecao))
    parametros.tempo_avanco_antes_verde = max(0.0, float(parametros.tempo_avanco_antes_verde))
    parametros.tempo_giro_verde = max(0.0, float(parametros.tempo_giro_verde))
    parametros.tempo_avanco_verde = max(0.0, float(parametros.tempo_avanco_verde))
    parametros.limiar_deslocamento_verde_linha = _limitar(
        float(parametros.limiar_deslocamento_verde_linha),
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
    parametros.velocidade_retomada_pos_90_frente = int(
        _limitar(
            parametros.velocidade_retomada_pos_90_frente,
            0,
            parametros.velocidade_maxima_contencao,
        )
    )
    parametros.velocidade_retomada_pos_90_re = int(
        _limitar(
            parametros.velocidade_retomada_pos_90_re,
            0,
            parametros.velocidade_maxima_contencao,
        )
    )
    parametros.velocidade_avanco_intersecao = int(
        _limitar(
            parametros.velocidade_avanco_intersecao,
            0,
            parametros.velocidade_maxima_contencao,
        )
    )
    parametros.velocidade_giro_intersecao_frente = int(
        _limitar(
            parametros.velocidade_giro_intersecao_frente,
            0,
            parametros.velocidade_maxima,
        )
    )
    parametros.velocidade_giro_intersecao_re = int(
        _limitar(
            parametros.velocidade_giro_intersecao_re,
            0,
            parametros.velocidade_maxima,
        )
    )
    parametros.velocidade_reto_intersecao = int(
        _limitar(
            parametros.velocidade_reto_intersecao,
            0,
            parametros.velocidade_maxima_contencao,
        )
    )
    parametros.velocidade_retorno_intersecao = int(
        _limitar(
            parametros.velocidade_retorno_intersecao,
            0,
            parametros.velocidade_maxima,
        )
    )
    parametros.velocidade_retomada_pos_intersecao_frente = int(
        _limitar(
            parametros.velocidade_retomada_pos_intersecao_frente,
            0,
            parametros.velocidade_maxima_contencao,
        )
    )
    parametros.velocidade_retomada_pos_intersecao_re = int(
        _limitar(
            parametros.velocidade_retomada_pos_intersecao_re,
            0,
            parametros.velocidade_maxima_contencao,
        )
    )
    parametros.velocidade_avanco_antes_verde = int(
        _limitar(
            parametros.velocidade_avanco_antes_verde,
            0,
            parametros.velocidade_maxima_contencao,
        )
    )
    parametros.velocidade_giro_verde_frente = int(
        _limitar(
            parametros.velocidade_giro_verde_frente,
            0,
            parametros.velocidade_maxima,
        )
    )
    parametros.velocidade_giro_verde_re = int(
        _limitar(
            parametros.velocidade_giro_verde_re,
            0,
            parametros.velocidade_maxima,
        )
    )
    parametros.velocidade_avanco_verde = int(
        _limitar(
            parametros.velocidade_avanco_verde,
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
        alpha_correcao_saida=parametros.alpha_correcao_saida,
        delta_correcao_maxima=parametros.delta_correcao_maxima,
    )

    estado_controle.tempo_entrada_estado = time.monotonic()

    tem_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    exibir_janela = parametros.show or (tem_display and not parametros.no_show)
    gerar_debug_visual = bool(exibir_janela or parametros.stream or parametros.debug_path)
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

            dados_visao = analisar_quadro(
                quadro_bgr,
                configuracao_visao,
                estado_visao,
                gerar_debug=gerar_debug_visual,
            )
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

            quadro_debug = None
            if gerar_debug_visual and dados_visao["quadro_debug"] is not None:
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

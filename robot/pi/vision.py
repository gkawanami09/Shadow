import time
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class ConfiguracaoVisao:
    roi: float = 0.48
    limiar_binario: int | None = None
    inverter_linha: bool = False
    usar_limiar_adaptativo: bool = True
    bloco_limiar_adaptativo: int = 41
    constante_limiar_adaptativo: int = 9
    fracao_minima_pixels_refinada: float = 0.60
    area_minima_contorno: int = 180
    area_minima_linha: int = 320
    suavizacao_erro: float = 0.40
    limiar_confianca: float = 0.10
    faixa_base_contorno: float = 0.16
    margem_lateral_descarte: float = 0.10
    lookahead_fracao: float = 0.42
    lookahead_minimo_pixels: int = 18
    limiar_confianca_curva_90: float = 0.18
    limiar_confianca_lookahead_curva_90: float = 0.14
    limiar_erro_lookahead_curva_90: float = 0.36
    limiar_delta_erro_curva_90: float = 0.10
    limiar_erro_base_curva_90: float = 0.30
    faixa_superior_curva_90: float = 0.48
    faixa_inferior_curva_90: float = 0.24
    densidade_lateral_curva_90: float = 0.16
    densidade_oposta_max_curva_90: float = 0.08
    densidade_base_centro_curva_90: float = 0.10
    largura_janela_branca_curva_90: float = 0.16
    largura_janela_lateral_curva_90: float = 0.28
    densidade_frontal_max_curva_90: float = 0.08
    limiar_deslocamento_topo_curva_90: float = 0.18
    densidade_minima_topo_curva_90: float = 0.08
    tempo_memoria_curva_90: float = 0.42
    limiar_confianca_memoria_curva_90: float = 0.12
    limiar_deslocamento_topo_memoria_curva_90: float = 0.12
    densidade_minima_topo_memoria_curva_90: float = 0.05
    roi_verde: float = 0.75
    verde_h_min: int = 35
    verde_h_max: int = 95
    verde_s_min: int = 60
    verde_v_min: int = 45
    area_minima_verde: int = 180


@dataclass
class EstadoVisao:
    tempo_ultima_linha: float = field(default_factory=time.monotonic)
    centro_x_anterior: float | None = None
    erro_suavizado_anterior: float = 0.0
    erro_lookahead_anterior: float = 0.0
    instante_ultima_curva_90_esquerda: float = -999.0
    instante_ultima_curva_90_direita: float = -999.0
    confianca_memoria_curva_90_esquerda: float = 0.0
    confianca_memoria_curva_90_direita: float = 0.0


def _limitar(valor, minimo, maximo):
    return max(minimo, min(maximo, valor))


def _obter_limites_roi(formato_quadro, fracao_roi):
    altura_total, largura_total = formato_quadro[:2]
    fracao_roi = float(_limitar(fracao_roi, 0.20, 0.90))
    altura_roi = max(1, int(altura_total * fracao_roi))
    y_inicio = altura_total - altura_roi
    return y_inicio, altura_total, largura_total


def _gerar_mascara_linha(quadro_roi_bgr, limiar_binario, inverter_linha):
    quadro_cinza = cv2.cvtColor(quadro_roi_bgr, cv2.COLOR_BGR2GRAY)
    quadro_suave = cv2.GaussianBlur(quadro_cinza, (5, 5), 0)

    modo_limiar = cv2.THRESH_BINARY if inverter_linha else cv2.THRESH_BINARY_INV
    if limiar_binario is None:
        _, mascara_linha = cv2.threshold(
            quadro_suave,
            0,
            255,
            modo_limiar | cv2.THRESH_OTSU,
        )
        limiar_usado = "otsu"
    else:
        _, mascara_linha = cv2.threshold(
            quadro_suave,
            int(limiar_binario),
            255,
            modo_limiar,
        )
        limiar_usado = str(int(limiar_binario))

    return mascara_linha, limiar_usado, quadro_suave


def _gerar_mascara_adaptativa(quadro_suave, configuracao):
    if not configuracao.usar_limiar_adaptativo:
        return None

    bloco = max(3, int(configuracao.bloco_limiar_adaptativo))
    if bloco % 2 == 0:
        bloco += 1

    modo_limiar = cv2.THRESH_BINARY if configuracao.inverter_linha else cv2.THRESH_BINARY_INV
    return cv2.adaptiveThreshold(
        quadro_suave,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        modo_limiar,
        bloco,
        int(configuracao.constante_limiar_adaptativo),
    )


def _pos_processar_mascara_linha(mascara_linha):
    nucleo_abertura = np.ones((3, 3), dtype=np.uint8)
    nucleo_fechamento = np.ones((5, 5), dtype=np.uint8)
    mascara_linha = cv2.morphologyEx(mascara_linha, cv2.MORPH_OPEN, nucleo_abertura, iterations=1)
    mascara_linha = cv2.morphologyEx(mascara_linha, cv2.MORPH_CLOSE, nucleo_fechamento, iterations=2)
    return mascara_linha


def _refinar_mascara_linha(mascara_linha, quadro_suave, configuracao):
    mascara_base = mascara_linha.copy()
    mascara_adaptativa = _gerar_mascara_adaptativa(quadro_suave, configuracao)
    if mascara_adaptativa is not None:
        mascara_linha = cv2.bitwise_and(mascara_linha, mascara_adaptativa)

        pixels_base = float(np.count_nonzero(mascara_base))
        pixels_refinados = float(np.count_nonzero(mascara_linha))
        if pixels_base > 0.0:
            proporcao_refinada = pixels_refinados / pixels_base
            if proporcao_refinada < float(configuracao.fracao_minima_pixels_refinada):
                mascara_linha = mascara_base

    return _pos_processar_mascara_linha(mascara_linha)


def _selecionar_linha(mascara_linha, estado, configuracao):
    altura_roi, largura_roi = mascara_linha.shape[:2]
    contornos, _ = cv2.findContours(mascara_linha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    melhor = None
    melhor_pontuacao = -1.0

    for contorno in contornos:
        area = float(cv2.contourArea(contorno))
        if area < configuracao.area_minima_contorno:
            continue

        x, y, largura_caixa, altura_caixa = cv2.boundingRect(contorno)
        momentos = cv2.moments(contorno)
        if momentos["m00"] > 0:
            centro_x = float(momentos["m10"] / momentos["m00"])
            centro_y = float(momentos["m01"] / momentos["m00"])
        else:
            centro_x = float(x + largura_caixa / 2.0)
            centro_y = float(y + altura_caixa / 2.0)

        area_relativa = area / max(1.0, altura_roi * largura_roi)
        proximidade_base = (y + altura_caixa) / max(1.0, altura_roi)
        largura_relativa = largura_caixa / max(1.0, largura_roi)

        if estado.centro_x_anterior is None:
            proximidade_anterior = 0.5
        else:
            distancia_anterior = abs(centro_x - estado.centro_x_anterior)
            proximidade_anterior = 1.0 - min(1.0, distancia_anterior / max(1.0, largura_roi / 2.0))

        mascara_contorno = np.zeros_like(mascara_linha)
        cv2.drawContours(mascara_contorno, [contorno], -1, 255, thickness=cv2.FILLED)

        faixa_base_pixels = max(1, int(altura_roi * float(configuracao.faixa_base_contorno)))
        y_base = max(0, altura_roi - faixa_base_pixels)
        pixels_base = float(np.count_nonzero(mascara_contorno[y_base:, :]))
        ancoragem_base = pixels_base / max(1.0, area)

        margem_lateral_pixels = max(1, int(largura_roi * float(configuracao.margem_lateral_descarte)))
        pixels_laterais = float(
            np.count_nonzero(mascara_contorno[:, :margem_lateral_pixels])
            + np.count_nonzero(mascara_contorno[:, largura_roi - margem_lateral_pixels :])
        )
        penalidade_lateral = min(1.0, pixels_laterais / max(1.0, area))
        penalidade_largura = max(0.0, (largura_relativa - 0.72) / 0.28)

        pontuacao = (
            area_relativa * 1.6
            + proximidade_base * 0.7
            + proximidade_anterior * 1.3
            + min(1.0, ancoragem_base * 1.8) * 1.2
            - penalidade_lateral * 1.1
            - penalidade_largura * 0.9
        )

        if pontuacao > melhor_pontuacao:
            melhor_pontuacao = pontuacao
            melhor = {
                "contorno": contorno,
                "area": area,
                "centro_x": centro_x,
                "centro_y": centro_y,
                "caixa": (x, y, largura_caixa, altura_caixa),
                "largura_caixa": largura_caixa,
                "altura_caixa": altura_caixa,
                "area_relativa": area_relativa,
                "proximidade_anterior": proximidade_anterior,
                "ancoragem_base": ancoragem_base,
                "penalidade_lateral": penalidade_lateral,
            }

    if melhor is None or melhor["area"] < configuracao.area_minima_linha:
        return {
            "linha_encontrada": False,
            "area": 0.0,
            "erro_bruto": 0.0,
            "erro_suavizado": estado.erro_suavizado_anterior,
            "confianca": 0.0,
            "centro_x": None,
            "centro_y": None,
            "caixa": None,
            "largura_caixa": 0,
            "altura_caixa": 0,
            "contorno": None,
            "proximidade_anterior": 0.0,
            "pontuacao": -1.0,
        }

    erro_bruto = (melhor["centro_x"] - (largura_roi / 2.0)) / max(1.0, largura_roi / 2.0)
    erro_bruto = float(_limitar(erro_bruto, -1.0, 1.0))

    erro_suavizado = (
        (1.0 - configuracao.suavizacao_erro) * estado.erro_suavizado_anterior
        + configuracao.suavizacao_erro * erro_bruto
    )
    erro_suavizado = float(_limitar(erro_suavizado, -1.0, 1.0))

    confianca_area = min(1.0, melhor["area_relativa"] / 0.20)
    confianca_continuidade = melhor["proximidade_anterior"]
    confianca_base = min(1.0, melhor["ancoragem_base"] * 1.6)
    confianca_lateral = 1.0 - min(1.0, melhor["penalidade_lateral"])
    confianca = float(
        _limitar(
            0.42 * confianca_area
            + 0.24 * confianca_continuidade
            + 0.24 * confianca_base
            + 0.10 * confianca_lateral,
            0.0,
            1.0,
        )
    )

    return {
        "linha_encontrada": True,
        "area": melhor["area"],
        "erro_bruto": erro_bruto,
        "erro_suavizado": erro_suavizado,
        "confianca": confianca,
        "centro_x": melhor["centro_x"],
        "centro_y": melhor["centro_y"],
        "caixa": melhor["caixa"],
        "largura_caixa": melhor["largura_caixa"],
        "altura_caixa": melhor["altura_caixa"],
        "contorno": melhor["contorno"],
        "proximidade_anterior": melhor["proximidade_anterior"],
        "pontuacao": melhor_pontuacao,
    }


def _escolher_resultado_linha(info_refinada, mascara_refinada, info_base, mascara_base, configuracao):
    if not info_base["linha_encontrada"]:
        return info_refinada, mascara_refinada, "refinada"

    if not info_refinada["linha_encontrada"]:
        return info_base, mascara_base, "base"

    limiar_fallback = max(float(configuracao.limiar_confianca), 0.18)
    refinada_fraca = info_refinada["confianca"] < limiar_fallback
    base_consistente = info_base["proximidade_anterior"] >= max(
        0.0,
        info_refinada["proximidade_anterior"] - 0.10,
    )
    base_melhor = info_base["confianca"] > (info_refinada["confianca"] + 0.05)
    base_mais_ancorada = info_base["pontuacao"] > (info_refinada["pontuacao"] + 0.12)
    centros_proximos = False
    if info_base["centro_x"] is not None and info_refinada["centro_x"] is not None:
        dist_centros = abs(float(info_base["centro_x"]) - float(info_refinada["centro_x"]))
        largura_ref = max(
            10.0,
            float(info_base.get("largura_caixa", 0)),
            float(info_refinada.get("largura_caixa", 0)),
        )
        centros_proximos = dist_centros <= max(12.0, largura_ref * 0.60)

    area_base = float(info_base.get("area", 0.0))
    area_refinada = float(info_refinada.get("area", 0.0))
    base_area_maior = area_base > (area_refinada * 1.35)
    base_largura_maior = float(info_base.get("largura_caixa", 0)) > (
        float(info_refinada.get("largura_caixa", 0)) * 1.25
    )
    base_equivalente = info_base["confianca"] >= (info_refinada["confianca"] - 0.08)
    refinada_estreita_demais = (
        float(info_refinada.get("largura_caixa", 0)) > 0.0
        and float(info_base.get("largura_caixa", 0)) >= 8.0
        and float(info_refinada.get("largura_caixa", 0))
        <= (float(info_base.get("largura_caixa", 0)) * 0.78)
    )

    if centros_proximos and base_equivalente and (base_area_maior or (base_largura_maior and refinada_estreita_demais)):
        return info_base, mascara_base, "base"

    if refinada_fraca and base_consistente and (base_melhor or base_mais_ancorada):
        return info_base, mascara_base, "base"

    return info_refinada, mascara_refinada, "refinada"


def _calcular_erro_lookahead(info_linha, mascara_linha, estado, configuracao):
    if not info_linha["linha_encontrada"] or info_linha["contorno"] is None:
        return estado.erro_lookahead_anterior, 0.0, None

    altura_roi, largura_roi = mascara_linha.shape[:2]
    y_referencia = int(_limitar(altura_roi * float(configuracao.lookahead_fracao), 0, altura_roi - 1))

    mascara_contorno = np.zeros_like(mascara_linha)
    cv2.drawContours(mascara_contorno, [info_linha["contorno"]], -1, 255, thickness=cv2.FILLED)

    faixa_superior = mascara_contorno[: max(1, y_referencia + 1), :]
    pontos_y, pontos_x = np.where(faixa_superior > 0)
    if pontos_x.size < max(1, int(configuracao.lookahead_minimo_pixels)):
        return estado.erro_lookahead_anterior, 0.0, None

    pesos = 1.0 + (1.0 - (pontos_y.astype(np.float32) / max(1.0, float(y_referencia + 1))))
    centro_x_lookahead = float(np.average(pontos_x.astype(np.float32), weights=pesos))
    erro_bruto_lookahead = (centro_x_lookahead - (largura_roi / 2.0)) / max(1.0, largura_roi / 2.0)
    erro_bruto_lookahead = float(_limitar(erro_bruto_lookahead, -1.0, 1.0))

    erro_lookahead = (
        (1.0 - configuracao.suavizacao_erro) * estado.erro_lookahead_anterior
        + configuracao.suavizacao_erro * erro_bruto_lookahead
    )
    erro_lookahead = float(_limitar(erro_lookahead, -1.0, 1.0))

    faixa_relativa = pontos_x.size / max(1.0, largura_roi * max(1.0, y_referencia + 1))
    confianca_lookahead = float(_limitar(0.55 + min(0.45, faixa_relativa * 6.0), 0.0, 1.0))
    return erro_lookahead, confianca_lookahead, (int(round(centro_x_lookahead)), y_referencia)


def _calcular_indicadores_laterais(info_linha, largura_roi, configuracao):
    resultado_vazio = {
        "linha_toca_borda_esquerda": False,
        "linha_toca_borda_direita": False,
        "centro_linha_normalizado": 0.0,
        "largura_linha_relativa": 0.0,
    }

    if not info_linha["linha_encontrada"] or info_linha["caixa"] is None:
        return resultado_vazio

    x, _, largura_caixa, _ = info_linha["caixa"]
    margem_lateral_pixels = max(
        2,
        int(round(largura_roi * float(configuracao.margem_lateral_descarte))),
    )
    limite_direito = max(0, largura_roi - margem_lateral_pixels)

    centro_x = float(info_linha.get("centro_x", largura_roi / 2.0))
    centro_normalizado = (centro_x - (largura_roi / 2.0)) / max(1.0, largura_roi / 2.0)

    return {
        "linha_toca_borda_esquerda": bool(int(x) <= margem_lateral_pixels),
        "linha_toca_borda_direita": bool(int(x + largura_caixa) >= limite_direito),
        "centro_linha_normalizado": float(_limitar(centro_normalizado, -1.0, 1.0)),
        "largura_linha_relativa": float(largura_caixa / max(1.0, largura_roi)),
    }


def _densidade_faixa(mascara, x_inicio, x_fim, y_inicio, y_fim):
    altura, largura = mascara.shape[:2]
    x_inicio = int(_limitar(x_inicio, 0, largura))
    x_fim = int(_limitar(x_fim, 0, largura))
    y_inicio = int(_limitar(y_inicio, 0, altura))
    y_fim = int(_limitar(y_fim, 0, altura))
    if x_fim <= x_inicio or y_fim <= y_inicio:
        return 0.0

    recorte = mascara[y_inicio:y_fim, x_inicio:x_fim]
    return float(np.count_nonzero(recorte)) / float(recorte.size)


def _calcular_centro_x_base_contorno(mascara_contorno, configuracao):
    altura_roi, largura_roi = mascara_contorno.shape[:2]
    faixa_base_pixels = max(1, int(altura_roi * float(configuracao.faixa_base_contorno)))
    y_base = max(0, altura_roi - faixa_base_pixels)
    _, pontos_x = np.where(mascara_contorno[y_base:, :] > 0)
    if pontos_x.size == 0:
        return float(largura_roi / 2.0)
    return float(np.mean(pontos_x.astype(np.float32)))


def _calcular_centro_x_faixa(mascara, y_inicio, y_fim):
    altura_roi, largura_roi = mascara.shape[:2]
    y_inicio = int(_limitar(y_inicio, 0, altura_roi))
    y_fim = int(_limitar(y_fim, 0, altura_roi))
    if y_fim <= y_inicio:
        return None, 0.0

    recorte = mascara[y_inicio:y_fim, :]
    pontos_y, pontos_x = np.where(recorte > 0)
    if pontos_x.size == 0:
        return None, 0.0

    centro_x = float(np.mean(pontos_x.astype(np.float32)))
    densidade = float(pontos_x.size) / float(max(1, recorte.size))
    return centro_x, densidade


def _detectar_cotovelo_90_branco(info_linha, mascara_contorno, configuracao):
    resultado_vazio = {
        "curva_90_literal_esquerda": False,
        "curva_90_literal_direita": False,
        "frente_branca_curva_90": False,
        "confianca_curva_90_literal": 0.0,
        "deslocamento_topo_curva_90": 0.0,
        "densidade_topo_curva_90": 0.0,
    }

    if info_linha["contorno"] is None:
        return resultado_vazio

    altura_roi, largura_roi = mascara_contorno.shape[:2]
    y_faixa_superior = int(altura_roi * configuracao.faixa_superior_curva_90)
    y_faixa_inferior = int(altura_roi * (1.0 - configuracao.faixa_inferior_curva_90))

    centro_x_base = _calcular_centro_x_base_contorno(mascara_contorno, configuracao)
    largura_base = max(
        10.0,
        float(info_linha.get("largura_caixa", 0)),
        float(largura_roi) * float(configuracao.largura_janela_branca_curva_90),
    )
    meia_janela_frontal = max(6, int(round(largura_base * 0.35)))
    largura_lateral = max(
        meia_janela_frontal + 6,
        int(round(largura_roi * float(configuracao.largura_janela_lateral_curva_90))),
    )

    x_coluna_inicio = int(round(centro_x_base - meia_janela_frontal))
    x_coluna_fim = int(round(centro_x_base + meia_janela_frontal))
    dens_frente = _densidade_faixa(
        mascara_contorno,
        x_coluna_inicio,
        x_coluna_fim,
        0,
        y_faixa_superior,
    )
    dens_coluna_base = _densidade_faixa(
        mascara_contorno,
        x_coluna_inicio,
        x_coluna_fim,
        y_faixa_inferior,
        altura_roi,
    )
    dens_top_left_proximo = _densidade_faixa(
        mascara_contorno,
        int(round(centro_x_base - largura_lateral)),
        x_coluna_inicio,
        0,
        y_faixa_superior,
    )
    dens_top_right_proximo = _densidade_faixa(
        mascara_contorno,
        x_coluna_fim,
        int(round(centro_x_base + largura_lateral)),
        0,
        y_faixa_superior,
    )
    centro_topo, dens_topo_total = _calcular_centro_x_faixa(
        mascara_contorno,
        0,
        y_faixa_superior,
    )
    if centro_topo is None:
        deslocamento_topo = 0.0
    else:
        deslocamento_topo = (centro_topo - centro_x_base) / max(1.0, largura_roi / 2.0)
    deslocamento_topo = float(_limitar(deslocamento_topo, -1.0, 1.0))

    frente_branca = dens_frente <= float(configuracao.densidade_frontal_max_curva_90)
    coluna_base_valida = dens_coluna_base >= float(configuracao.densidade_base_centro_curva_90)
    topo_deslocado_esquerda = bool(
        dens_topo_total >= float(configuracao.densidade_minima_topo_curva_90)
        and deslocamento_topo <= -float(configuracao.limiar_deslocamento_topo_curva_90)
    )
    topo_deslocado_direita = bool(
        dens_topo_total >= float(configuracao.densidade_minima_topo_curva_90)
        and deslocamento_topo >= float(configuracao.limiar_deslocamento_topo_curva_90)
    )
    frente_aceitavel = bool(
        frente_branca
        or (
            dens_frente
            <= max(
                float(configuracao.densidade_frontal_max_curva_90) * 2.0,
                float(configuracao.densidade_lateral_curva_90) * 0.65,
            )
            and (topo_deslocado_esquerda or topo_deslocado_direita)
        )
    )

    curva_90_literal_esquerda = bool(
        frente_aceitavel
        and coluna_base_valida
        and (
            dens_top_left_proximo >= float(configuracao.densidade_lateral_curva_90)
            or topo_deslocado_esquerda
        )
        and dens_top_right_proximo <= float(configuracao.densidade_oposta_max_curva_90)
    )
    curva_90_literal_direita = bool(
        frente_aceitavel
        and coluna_base_valida
        and (
            dens_top_right_proximo >= float(configuracao.densidade_lateral_curva_90)
            or topo_deslocado_direita
        )
        and dens_top_left_proximo <= float(configuracao.densidade_oposta_max_curva_90)
    )

    confianca_literal = 0.0
    if curva_90_literal_esquerda:
        confianca_literal = min(
            1.0,
            dens_top_left_proximo
            + abs(min(0.0, deslocamento_topo)) * 0.45
            + (1.0 - dens_frente) * 0.30
            + dens_coluna_base * 0.35,
        )
    elif curva_90_literal_direita:
        confianca_literal = min(
            1.0,
            dens_top_right_proximo
            + max(0.0, deslocamento_topo) * 0.45
            + (1.0 - dens_frente) * 0.30
            + dens_coluna_base * 0.35,
        )

    return {
        "curva_90_literal_esquerda": curva_90_literal_esquerda,
        "curva_90_literal_direita": curva_90_literal_direita,
        "frente_branca_curva_90": frente_aceitavel and coluna_base_valida,
        "confianca_curva_90_literal": float(confianca_literal),
        "deslocamento_topo_curva_90": deslocamento_topo,
        "densidade_topo_curva_90": float(dens_topo_total),
    }


def _detectar_curva_90(info_linha, mascara_linha, erro_lookahead, confianca_lookahead, configuracao):
    resultado_vazio = {
        "curva_90_esquerda": False,
        "curva_90_direita": False,
        "confianca_curva_90": 0.0,
        "curva_90_literal_esquerda": False,
        "curva_90_literal_direita": False,
        "frente_branca_curva_90": False,
        "confianca_curva_90_literal": 0.0,
        "deslocamento_topo_curva_90": 0.0,
        "densidade_topo_curva_90": 0.0,
    }

    if not info_linha["linha_encontrada"] or info_linha["contorno"] is None:
        return resultado_vazio
    if info_linha["confianca"] < configuracao.limiar_confianca_curva_90:
        return resultado_vazio

    altura_roi, largura_roi = mascara_linha.shape[:2]
    mascara_contorno = np.zeros_like(mascara_linha)
    cv2.drawContours(mascara_contorno, [info_linha["contorno"]], -1, 255, thickness=cv2.FILLED)
    deteccao_literal = _detectar_cotovelo_90_branco(
        info_linha,
        mascara_contorno,
        configuracao,
    )

    y_faixa_superior = int(altura_roi * configuracao.faixa_superior_curva_90)
    y_faixa_inferior = int(altura_roi * (1.0 - configuracao.faixa_inferior_curva_90))
    terco = max(1, largura_roi // 3)

    dens_top_left = _densidade_faixa(mascara_contorno, 0, terco, 0, y_faixa_superior)
    dens_top_right = _densidade_faixa(mascara_contorno, largura_roi - terco, largura_roi, 0, y_faixa_superior)
    dens_base_center = _densidade_faixa(
        mascara_contorno,
        terco,
        largura_roi - terco,
        y_faixa_inferior,
        altura_roi,
    )

    if dens_base_center < configuracao.densidade_base_centro_curva_90:
        return {
            **resultado_vazio,
            **deteccao_literal,
            "curva_90_esquerda": deteccao_literal["curva_90_literal_esquerda"],
            "curva_90_direita": deteccao_literal["curva_90_literal_direita"],
            "confianca_curva_90": deteccao_literal["confianca_curva_90_literal"],
        }

    curva_90_esquerda = False
    curva_90_direita = False
    confianca_curva_90 = 0.0

    if (
        confianca_lookahead >= configuracao.limiar_confianca_lookahead_curva_90
        and deteccao_literal["frente_branca_curva_90"]
    ):
        erro_base = float(info_linha["erro_suavizado"])
        if (
            abs(erro_lookahead) >= configuracao.limiar_erro_lookahead_curva_90
            and (abs(erro_lookahead) - abs(erro_base)) >= configuracao.limiar_delta_erro_curva_90
            and abs(erro_base) <= configuracao.limiar_erro_base_curva_90
        ):
            curva_90_esquerda = bool(
                erro_lookahead < 0.0
                and dens_top_left >= configuracao.densidade_lateral_curva_90
                and dens_top_right <= configuracao.densidade_oposta_max_curva_90
            )
            curva_90_direita = bool(
                erro_lookahead > 0.0
                and dens_top_right >= configuracao.densidade_lateral_curva_90
                and dens_top_left <= configuracao.densidade_oposta_max_curva_90
            )

            if curva_90_esquerda:
                confianca_curva_90 = min(
                    1.0,
                    dens_top_left
                    + abs(erro_lookahead) * 0.5
                    + deteccao_literal["confianca_curva_90_literal"] * 0.25,
                )
            elif curva_90_direita:
                confianca_curva_90 = min(
                    1.0,
                    dens_top_right
                    + abs(erro_lookahead) * 0.5
                    + deteccao_literal["confianca_curva_90_literal"] * 0.25,
                )

    curva_90_esquerda = bool(curva_90_esquerda or deteccao_literal["curva_90_literal_esquerda"])
    curva_90_direita = bool(curva_90_direita or deteccao_literal["curva_90_literal_direita"])
    confianca_curva_90 = max(confianca_curva_90, deteccao_literal["confianca_curva_90_literal"])

    return {
        "curva_90_esquerda": curva_90_esquerda,
        "curva_90_direita": curva_90_direita,
        "confianca_curva_90": float(confianca_curva_90),
        "curva_90_literal_esquerda": deteccao_literal["curva_90_literal_esquerda"],
        "curva_90_literal_direita": deteccao_literal["curva_90_literal_direita"],
        "frente_branca_curva_90": deteccao_literal["frente_branca_curva_90"],
        "confianca_curva_90_literal": deteccao_literal["confianca_curva_90_literal"],
        "deslocamento_topo_curva_90": deteccao_literal["deslocamento_topo_curva_90"],
        "densidade_topo_curva_90": deteccao_literal["densidade_topo_curva_90"],
    }


def _atualizar_memoria_curva_90(info_linha, deteccao_curva_90, estado, configuracao, tempo_atual):
    confianca_linha = float(info_linha.get("confianca", 0.0))
    deslocamento_topo = float(deteccao_curva_90.get("deslocamento_topo_curva_90", 0.0))
    densidade_topo = float(deteccao_curva_90.get("densidade_topo_curva_90", 0.0))
    confianca_curva = float(deteccao_curva_90.get("confianca_curva_90", 0.0))

    hint_esquerda = bool(
        deteccao_curva_90.get("curva_90_esquerda")
        or (
            deslocamento_topo <= -float(configuracao.limiar_deslocamento_topo_memoria_curva_90)
            and densidade_topo >= float(configuracao.densidade_minima_topo_memoria_curva_90)
            and confianca_linha >= max(0.08, float(configuracao.limiar_confianca) * 0.8)
        )
    )
    hint_direita = bool(
        deteccao_curva_90.get("curva_90_direita")
        or (
            deslocamento_topo >= float(configuracao.limiar_deslocamento_topo_memoria_curva_90)
            and densidade_topo >= float(configuracao.densidade_minima_topo_memoria_curva_90)
            and confianca_linha >= max(0.08, float(configuracao.limiar_confianca) * 0.8)
        )
    )

    confianca_hint = max(
        confianca_curva,
        min(1.0, abs(deslocamento_topo) * 1.8 + densidade_topo * 1.4 + confianca_linha * 0.3),
    )

    if hint_esquerda:
        estado.instante_ultima_curva_90_esquerda = tempo_atual
        estado.confianca_memoria_curva_90_esquerda = confianca_hint
    if hint_direita:
        estado.instante_ultima_curva_90_direita = tempo_atual
        estado.confianca_memoria_curva_90_direita = confianca_hint

    memoria_ativa_esquerda = (
        (tempo_atual - estado.instante_ultima_curva_90_esquerda) <= float(configuracao.tempo_memoria_curva_90)
        and estado.confianca_memoria_curva_90_esquerda >= float(configuracao.limiar_confianca_memoria_curva_90)
    )
    memoria_ativa_direita = (
        (tempo_atual - estado.instante_ultima_curva_90_direita) <= float(configuracao.tempo_memoria_curva_90)
        and estado.confianca_memoria_curva_90_direita >= float(configuracao.limiar_confianca_memoria_curva_90)
    )

    if memoria_ativa_esquerda and memoria_ativa_direita:
        if estado.instante_ultima_curva_90_esquerda >= estado.instante_ultima_curva_90_direita:
            memoria_ativa_direita = False
        else:
            memoria_ativa_esquerda = False

    confianca_memoria = 0.0
    if memoria_ativa_esquerda:
        confianca_memoria = float(estado.confianca_memoria_curva_90_esquerda)
    elif memoria_ativa_direita:
        confianca_memoria = float(estado.confianca_memoria_curva_90_direita)

    return {
        "curva_90_memoria_esquerda": bool(memoria_ativa_esquerda),
        "curva_90_memoria_direita": bool(memoria_ativa_direita),
        "confianca_curva_90_memoria": float(_limitar(confianca_memoria, 0.0, 1.0)),
    }


def _detectar_verde(quadro_bgr, configuracao):
    y_inicio, y_fim, largura_quadro = _obter_limites_roi(quadro_bgr.shape, configuracao.roi_verde)
    quadro_roi_bgr = quadro_bgr[y_inicio:y_fim]
    quadro_hsv = cv2.cvtColor(quadro_roi_bgr, cv2.COLOR_BGR2HSV)

    limite_inferior = np.array(
        [configuracao.verde_h_min, configuracao.verde_s_min, configuracao.verde_v_min],
        dtype=np.uint8,
    )
    limite_superior = np.array([configuracao.verde_h_max, 255, 255], dtype=np.uint8)
    mascara_verde = cv2.inRange(quadro_hsv, limite_inferior, limite_superior)

    nucleo = np.ones((5, 5), dtype=np.uint8)
    mascara_verde = cv2.morphologyEx(mascara_verde, cv2.MORPH_OPEN, nucleo, iterations=1)
    mascara_verde = cv2.morphologyEx(mascara_verde, cv2.MORPH_CLOSE, nucleo, iterations=2)

    contornos, _ = cv2.findContours(mascara_verde, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    melhor = None

    for contorno in contornos:
        area = float(cv2.contourArea(contorno))
        if area < configuracao.area_minima_verde:
            continue

        x, y, largura, altura = cv2.boundingRect(contorno)
        pontuacao = area + (altura * largura * 0.15)
        if melhor is None or pontuacao > melhor["pontuacao"]:
            melhor = {
                "pontuacao": pontuacao,
                "area": area,
                "caixa": (x, y + y_inicio, largura, altura),
                "centro": (x + (largura / 2.0), y_inicio + y + (altura / 2.0)),
            }

    if melhor is None:
        return {
            "verde_detectado": False,
            "confianca_verde": 0.0,
            "area_verde": 0.0,
            "caixa_verde": None,
            "centro_verde": None,
            "faixa_verde": (y_inicio, y_fim, largura_quadro),
        }

    area_referencia = max(1.0, float(mascara_verde.shape[0] * mascara_verde.shape[1]))
    confianca_verde = float(_limitar((melhor["area"] / area_referencia) / 0.08, 0.0, 1.0))
    return {
        "verde_detectado": True,
        "confianca_verde": confianca_verde,
        "area_verde": melhor["area"],
        "caixa_verde": melhor["caixa"],
        "centro_verde": melhor["centro"],
        "faixa_verde": (y_inicio, y_fim, largura_quadro),
    }


def analisar_quadro(quadro_bgr, configuracao, estado):
    tempo_atual = time.monotonic()
    y_roi, y_fim, largura_quadro = _obter_limites_roi(quadro_bgr.shape, configuracao.roi)

    quadro_roi_bgr = quadro_bgr[y_roi:y_fim].copy()
    mascara_linha_crua, limiar_usado, quadro_suave = _gerar_mascara_linha(
        quadro_roi_bgr,
        configuracao.limiar_binario,
        configuracao.inverter_linha,
    )
    mascara_linha_base = _pos_processar_mascara_linha(mascara_linha_crua.copy())
    mascara_linha_refinada = _refinar_mascara_linha(
        mascara_linha_crua.copy(),
        quadro_suave,
        configuracao,
    )

    info_linha_refinada = _selecionar_linha(mascara_linha_refinada, estado, configuracao)
    info_linha_base = _selecionar_linha(mascara_linha_base, estado, configuracao)
    info_linha, mascara_linha, origem_mascara = _escolher_resultado_linha(
        info_linha_refinada,
        mascara_linha_refinada,
        info_linha_base,
        mascara_linha_base,
        configuracao,
    )

    erro_lookahead, confianca_lookahead, ponto_lookahead = _calcular_erro_lookahead(
        info_linha,
        mascara_linha,
        estado,
        configuracao,
    )
    deteccao_curva_90 = _detectar_curva_90(
        info_linha,
        mascara_linha,
        erro_lookahead,
        confianca_lookahead,
        configuracao,
    )
    memoria_curva_90 = _atualizar_memoria_curva_90(
        info_linha,
        deteccao_curva_90,
        estado,
        configuracao,
        tempo_atual,
    )
    indicadores_laterais = _calcular_indicadores_laterais(
        info_linha,
        mascara_linha.shape[1],
        configuracao,
    )
    deteccao_verde = _detectar_verde(quadro_bgr, configuracao)

    if info_linha["linha_encontrada"]:
        estado.tempo_ultima_linha = tempo_atual
        estado.centro_x_anterior = info_linha["centro_x"]
        estado.erro_suavizado_anterior = info_linha["erro_suavizado"]
        estado.erro_lookahead_anterior = erro_lookahead

    tempo_sem_linha = max(0.0, tempo_atual - estado.tempo_ultima_linha)

    quadro_debug = quadro_bgr.copy()
    cv2.rectangle(quadro_debug, (0, y_roi), (largura_quadro - 1, y_fim - 1), (255, 255, 0), 2)

    centro_quadro_x = largura_quadro // 2
    cv2.line(quadro_debug, (centro_quadro_x, y_roi), (centro_quadro_x, y_fim), (255, 0, 0), 2)

    if info_linha["contorno"] is not None:
        deslocamento = np.array([[[0, y_roi]]], dtype=np.int32)
        contorno_deslocado = info_linha["contorno"] + deslocamento
        cv2.drawContours(quadro_debug, [contorno_deslocado], -1, (0, 0, 255), 2)

    if info_linha["linha_encontrada"]:
        centro_linha = (int(info_linha["centro_x"]), y_roi + int(info_linha["centro_y"]))
        cv2.circle(quadro_debug, centro_linha, 7, (0, 165, 255), -1)
        cv2.line(quadro_debug, (centro_quadro_x, centro_linha[1]), centro_linha, (0, 165, 255), 2)

    if ponto_lookahead is not None:
        ponto_debug = (int(ponto_lookahead[0]), y_roi + int(ponto_lookahead[1]))
        cv2.circle(quadro_debug, ponto_debug, 6, (0, 255, 0), -1)
        cv2.line(quadro_debug, (centro_quadro_x, ponto_debug[1]), ponto_debug, (0, 255, 0), 2)

    y_verde_inicio, y_verde_fim, _ = deteccao_verde["faixa_verde"]
    cv2.rectangle(quadro_debug, (0, y_verde_inicio), (largura_quadro - 1, y_verde_fim - 1), (0, 96, 0), 1)

    if deteccao_verde["caixa_verde"] is not None:
        x_verde, y_verde, largura_verde, altura_verde = deteccao_verde["caixa_verde"]
        cv2.rectangle(
            quadro_debug,
            (int(x_verde), int(y_verde)),
            (int(x_verde + largura_verde), int(y_verde + altura_verde)),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            quadro_debug,
            "VERDE",
            (int(x_verde), max(18, int(y_verde) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    textos = [
        f"linha={'SIM' if info_linha['linha_encontrada'] else 'NAO'} conf={info_linha['confianca']:.2f}",
        f"erro={info_linha['erro_suavizado']:+.3f} bruto={info_linha['erro_bruto']:+.3f}",
        f"lookahead={erro_lookahead:+.3f} conf_la={confianca_lookahead:.2f}",
        (
            "90="
            f"{'E' if deteccao_curva_90['curva_90_esquerda'] else ('D' if deteccao_curva_90['curva_90_direita'] else 'NAO')}"
            f" lit={('E' if deteccao_curva_90['curva_90_literal_esquerda'] else ('D' if deteccao_curva_90['curva_90_literal_direita'] else 'NAO'))}"
            f" branco={('SIM' if deteccao_curva_90['frente_branca_curva_90'] else 'NAO')}"
            f" conf90={deteccao_curva_90['confianca_curva_90']:.2f}"
        ),
        (
            f"topo90={deteccao_curva_90.get('deslocamento_topo_curva_90', 0.0):+.2f} "
            f"densTopo={deteccao_curva_90.get('densidade_topo_curva_90', 0.0):.2f}"
        ),
        (
            "mem90="
            f"{'E' if memoria_curva_90['curva_90_memoria_esquerda'] else ('D' if memoria_curva_90['curva_90_memoria_direita'] else 'NAO')}"
            f" confM={memoria_curva_90['confianca_curva_90_memoria']:.2f}"
        ),
        (
            "borda="
            f"{'E' if indicadores_laterais['linha_toca_borda_esquerda'] else ('D' if indicadores_laterais['linha_toca_borda_direita'] else 'NAO')}"
            f" largura={indicadores_laterais['largura_linha_relativa']:.2f}"
        ),
        (
            "verde="
            f"{'SIM' if deteccao_verde['verde_detectado'] else 'NAO'} "
            f"confV={deteccao_verde['confianca_verde']:.2f} areaV={deteccao_verde['area_verde']:.0f}"
        ),
        f"tempo_sem_linha={tempo_sem_linha:.2f}s",
        f"limiar={limiar_usado} mascara={origem_mascara}",
    ]

    for indice, texto in enumerate(textos):
        y_texto = 24 + indice * 22
        cv2.putText(
            quadro_debug,
            texto,
            (10, y_texto),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return {
        "erro_linha": info_linha["erro_suavizado"],
        "erro_bruto": info_linha["erro_bruto"],
        "erro_lookahead": erro_lookahead,
        "confianca_lookahead": confianca_lookahead,
        "curva_90_esquerda": deteccao_curva_90["curva_90_esquerda"],
        "curva_90_direita": deteccao_curva_90["curva_90_direita"],
        "confianca_curva_90": deteccao_curva_90["confianca_curva_90"],
        "curva_90_literal_esquerda": deteccao_curva_90["curva_90_literal_esquerda"],
        "curva_90_literal_direita": deteccao_curva_90["curva_90_literal_direita"],
        "frente_branca_curva_90": deteccao_curva_90["frente_branca_curva_90"],
        "confianca_curva_90_literal": deteccao_curva_90["confianca_curva_90_literal"],
        "deslocamento_topo_curva_90": deteccao_curva_90["deslocamento_topo_curva_90"],
        "densidade_topo_curva_90": deteccao_curva_90["densidade_topo_curva_90"],
        "curva_90_memoria_esquerda": memoria_curva_90["curva_90_memoria_esquerda"],
        "curva_90_memoria_direita": memoria_curva_90["curva_90_memoria_direita"],
        "confianca_curva_90_memoria": memoria_curva_90["confianca_curva_90_memoria"],
        "verde_detectado": deteccao_verde["verde_detectado"],
        "confianca_verde": deteccao_verde["confianca_verde"],
        "area_verde": deteccao_verde["area_verde"],
        "confianca_linha": info_linha["confianca"],
        "linha_encontrada": info_linha["linha_encontrada"],
        "linha_toca_borda_esquerda": indicadores_laterais["linha_toca_borda_esquerda"],
        "linha_toca_borda_direita": indicadores_laterais["linha_toca_borda_direita"],
        "centro_linha_normalizado": indicadores_laterais["centro_linha_normalizado"],
        "largura_linha_relativa": indicadores_laterais["largura_linha_relativa"],
        "tempo_sem_linha": tempo_sem_linha,
        "quadro_debug": quadro_debug,
        "mascara_linha": mascara_linha,
    }

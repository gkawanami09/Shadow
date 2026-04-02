import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class ConfiguracaoVisao:
    roi: float = 0.45
    limiar_binario: int | None = None
    inverter_linha: bool = False
    area_minima_linha: int = 260
    area_minima_contorno: int = 90
    limiar_confianca: float = 0.12
    suavizacao_offset: float = 0.35
    historico_maximo: int = 16
    limiar_intersecao: float = 0.52
    limiar_lado_intersecao: float = 0.18
    minimo_ramos_intersecao: int = 2
    verde_hmin: int = 35
    verde_hmax: int = 95
    verde_smin: int = 65
    verde_vmin: int = 65
    verde_area_minima: int = 230
    verde_area_falsa: int = 90
    verde_zona_min: float = 0.45
    verde_zona_max: float = 0.95
    verde_margem_pre_intersecao: float = 0.06
    vermelho_hmin1: int = 0
    vermelho_hmax1: int = 12
    vermelho_hmin2: int = 168
    vermelho_hmax2: int = 180
    vermelho_smin: int = 70
    vermelho_vmin: int = 60
    vermelho_area_minima: int = 260
    vermelho_area_falsa: int = 110
    vermelho_zona_min: float = 0.56
    vermelho_zona_max: float = 0.98
    vermelho_frames_confirmacao: int = 3
    vermelho_frames_liberacao: int = 5
    tempo_memoria_gap: float = 0.75
    limiar_offset_reto_gap: float = 0.11
    limiar_desvio_gap: float = 0.08
    limiar_confianca_gap: float = 0.42


@dataclass
class EstadoVisao:
    tempo_ultima_linha: float = field(default_factory=time.monotonic)
    centro_x_linha_anterior: float | None = None
    erro_linha_suavizado: float = 0.0
    lado_ultimo_erro: int = 0
    historico_offsets: deque = field(default_factory=lambda: deque(maxlen=16))
    historico_confianca: deque = field(default_factory=lambda: deque(maxlen=16))
    contador_vermelho_confirmacao: int = 0
    contador_vermelho_liberacao: int = 0
    vermelho_confirmado: bool = False


def _limitar(valor, minimo, maximo):
    return max(minimo, min(maximo, valor))


def _limites_roi(formato_quadro, fracao_roi):
    altura_total, largura_total = formato_quadro[:2]
    fracao_roi = float(_limitar(fracao_roi, 0.20, 0.90))
    altura_roi = max(1, int(altura_total * fracao_roi))
    y_inicial = altura_total - altura_roi
    return y_inicial, altura_total, largura_total, altura_roi


def _gerar_mascara_linha(quadro_roi_bgr, limiar_binario, inverter_linha):
    quadro_cinza = cv2.cvtColor(quadro_roi_bgr, cv2.COLOR_BGR2GRAY)
    quadro_suave = cv2.GaussianBlur(quadro_cinza, (5, 5), 0)
    modo_binario = cv2.THRESH_BINARY if inverter_linha else cv2.THRESH_BINARY_INV

    if limiar_binario is None:
        _, mascara_linha = cv2.threshold(
            quadro_suave,
            0,
            255,
            modo_binario | cv2.THRESH_OTSU,
        )
        limiar_usado = "otsu"
    else:
        _, mascara_linha = cv2.threshold(
            quadro_suave,
            int(limiar_binario),
            255,
            modo_binario,
        )
        limiar_usado = str(int(limiar_binario))

    nucleo_abertura = np.ones((3, 3), dtype=np.uint8)
    nucleo_fechamento = np.ones((5, 5), dtype=np.uint8)
    mascara_linha = cv2.morphologyEx(mascara_linha, cv2.MORPH_OPEN, nucleo_abertura, iterations=1)
    mascara_linha = cv2.morphologyEx(mascara_linha, cv2.MORPH_CLOSE, nucleo_fechamento, iterations=2)
    return quadro_cinza, mascara_linha, limiar_usado


def _extrair_segmentos(vetor_binario, largura_minima):
    segmentos = []
    inicio = None

    for indice, ativo in enumerate(vetor_binario):
        if ativo and inicio is None:
            inicio = indice
        elif not ativo and inicio is not None:
            fim = indice - 1
            if (fim - inicio + 1) >= largura_minima:
                segmentos.append((inicio, fim))
            inicio = None

    if inicio is not None:
        fim = len(vetor_binario) - 1
        if (fim - inicio + 1) >= largura_minima:
            segmentos.append((inicio, fim))

    return segmentos


def _detectar_intersecao(mascara_linha, configuracao):
    altura, largura = mascara_linha.shape[:2]
    y_faixa_inicio = int(altura * 0.16)
    y_faixa_fim = int(altura * 0.48)
    if y_faixa_fim <= y_faixa_inicio:
        y_faixa_fim = min(altura, y_faixa_inicio + 2)

    faixa = mascara_linha[y_faixa_inicio:y_faixa_fim, :]
    if faixa.size == 0:
        return {
            "intersecao": False,
            "ramos": 0,
            "largura_total": 0.0,
            "cobertura_esquerda": 0.0,
            "cobertura_direita": 0.0,
            "y_cruzamento_norm": None,
            "segmentos": [],
        }

    altura_faixa = max(1, faixa.shape[0])
    largura_minima_segmento = max(6, int(largura * 0.03))
    colunas_ativas = np.count_nonzero(faixa, axis=0) >= int(altura_faixa * 0.33)
    segmentos = _extrair_segmentos(colunas_ativas, largura_minima_segmento)

    cobertura_total = float(np.count_nonzero(colunas_ativas) / max(1, largura))
    metade = largura // 2
    cobertura_esquerda = float(np.count_nonzero(colunas_ativas[:metade]) / max(1, metade))
    cobertura_direita = float(np.count_nonzero(colunas_ativas[metade:]) / max(1, largura - metade))

    cobertura_linhas = np.count_nonzero(faixa, axis=1) / max(1, largura)
    indices_cruzamento = np.where(cobertura_linhas >= configuracao.limiar_intersecao)[0]
    if len(indices_cruzamento) > 0:
        y_cruzamento = y_faixa_inicio + int(indices_cruzamento[0])
        y_cruzamento_norm = y_cruzamento / max(1, altura)
    else:
        y_cruzamento_norm = None

    intersecao = bool(
        len(segmentos) >= configuracao.minimo_ramos_intersecao
        and cobertura_total >= configuracao.limiar_intersecao
        and cobertura_esquerda >= configuracao.limiar_lado_intersecao
        and cobertura_direita >= configuracao.limiar_lado_intersecao
    )

    return {
        "intersecao": intersecao,
        "ramos": len(segmentos),
        "largura_total": cobertura_total,
        "cobertura_esquerda": cobertura_esquerda,
        "cobertura_direita": cobertura_direita,
        "y_cruzamento_norm": y_cruzamento_norm,
        "segmentos": segmentos,
    }


def _selecionar_contorno_linha(mascara_linha, estado, configuracao):
    altura, largura = mascara_linha.shape[:2]
    contornos, _ = cv2.findContours(mascara_linha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    melhor = None
    melhor_pontuacao = -1.0
    candidatos = []

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

        if estado.centro_x_linha_anterior is None:
            proximidade_anterior = 0.5
        else:
            distancia = abs(centro_x - estado.centro_x_linha_anterior)
            proximidade_anterior = 1.0 - min(1.0, distancia / max(1.0, largura))

        proximidade_centro = 1.0 - min(1.0, abs(centro_x - (largura / 2.0)) / max(1.0, largura / 2.0))
        toque_base = 1.0 if (y + altura_caixa) >= int(altura * 0.92) else 0.0
        altura_relativa = altura_caixa / max(1.0, altura)
        area_relativa = area / max(1.0, altura * largura)

        pontuacao = (
            area_relativa * 2.0
            + proximidade_anterior * 1.2
            + toque_base * 0.8
            + altura_relativa * 0.5
            + proximidade_centro * 0.3
        )

        candidato = {
            "contorno": contorno,
            "area": area,
            "centro_x": centro_x,
            "centro_y": centro_y,
            "caixa": (x, y, largura_caixa, altura_caixa),
            "pontuacao": pontuacao,
            "toque_base": toque_base,
            "area_relativa": area_relativa,
            "proximidade_anterior": proximidade_anterior,
        }
        candidatos.append(candidato)

        if pontuacao > melhor_pontuacao:
            melhor = candidato
            melhor_pontuacao = pontuacao

    if melhor is None or melhor["area"] < configuracao.area_minima_linha:
        return {
            "linha_encontrada": False,
            "erro_bruto": 0.0,
            "erro_suavizado": estado.erro_linha_suavizado,
            "confianca": 0.0,
            "centro_x": None,
            "centro_y": None,
            "caixa": None,
            "contorno": None,
            "candidatos": candidatos,
        }

    erro_bruto = ((melhor["centro_x"] - (largura / 2.0)) / max(1.0, largura / 2.0))
    erro_bruto = float(_limitar(erro_bruto, -1.0, 1.0))

    erro_suavizado = (
        (1.0 - configuracao.suavizacao_offset) * estado.erro_linha_suavizado
        + configuracao.suavizacao_offset * erro_bruto
    )

    confianca_area = min(1.0, melhor["area_relativa"] / 0.22)
    confianca_continuidade = melhor["proximidade_anterior"]
    confianca_base = 1.0 if melhor["toque_base"] > 0.5 else 0.35
    confianca = float(_limitar(0.50 * confianca_area + 0.30 * confianca_continuidade + 0.20 * confianca_base, 0.0, 1.0))

    return {
        "linha_encontrada": True,
        "erro_bruto": erro_bruto,
        "erro_suavizado": float(_limitar(erro_suavizado, -1.0, 1.0)),
        "confianca": confianca,
        "centro_x": melhor["centro_x"],
        "centro_y": melhor["centro_y"],
        "caixa": melhor["caixa"],
        "contorno": melhor["contorno"],
        "candidatos": candidatos,
    }


def _detectar_verde(quadro_roi_bgr, configuracao, y_cruzamento_norm):
    quadro_hsv = cv2.cvtColor(quadro_roi_bgr, cv2.COLOR_BGR2HSV)
    limite_inferior = np.array([configuracao.verde_hmin, configuracao.verde_smin, configuracao.verde_vmin], dtype=np.uint8)
    limite_superior = np.array([configuracao.verde_hmax, 255, 255], dtype=np.uint8)
    mascara = cv2.inRange(quadro_hsv, limite_inferior, limite_superior)

    nucleo = np.ones((5, 5), dtype=np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, nucleo, iterations=1)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, nucleo, iterations=2)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    altura, largura = mascara.shape[:2]

    validos = []
    fora_zona = []
    falsos = []

    for contorno in contornos:
        area = float(cv2.contourArea(contorno))
        x, y, largura_caixa, altura_caixa = cv2.boundingRect(contorno)
        centro_x = x + (largura_caixa // 2)
        centro_y = y + (altura_caixa // 2)
        y_norm = centro_y / max(1.0, altura)

        dentro_zona = configuracao.verde_zona_min <= y_norm <= configuracao.verde_zona_max
        if y_cruzamento_norm is None:
            antes_intersecao = True
        else:
            antes_intersecao = y_norm >= min(0.99, y_cruzamento_norm + configuracao.verde_margem_pre_intersecao)

        item = {
            "area": area,
            "caixa": (x, y, largura_caixa, altura_caixa),
            "centro": (centro_x, centro_y),
            "dentro_zona": dentro_zona,
            "antes_intersecao": antes_intersecao,
            "lado": "ESQUERDA" if centro_x < (largura * 0.5) else "DIREITA",
        }

        if area < configuracao.verde_area_falsa:
            falsos.append(item)
            continue

        if area >= configuracao.verde_area_minima and dentro_zona and antes_intersecao:
            validos.append(item)
        else:
            fora_zona.append(item)

    quantidade_esquerda = sum(1 for item in validos if item["lado"] == "ESQUERDA")
    quantidade_direita = sum(1 for item in validos if item["lado"] == "DIREITA")

    if quantidade_esquerda > 0 and quantidade_direita > 0:
        tipo = "VERDE_DUPLO"
    elif quantidade_esquerda > 0:
        tipo = "VERDE_ESQUERDA"
    elif quantidade_direita > 0:
        tipo = "VERDE_DIREITA"
    elif len(fora_zona) > 0:
        tipo = "VERDE_FORA_ZONA"
    elif len(falsos) > 0:
        tipo = "VERDE_FALSO"
    else:
        tipo = "VERDE_AUSENTE"

    return {
        "mascara": mascara,
        "tipo": tipo,
        "validos": validos,
        "fora_zona": fora_zona,
        "falsos": falsos,
        "zona_min": configuracao.verde_zona_min,
        "zona_max": configuracao.verde_zona_max,
    }


def _detectar_vermelho(quadro_roi_bgr, configuracao, estado):
    quadro_hsv = cv2.cvtColor(quadro_roi_bgr, cv2.COLOR_BGR2HSV)

    limite_inferior_1 = np.array([configuracao.vermelho_hmin1, configuracao.vermelho_smin, configuracao.vermelho_vmin], dtype=np.uint8)
    limite_superior_1 = np.array([configuracao.vermelho_hmax1, 255, 255], dtype=np.uint8)
    limite_inferior_2 = np.array([configuracao.vermelho_hmin2, configuracao.vermelho_smin, configuracao.vermelho_vmin], dtype=np.uint8)
    limite_superior_2 = np.array([configuracao.vermelho_hmax2, 255, 255], dtype=np.uint8)

    mascara_1 = cv2.inRange(quadro_hsv, limite_inferior_1, limite_superior_1)
    mascara_2 = cv2.inRange(quadro_hsv, limite_inferior_2, limite_superior_2)
    mascara = cv2.bitwise_or(mascara_1, mascara_2)

    nucleo = np.ones((5, 5), dtype=np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, nucleo, iterations=1)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, nucleo, iterations=2)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    altura, largura = mascara.shape[:2]

    validos = []
    fora_zona = []
    falsos = []

    for contorno in contornos:
        area = float(cv2.contourArea(contorno))
        x, y, largura_caixa, altura_caixa = cv2.boundingRect(contorno)
        centro_x = x + (largura_caixa // 2)
        centro_y = y + (altura_caixa // 2)
        y_norm = centro_y / max(1.0, altura)

        dentro_zona = configuracao.vermelho_zona_min <= y_norm <= configuracao.vermelho_zona_max
        item = {
            "area": area,
            "caixa": (x, y, largura_caixa, altura_caixa),
            "centro": (centro_x, centro_y),
            "dentro_zona": dentro_zona,
        }

        if area < configuracao.vermelho_area_falsa:
            falsos.append(item)
            continue

        if area >= configuracao.vermelho_area_minima and dentro_zona:
            validos.append(item)
        else:
            fora_zona.append(item)

    vermelho_valido = len(validos) > 0

    if vermelho_valido:
        estado.contador_vermelho_confirmacao = min(
            configuracao.vermelho_frames_confirmacao,
            estado.contador_vermelho_confirmacao + 1,
        )
        estado.contador_vermelho_liberacao = 0
    else:
        estado.contador_vermelho_liberacao = min(
            configuracao.vermelho_frames_liberacao,
            estado.contador_vermelho_liberacao + 1,
        )
        if estado.contador_vermelho_confirmacao > 0:
            estado.contador_vermelho_confirmacao -= 1

    if estado.contador_vermelho_confirmacao >= configuracao.vermelho_frames_confirmacao:
        estado.vermelho_confirmado = True

    if estado.contador_vermelho_liberacao >= configuracao.vermelho_frames_liberacao:
        estado.vermelho_confirmado = False
        estado.contador_vermelho_confirmacao = 0

    if vermelho_valido:
        tipo = "VERMELHO_VALIDO"
    elif len(fora_zona) > 0:
        tipo = "VERMELHO_FORA_ZONA"
    elif len(falsos) > 0:
        tipo = "VERMELHO_FALSO"
    else:
        tipo = "VERMELHO_AUSENTE"

    return {
        "mascara": mascara,
        "tipo": tipo,
        "validos": validos,
        "fora_zona": fora_zona,
        "falsos": falsos,
        "vermelho_valido": vermelho_valido,
        "vermelho_confirmado": estado.vermelho_confirmado,
        "zona_min": configuracao.vermelho_zona_min,
        "zona_max": configuracao.vermelho_zona_max,
    }


def _atualizar_historico_linha(estado, configuracao, erro, confianca):
    if estado.historico_offsets.maxlen != configuracao.historico_maximo:
        estado.historico_offsets = deque(estado.historico_offsets, maxlen=configuracao.historico_maximo)
        estado.historico_confianca = deque(estado.historico_confianca, maxlen=configuracao.historico_maximo)

    estado.historico_offsets.append(float(erro))
    estado.historico_confianca.append(float(confianca))


def _calcular_gap_provavel(estado, configuracao, linha_encontrada, tempo_atual):
    tempo_sem_linha = max(0.0, tempo_atual - estado.tempo_ultima_linha)
    if linha_encontrada:
        return False, tempo_sem_linha

    if tempo_sem_linha > configuracao.tempo_memoria_gap:
        return False, tempo_sem_linha

    quantidade = min(8, len(estado.historico_offsets))
    if quantidade < 4:
        return False, tempo_sem_linha

    offsets = np.array(list(estado.historico_offsets)[-quantidade:], dtype=np.float32)
    confiancas = np.array(list(estado.historico_confianca)[-quantidade:], dtype=np.float32)

    media_abs = float(np.mean(np.abs(offsets)))
    desvio = float(np.std(offsets))
    confianca_media = float(np.mean(confiancas))

    gap_provavel = bool(
        media_abs <= configuracao.limiar_offset_reto_gap
        and desvio <= configuracao.limiar_desvio_gap
        and confianca_media >= configuracao.limiar_confianca_gap
    )
    return gap_provavel, tempo_sem_linha


def _desenhar_deteccao_cor(quadro_debug, y_roi, deteccao_cor, cor_valido, cor_fora, cor_falso):
    for item in deteccao_cor["validos"]:
        x, y, largura_caixa, altura_caixa = item["caixa"]
        cv2.rectangle(quadro_debug, (x, y_roi + y), (x + largura_caixa, y_roi + y + altura_caixa), cor_valido, 2)

    for item in deteccao_cor["fora_zona"]:
        x, y, largura_caixa, altura_caixa = item["caixa"]
        cv2.rectangle(quadro_debug, (x, y_roi + y), (x + largura_caixa, y_roi + y + altura_caixa), cor_fora, 1)

    for item in deteccao_cor["falsos"]:
        x, y, largura_caixa, altura_caixa = item["caixa"]
        cv2.rectangle(quadro_debug, (x, y_roi + y), (x + largura_caixa, y_roi + y + altura_caixa), cor_falso, 1)


def analisar_quadro(quadro_bgr, configuracao, estado):
    tempo_atual = time.monotonic()
    y_roi, y_fim, largura_quadro, _ = _limites_roi(quadro_bgr.shape, configuracao.roi)

    quadro_roi_bgr = quadro_bgr[y_roi:y_fim].copy()
    _, mascara_linha, limiar_usado = _gerar_mascara_linha(
        quadro_roi_bgr,
        configuracao.limiar_binario,
        configuracao.inverter_linha,
    )

    info_linha = _selecionar_contorno_linha(mascara_linha, estado, configuracao)
    info_intersecao = _detectar_intersecao(mascara_linha, configuracao)
    info_verde = _detectar_verde(quadro_roi_bgr, configuracao, info_intersecao["y_cruzamento_norm"])
    info_vermelho = _detectar_vermelho(quadro_roi_bgr, configuracao, estado)

    if info_linha["linha_encontrada"]:
        estado.tempo_ultima_linha = tempo_atual
        estado.centro_x_linha_anterior = info_linha["centro_x"]
        estado.erro_linha_suavizado = info_linha["erro_suavizado"]
        if info_linha["erro_suavizado"] > 0.02:
            estado.lado_ultimo_erro = 1
        elif info_linha["erro_suavizado"] < -0.02:
            estado.lado_ultimo_erro = -1

        _atualizar_historico_linha(
            estado,
            configuracao,
            info_linha["erro_suavizado"],
            info_linha["confianca"],
        )

    gap_provavel, tempo_sem_linha = _calcular_gap_provavel(
        estado,
        configuracao,
        info_linha["linha_encontrada"],
        tempo_atual,
    )

    quadro_debug = quadro_bgr.copy()
    cv2.rectangle(quadro_debug, (0, y_roi), (largura_quadro - 1, y_fim - 1), (255, 255, 0), 2)
    centro_quadro_x = largura_quadro // 2
    cv2.line(quadro_debug, (centro_quadro_x, y_roi), (centro_quadro_x, y_fim), (255, 0, 0), 2)

    for candidato in info_linha["candidatos"]:
        x, y, largura_caixa, altura_caixa = candidato["caixa"]
        cv2.rectangle(
            quadro_debug,
            (x, y_roi + y),
            (x + largura_caixa, y_roi + y + altura_caixa),
            (90, 90, 90),
            1,
        )

    if info_linha["contorno"] is not None:
        deslocamento = np.array([[[0, y_roi]]], dtype=np.int32)
        contorno_deslocado = info_linha["contorno"] + deslocamento
        cv2.drawContours(quadro_debug, [contorno_deslocado], -1, (0, 0, 255), 2)

    if info_linha["linha_encontrada"]:
        centro_linha = (int(info_linha["centro_x"]), y_roi + int(info_linha["centro_y"]))
        cv2.circle(quadro_debug, centro_linha, 7, (0, 165, 255), -1)
        cv2.line(quadro_debug, (centro_quadro_x, centro_linha[1]), centro_linha, (0, 165, 255), 2)

    y_zona_verde_min = y_roi + int(configuracao.verde_zona_min * mascara_linha.shape[0])
    y_zona_verde_max = y_roi + int(configuracao.verde_zona_max * mascara_linha.shape[0])
    cv2.line(quadro_debug, (0, y_zona_verde_min), (largura_quadro - 1, y_zona_verde_min), (0, 180, 0), 1)
    cv2.line(quadro_debug, (0, y_zona_verde_max), (largura_quadro - 1, y_zona_verde_max), (0, 180, 0), 1)

    y_zona_vermelho_min = y_roi + int(configuracao.vermelho_zona_min * mascara_linha.shape[0])
    y_zona_vermelho_max = y_roi + int(configuracao.vermelho_zona_max * mascara_linha.shape[0])
    cv2.line(quadro_debug, (0, y_zona_vermelho_min), (largura_quadro - 1, y_zona_vermelho_min), (0, 0, 180), 1)
    cv2.line(quadro_debug, (0, y_zona_vermelho_max), (largura_quadro - 1, y_zona_vermelho_max), (0, 0, 180), 1)

    _desenhar_deteccao_cor(
        quadro_debug,
        y_roi,
        info_verde,
        cor_valido=(0, 255, 0),
        cor_fora=(0, 200, 200),
        cor_falso=(110, 110, 110),
    )
    _desenhar_deteccao_cor(
        quadro_debug,
        y_roi,
        info_vermelho,
        cor_valido=(0, 0, 255),
        cor_fora=(0, 90, 200),
        cor_falso=(120, 120, 120),
    )

    if info_intersecao["y_cruzamento_norm"] is not None:
        y_cruzamento = y_roi + int(info_intersecao["y_cruzamento_norm"] * mascara_linha.shape[0])
        cv2.line(quadro_debug, (0, y_cruzamento), (largura_quadro - 1, y_cruzamento), (255, 140, 0), 1)

    for indice, segmento in enumerate(info_intersecao["segmentos"]):
        x_inicial, x_final = segmento
        y_segmento = y_roi + int(mascara_linha.shape[0] * 0.18)
        cv2.line(quadro_debug, (x_inicial, y_segmento + indice * 3), (x_final, y_segmento + indice * 3), (255, 140, 0), 2)

    texto_debug = [
        f"linha={'SIM' if info_linha['linha_encontrada'] else 'NAO'} conf={info_linha['confianca']:.2f}",
        f"erro={info_linha['erro_suavizado']:+.3f} bruto={info_linha['erro_bruto']:+.3f}",
        f"intersecao={'SIM' if info_intersecao['intersecao'] else 'NAO'} ramos={info_intersecao['ramos']}",
        f"verde={info_verde['tipo']}",
        f"vermelho={info_vermelho['tipo']} conf={info_vermelho['vermelho_confirmado']}",
        f"gap_provavel={gap_provavel} tempo_sem_linha={tempo_sem_linha:.2f}s",
        f"limiar={limiar_usado}",
    ]

    for indice, texto in enumerate(texto_debug):
        y_texto = 25 + indice * 22
        cv2.putText(
            quadro_debug,
            texto,
            (10, y_texto),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return {
        "erro_linha": info_linha["erro_suavizado"],
        "erro_bruto": info_linha["erro_bruto"],
        "confianca_linha": info_linha["confianca"],
        "linha_encontrada": info_linha["linha_encontrada"],
        "intersecao_detectada": info_intersecao["intersecao"],
        "ramos_intersecao": info_intersecao["ramos"],
        "tipo_verde": info_verde["tipo"],
        "verde_valido_pre_intersecao": info_verde["tipo"] in {"VERDE_ESQUERDA", "VERDE_DIREITA", "VERDE_DUPLO"},
        "tipo_vermelho": info_vermelho["tipo"],
        "vermelho_valido": info_vermelho["vermelho_valido"],
        "vermelho_confirmado": info_vermelho["vermelho_confirmado"],
        "gap_provavel": gap_provavel,
        "tempo_sem_linha": tempo_sem_linha,
        "lado_ultimo_erro": estado.lado_ultimo_erro,
        "largura_intersecao": info_intersecao["largura_total"],
        "quadro_debug": quadro_debug,
        "mascara_linha": mascara_linha,
        "mascara_verde": info_verde["mascara"],
        "mascara_vermelho": info_vermelho["mascara"],
    }

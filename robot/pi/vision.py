import time
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class ConfiguracaoVisao:
    roi: float = 0.45
    limiar_binario: int | None = None
    inverter_linha: bool = False
    area_minima_contorno: int = 180
    area_minima_linha: int = 320
    suavizacao_erro: float = 0.40
    limiar_confianca: float = 0.10


@dataclass
class EstadoVisao:
    tempo_ultima_linha: float = field(default_factory=time.monotonic)
    centro_x_anterior: float | None = None
    erro_suavizado_anterior: float = 0.0
    lado_ultimo_erro: int = 0


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

    nucleo_abertura = np.ones((3, 3), dtype=np.uint8)
    nucleo_fechamento = np.ones((5, 5), dtype=np.uint8)
    mascara_linha = cv2.morphologyEx(mascara_linha, cv2.MORPH_OPEN, nucleo_abertura, iterations=1)
    mascara_linha = cv2.morphologyEx(mascara_linha, cv2.MORPH_CLOSE, nucleo_fechamento, iterations=2)

    return mascara_linha, limiar_usado


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

        if estado.centro_x_anterior is None:
            proximidade_anterior = 0.5
        else:
            distancia_anterior = abs(centro_x - estado.centro_x_anterior)
            proximidade_anterior = 1.0 - min(1.0, distancia_anterior / max(1.0, largura_roi / 2.0))

        pontuacao = (
            area_relativa * 2.1
            + proximidade_base * 0.9
            + proximidade_anterior * 1.3
        )

        if pontuacao > melhor_pontuacao:
            melhor_pontuacao = pontuacao
            melhor = {
                "contorno": contorno,
                "area": area,
                "centro_x": centro_x,
                "centro_y": centro_y,
                "caixa": (x, y, largura_caixa, altura_caixa),
                "area_relativa": area_relativa,
                "proximidade_anterior": proximidade_anterior,
            }

    if melhor is None or melhor["area"] < configuracao.area_minima_linha:
        return {
            "linha_encontrada": False,
            "erro_bruto": 0.0,
            "erro_suavizado": estado.erro_suavizado_anterior,
            "confianca": 0.0,
            "centro_x": None,
            "centro_y": None,
            "caixa": None,
            "contorno": None,
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
    confianca = float(_limitar(0.68 * confianca_area + 0.32 * confianca_continuidade, 0.0, 1.0))

    return {
        "linha_encontrada": True,
        "erro_bruto": erro_bruto,
        "erro_suavizado": erro_suavizado,
        "confianca": confianca,
        "centro_x": melhor["centro_x"],
        "centro_y": melhor["centro_y"],
        "caixa": melhor["caixa"],
        "contorno": melhor["contorno"],
    }


def analisar_quadro(quadro_bgr, configuracao, estado):
    tempo_atual = time.monotonic()
    y_roi, y_fim, largura_quadro = _obter_limites_roi(quadro_bgr.shape, configuracao.roi)

    quadro_roi_bgr = quadro_bgr[y_roi:y_fim].copy()
    mascara_linha, limiar_usado = _gerar_mascara_linha(
        quadro_roi_bgr,
        configuracao.limiar_binario,
        configuracao.inverter_linha,
    )

    info_linha = _selecionar_linha(mascara_linha, estado, configuracao)

    if info_linha["linha_encontrada"]:
        estado.tempo_ultima_linha = tempo_atual
        estado.centro_x_anterior = info_linha["centro_x"]
        estado.erro_suavizado_anterior = info_linha["erro_suavizado"]

        if info_linha["erro_suavizado"] > 0.02:
            estado.lado_ultimo_erro = 1
        elif info_linha["erro_suavizado"] < -0.02:
            estado.lado_ultimo_erro = -1

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

    textos = [
        f"linha={'SIM' if info_linha['linha_encontrada'] else 'NAO'} conf={info_linha['confianca']:.2f}",
        f"erro={info_linha['erro_suavizado']:+.3f} bruto={info_linha['erro_bruto']:+.3f}",
        f"tempo_sem_linha={tempo_sem_linha:.2f}s",
        f"limiar={limiar_usado}",
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
        "confianca_linha": info_linha["confianca"],
        "linha_encontrada": info_linha["linha_encontrada"],
        "lado_ultimo_erro": estado.lado_ultimo_erro,
        "tempo_sem_linha": tempo_sem_linha,
        "quadro_debug": quadro_debug,
        "mascara_linha": mascara_linha,
    }

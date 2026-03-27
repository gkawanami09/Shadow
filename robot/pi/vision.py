import time
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class VisionConfig:
    roi: float = 0.40
    threshold: int | None = None
    invert: bool = False
    gap_tempo: float = 0.35
    intersecao_largura: float = 0.55
    intersecao_lado_min: float = 0.20
    giro_180_tempo: float = 1.2
    giro_180_offset: float = 1.0
    beco_cooldown: float = 2.0
    verde_hmin: int = 35
    verde_hmax: int = 95
    verde_smin: int = 60
    verde_vmin: int = 60
    verde_area_min: int = 250
    verde_zona: float = 0.45
    vermelho_hmin1: int = 0
    vermelho_hmax1: int = 10
    vermelho_hmin2: int = 170
    vermelho_hmax2: int = 180
    vermelho_smin: int = 70
    vermelho_vmin: int = 50
    vermelho_area_min: int = 250
    vermelho_zona: float = 0.45
    vermelho_tempo_parado: float = 20.0
    min_line_area: int = 300
    min_confidence: float = 0.08


@dataclass
class VisionState:
    last_seen_line_time: float = field(default_factory=time.monotonic)
    last_offset: float = 0.0
    last_command: str = "S"
    line_lost_since: float | None = None
    turn_until: float = 0.0
    turn_reason: str = ""
    dead_end_cooldown_until: float = 0.0
    red_stop_until: float = 0.0
    red_armed: bool = True


def _clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def _roi_bounds(frame_shape, roi_fraction):
    height, width = frame_shape[:2]
    roi_fraction = _clamp(roi_fraction, 0.10, 0.90)
    roi_height = max(1, int(height * roi_fraction))
    y0 = height - roi_height
    return y0, height, width, roi_height


def _build_line_mask(roi_bgr, threshold, invert):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh_mode = cv2.THRESH_BINARY_INV if not invert else cv2.THRESH_BINARY

    if threshold is None:
        _, binary = cv2.threshold(
            blurred, 0, 255, thresh_mode | cv2.THRESH_OTSU
        )
        threshold_used = "otsu"
    else:
        _, binary = cv2.threshold(blurred, threshold, 255, thresh_mode)
        threshold_used = str(threshold)

    kernel = np.ones((3, 3), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return gray, binary, threshold_used


def _detect_line(binary, min_line_area, frame_width):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "found": False,
            "center_x": None,
            "center_y": None,
            "offset": 0.0,
            "confidence": 0.0,
            "bbox": None,
            "contour": None,
            "intersection": False,
            "area": 0.0,
            "width_ratio": 0.0,
        }

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < min_line_area:
        return {
            "found": False,
            "center_x": None,
            "center_y": None,
            "offset": 0.0,
            "confidence": 0.0,
            "bbox": None,
            "contour": contour,
            "intersection": False,
            "area": area,
            "width_ratio": 0.0,
        }

    x, y, w, h = cv2.boundingRect(contour)
    moments = cv2.moments(contour)
    if moments["m00"] > 0:
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
    else:
        center_x = x + w // 2
        center_y = y + h // 2

    frame_center = frame_width / 2.0
    offset = (center_x - frame_center) / max(1.0, frame_center)
    width_ratio = w / max(1.0, frame_width)
    confidence = _clamp(area / (binary.shape[0] * binary.shape[1] * 0.45), 0.0, 1.0)
    return {
        "found": True,
        "center_x": center_x,
        "center_y": center_y,
        "offset": float(_clamp(offset, -1.0, 1.0)),
        "confidence": float(confidence),
        "bbox": (x, y, w, h),
        "contour": contour,
        "intersection": False,
        "area": area,
        "width_ratio": width_ratio,
    }


def _detect_intersection(line_mask, config):
    height, width = line_mask.shape[:2]
    if height < 5:
        return False

    linhas_teste = 3
    limite_total = int(width * config.intersecao_largura)
    limite_lado = int((width / 2.0) * config.intersecao_lado_min)

    for i in range(linhas_teste):
        y = int((i + 1) * height / (linhas_teste + 1))
        linha = line_mask[y : y + 1, :]
        if linha.size == 0:
            continue
        pixels_total = int(np.count_nonzero(linha))
        if pixels_total < limite_total:
            continue
        esquerda = linha[:, : width // 2]
        direita = linha[:, width // 2 :]
        pixels_esq = int(np.count_nonzero(esquerda))
        pixels_dir = int(np.count_nonzero(direita))
        if pixels_esq >= limite_lado and pixels_dir >= limite_lado:
            return True

    return False


def _detect_green_markers(roi_bgr, config):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([config.verde_hmin, config.verde_smin, config.verde_vmin], dtype=np.uint8)
    upper = np.array([config.verde_hmax, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    zone_y = int(roi_bgr.shape[0] * (1.0 - config.verde_zona))
    valid = []
    outside = []
    false = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w // 2, y + h // 2)
        marker = {
            "area": area,
            "bbox": (x, y, w, h),
            "center": center,
            "valid_zone": center[1] >= zone_y,
        }
        if area < config.verde_area_min:
            false.append(marker)
        elif center[1] >= zone_y:
            valid.append(marker)
        else:
            outside.append(marker)

    return {
        "mask": mask,
        "valid": valid,
        "outside": outside,
        "false": false,
        "zone_y": zone_y,
    }


def _summarize_green(green_data):
    if green_data["valid"]:
        if len(green_data["valid"]) >= 2:
            return "BECO", "verde valido x2"
        return "VALIDO", "verde valido"
    if green_data["outside"]:
        return "FORA_ZONA", "verde fora da zona"
    if green_data["false"]:
        return "FALSO", "verde falso"
    return "AUSENTE", "sem verde"


def _detect_red_markers(roi_bgr, config):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    lower1 = np.array(
        [config.vermelho_hmin1, config.vermelho_smin, config.vermelho_vmin],
        dtype=np.uint8,
    )
    upper1 = np.array([config.vermelho_hmax1, 255, 255], dtype=np.uint8)
    lower2 = np.array(
        [config.vermelho_hmin2, config.vermelho_smin, config.vermelho_vmin],
        dtype=np.uint8,
    )
    upper2 = np.array([config.vermelho_hmax2, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    zone_y = int(roi_bgr.shape[0] * (1.0 - config.vermelho_zona))
    valid = []
    outside = []
    false = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w // 2, y + h // 2)
        marker = {
            "area": area,
            "bbox": (x, y, w, h),
            "center": center,
            "valid_zone": center[1] >= zone_y,
        }
        if area < config.vermelho_area_min:
            false.append(marker)
        elif center[1] >= zone_y:
            valid.append(marker)
        else:
            outside.append(marker)

    return {
        "mask": mask,
        "valid": valid,
        "outside": outside,
        "false": false,
        "zone_y": zone_y,
    }


def _summarize_red(red_data):
    if red_data["valid"]:
        return "VALIDO", "vermelho detectado"
    if red_data["outside"]:
        return "FORA_ZONA", "vermelho fora da zona"
    if red_data["false"]:
        return "FALSO", "vermelho falso"
    return "AUSENTE", "sem vermelho"


def _classify_command(offset, intersection, green_state):
    if green_state == "BECO":
        return "U", "BECO_SEM_SAIDA"
    if intersection:
        if green_state == "VALIDO":
            if offset < -0.12:
                return "L", "INTERSECAO_VERDE"
            if offset > 0.12:
                return "R", "INTERSECAO_VERDE"
            return "F", "INTERSECAO_RETO"
        return "F", "INTERSECAO"
    if offset < -0.18:
        return "L", "CORRIGIR_ESQUERDA"
    if offset > 0.18:
        return "R", "CORRIGIR_DIREITA"
    return "F", "SEGUIR_RETO"


def analyze_frame(frame_bgr, config, state):
    now = time.monotonic()
    y0, y1, frame_width, roi_height = _roi_bounds(frame_bgr.shape, config.roi)
    roi_bgr = frame_bgr[y0:y1].copy()

    gray, line_mask, threshold_used = _build_line_mask(
        roi_bgr, config.threshold, config.invert
    )
    line = _detect_line(line_mask, config.min_line_area, frame_width)
    green = _detect_green_markers(roi_bgr, config)
    green_state, green_detail = _summarize_green(green)
    red = _detect_red_markers(roi_bgr, config)
    red_state, red_detail = _summarize_red(red)

    if line["found"]:
        state.last_seen_line_time = now
        state.line_lost_since = None
        state.last_offset = line["offset"]
    else:
        if state.line_lost_since is None:
            state.line_lost_since = now

    line_missing_for = 0.0 if state.line_lost_since is None else now - state.line_lost_since
    gap_active = line_missing_for >= config.gap_tempo
    intersection = bool(
        line["found"]
        and line["confidence"] >= config.min_confidence
        and _detect_intersection(line_mask, config)
    )
    line["intersection"] = intersection

    command = "S"
    state_name = "SEM_LINHA"
    reason = "linha ausente"

    red_detected = bool(red["valid"])
    if red_detected and state.red_armed:
        state.red_stop_until = now + config.vermelho_tempo_parado
        state.red_armed = False
    if not red_detected and now >= state.red_stop_until:
        state.red_armed = True

    red_stop_active = now < state.red_stop_until

    if red_stop_active:
        command = "S"
        state_name = "PARADO_VERMELHO"
        reason = "vermelho detectado"
    elif now < state.turn_until:
        command = "U"
        state_name = state.turn_reason or "GIRO_180"
        reason = "giro 180 em andamento"
    elif green_state == "BECO" and now >= state.dead_end_cooldown_until:
        command = "U"
        state_name = "BECO_SEM_SAIDA"
        reason = "duas marcacoes verdes validas"
        state.turn_until = now + config.giro_180_tempo
        state.turn_reason = state_name
        state.dead_end_cooldown_until = now + config.beco_cooldown
    elif gap_active:
        command = state.last_command if state.last_command in {"F", "L", "R"} else "F"
        state_name = "GAP"
        reason = f"linha perdida ha {line_missing_for:.2f}s"
    elif line["found"]:
        command, state_name = _classify_command(line["offset"], intersection, green_state)
        reason = "linha detectada"
    else:
        command = "S"
        state_name = "SEM_LINHA"

    if line["found"] and state_name not in {"BECO_SEM_SAIDA", "GAP", "PARADO_VERMELHO"}:
        state.last_command = command

    debug = frame_bgr.copy()
    cv2.rectangle(debug, (0, y0), (frame_width - 1, y1 - 1), (255, 255, 0), 2)
    frame_center_x = frame_width // 2
    cv2.line(debug, (frame_center_x, y0), (frame_center_x, y1), (255, 0, 0), 2)
    cv2.line(
        debug,
        (0, y0 + green["zone_y"]),
        (frame_width - 1, y0 + green["zone_y"]),
        (0, 255, 255),
        2,
    )
    cv2.line(
        debug,
        (0, y0 + red["zone_y"]),
        (frame_width - 1, y0 + red["zone_y"]),
        (0, 0, 255),
        2,
    )

    if line["contour"] is not None:
        contour_shifted = line["contour"] + np.array([[[0, y0]]], dtype=np.int32)
        cv2.drawContours(debug, [contour_shifted], -1, (0, 0, 255), 2)

    if line["found"]:
        center = (int(line["center_x"]), y0 + int(line["center_y"]))
        cv2.circle(debug, center, 7, (0, 165, 255), -1)
        cv2.line(debug, (frame_center_x, center[1]), center, (0, 165, 255), 2)

    for marker in green["valid"]:
        x, y, w, h = marker["bbox"]
        cv2.rectangle(debug, (x, y0 + y), (x + w, y0 + y + h), (0, 255, 0), 2)
        cv2.putText(
            debug,
            "VERDE",
            (x, max(20, y0 + y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    for marker in green["outside"]:
        x, y, w, h = marker["bbox"]
        cv2.rectangle(debug, (x, y0 + y), (x + w, y0 + y + h), (0, 255, 255), 2)

    for marker in green["false"]:
        x, y, w, h = marker["bbox"]
        cv2.rectangle(debug, (x, y0 + y), (x + w, y0 + y + h), (128, 128, 128), 1)

    for marker in red["valid"]:
        x, y, w, h = marker["bbox"]
        cv2.rectangle(debug, (x, y0 + y), (x + w, y0 + y + h), (0, 0, 255), 2)
        cv2.putText(
            debug,
            "VERMELHO",
            (x, max(20, y0 + y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    for marker in red["outside"]:
        x, y, w, h = marker["bbox"]
        cv2.rectangle(debug, (x, y0 + y), (x + w, y0 + y + h), (0, 0, 160), 1)

    for marker in red["false"]:
        x, y, w, h = marker["bbox"]
        cv2.rectangle(debug, (x, y0 + y), (x + w, y0 + y + h), (96, 96, 96), 1)

    overlay_lines = [
        f"Estado: {state_name}",
        f"Cmd logico: {command}",
        f"Offset: {line['offset']:+.3f}",
        f"Confidence: {line['confidence']:.2f}",
        f"Intersecao: {'SIM' if intersection else 'NAO'}",
        f"Verde: {green_detail}",
        f"Vermelho: {red_detail}",
        f"Threshold: {threshold_used}",
        reason,
    ]
    for idx, text in enumerate(overlay_lines):
        y = 28 + idx * 24
        cv2.putText(
            debug,
            text,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return {
        "state": state_name,
        "offset": line["offset"],
        "confidence": line["confidence"],
        "intersection": intersection,
        "visual_decision": state_name,
        "suggested_command": command,
        "green_state": green_state,
        "green_detail": green_detail,
        "red_state": red_state,
        "red_detail": red_detail,
        "red_stop_active": red_stop_active,
        "gap_active": gap_active,
        "debug_frame": debug,
        "line_found": line["found"],
        "line_missing_for": line_missing_for,
    }

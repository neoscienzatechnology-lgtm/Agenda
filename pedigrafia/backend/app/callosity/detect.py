"""Detecção assistida de hiperqueratoses / calosidades.

**Isto não é um diagnóstico.** É uma sugestão visual: regiões da planta que aparecem
localmente mais claras e menos saturadas que a pele ao redor — o padrão típico de
hiperqueratose em imagem, mas também de reflexo, de cicatriz, de pó ou de
iluminação irregular.

Cada sugestão sai com ``confidence`` e ``accepted=False``. Só entra no laudo se o
profissional confirmar. Nenhuma sugestão altera qualquer medida.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..geometry import polygon as poly


@dataclass
class CallosityHint:
    id: str
    polygon_mm: np.ndarray
    centroid_mm: np.ndarray
    area_mm2: float
    confidence: float


MIN_AREA_MM2 = 12.0
MAX_AREA_MM2 = 900.0


def detect_callosities(rect_bgr: np.ndarray, foot_mask: np.ndarray,
                       px_per_mm: float, origin_mm: np.ndarray,
                       glare_mask: np.ndarray | None = None,
                       max_hints: int = 12) -> list[CallosityHint]:
    inside = foot_mask > 0
    if int(np.count_nonzero(inside)) < 1000:
        return []

    lab = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2Lab)
    lightness = lab[:, :, 0].astype(np.float32)
    chroma = np.sqrt((lab[:, :, 1].astype(np.float32) - 128.0) ** 2
                     + (lab[:, :, 2].astype(np.float32) - 128.0) ** 2)

    # Top-hat em escala anatômica (~8 mm): destaca manchas locais, não gradientes
    # de iluminação de grande escala.
    k = max(3, int(round(8.0 * px_per_mm)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    tophat = cv2.morphologyEx(lightness, cv2.MORPH_TOPHAT, kernel)

    vals = tophat[inside]
    if vals.size < 100:
        return []
    thr = float(np.percentile(vals, 98.5))
    if thr < 4.0:
        return []

    candidate = (tophat >= thr) & inside
    # Calosidade é desaturada; reflexo especular é quase branco e estourado.
    candidate &= chroma <= float(np.percentile(chroma[inside], 45))
    candidate &= lightness < 250.0
    if glare_mask is not None:
        candidate &= (glare_mask == 0)

    mask = (candidate.astype(np.uint8)) * 255
    small = max(1, int(round(0.8 * px_per_mm)))
    kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * small + 1, 2 * small + 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kk)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kk)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8)
    area_per_px = 1.0 / (px_per_mm ** 2)

    hints: list[CallosityHint] = []
    for i in range(1, num):
        area_mm2 = float(stats[i, cv2.CC_STAT_AREA]) * area_per_px
        if not (MIN_AREA_MM2 <= area_mm2 <= MAX_AREA_MM2):
            continue
        comp = np.where(labels == i, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cpx = contours[0].reshape(-1, 2).astype(np.float64)
        if len(cpx) < 5:
            continue
        cmm = origin_mm + cpx / px_per_mm
        cmm = poly.simplify_closed(cmm, 0.4)
        contrast = float(np.mean(tophat[labels == i]))
        confidence = float(np.clip(0.20 + 0.6 * min(1.0, (contrast - thr) / 14.0)
                                   + 0.2 * min(1.0, area_mm2 / 180.0), 0.05, 0.85))
        hints.append(CallosityHint(
            id=f"cal-{i}", polygon_mm=cmm,
            centroid_mm=origin_mm + np.asarray(centroids[i]) / px_per_mm,
            area_mm2=area_mm2, confidence=confidence,
        ))

    hints.sort(key=lambda h: -h.confidence)
    return hints[:max_hints]

"""Separação de componentes: nenhum pé, um pé ou dois pés.

Cada componente vira um contorno independente em milímetros. Não há suposição sobre
posição na imagem — a classificação D/E é feita depois, por geometria anatômica.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..calibration.homography import Rectification
from ..config import get_settings
from . import contour as contour_mod
from . import polygon as poly


@dataclass
class FootComponent:
    index: int
    mask: np.ndarray               # máscara isolada deste pé (uint8)
    contour_px: np.ndarray
    contour_mm: np.ndarray
    area_mm2: float
    touches_border: bool
    centroid_mm: np.ndarray


def split_feet(mask: np.ndarray, rect: Rectification,
               score_field: np.ndarray | None = None,
               score_threshold: float | None = None) -> list[FootComponent]:
    g = get_settings().geometry
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8)

    area_per_px_mm2 = rect.area_mm2_per_px
    out: list[FootComponent] = []
    h, w = mask.shape[:2]

    order = sorted(range(1, num),
                   key=lambda i: -float(stats[i, cv2.CC_STAT_AREA]))
    for idx, i in enumerate(order[:2]):
        area_px = float(stats[i, cv2.CC_STAT_AREA])
        area_mm2 = area_px * area_per_px_mm2
        if area_mm2 < g.min_foot_area_mm2 or area_mm2 > g.max_foot_area_mm2:
            continue
        comp = np.where(labels == i, 255, 0).astype(np.uint8)
        cpx = contour_mod.largest_subpixel_contour_px(comp)
        if cpx is None or len(cpx) < 24:
            continue
        if score_field is not None and score_threshold is not None:
            # Janela em unidade física (1,2 mm): independe da resolução do raster.
            cpx = contour_mod.refine_subpixel(
                cpx, score_field, score_threshold,
                max_shift_px=1.2 * rect.px_per_mm)

        cmm = rect.rect_px_to_mm(cpx)
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        touches = bool(x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1)

        out.append(FootComponent(
            index=idx, mask=comp, contour_px=cpx, contour_mm=cmm,
            area_mm2=float(poly.area(cmm)), touches_border=touches,
            centroid_mm=rect.rect_px_to_mm(np.array([centroids[i]]))[0],
        ))

    # Ordem estável: da esquerda para a direita na imagem retificada.
    out.sort(key=lambda c: float(c.centroid_mm[0]))
    for k, comp in enumerate(out):
        comp.index = k
    return out

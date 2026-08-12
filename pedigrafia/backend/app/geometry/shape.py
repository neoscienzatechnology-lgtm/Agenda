"""Plausibilidade de forma: "isto parece um pé?".

Usado para descartar componentes espúrios (moldura do podoscópio, sombras, piso,
bordas da plataforma) sem depender da posição na imagem nem de um modelo treinado.
Todas as medidas são adimensionais ou em milímetros — nunca em pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import polygon as poly


@dataclass
class ShapeStats:
    area_mm2: float
    length_mm: float
    width_mm: float
    elongation: float
    box_fill: float
    solidity: float
    likeness: float


def _plateau(value: float, lo: float, lo_ok: float, hi_ok: float, hi: float) -> float:
    """1,0 dentro de [lo_ok, hi_ok], caindo linearmente até 0 em lo/hi."""
    if value <= lo or value >= hi:
        return 0.0
    if value < lo_ok:
        return (value - lo) / max(lo_ok - lo, 1e-9)
    if value > hi_ok:
        return (hi - value) / max(hi - hi_ok, 1e-9)
    return 1.0


def shape_stats(contour_mm: np.ndarray) -> ShapeStats:
    pts = poly.as_points(contour_mm)
    area = poly.area(pts)
    rect = cv2.minAreaRect(pts.astype(np.float32))
    (w, h) = rect[1]
    length = float(max(w, h))
    width = float(min(w, h))
    elongation = length / width if width > 1e-6 else 0.0
    box_area = length * width
    box_fill = area / box_area if box_area > 1e-6 else 0.0
    hull = poly.convex_hull(pts)
    hull_area = poly.area(hull)
    solidity = area / hull_area if hull_area > 1e-6 else 0.0

    # Faixas derivadas da antropometria do pé em projeção plantar:
    #   alongamento  ~2,3–3,3   (comprimento/largura da caixa orientada)
    #   preenchimento ~0,64–0,84 (a planta não preenche o retângulo: há arco e dedos)
    #   solidez      ~0,82–0,96 (entalhes interdigitais e concavidade do arco)
    likeness = (
        _plateau(elongation, 1.55, 2.10, 3.60, 4.80)
        * _plateau(box_fill, 0.42, 0.58, 0.88, 0.97)
        * _plateau(solidity, 0.62, 0.78, 0.99, 1.001)
    )
    return ShapeStats(area_mm2=float(area), length_mm=length, width_mm=width,
                      elongation=float(elongation), box_fill=float(box_fill),
                      solidity=float(solidity), likeness=float(likeness))

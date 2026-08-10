"""Zonas visuais de contato/apoio.

**Nomenclatura obrigatória.** Sem sensor de pressão não existe mapa de pressão. O que
o sistema produz são *regiões visuais de contato/apoio* — recortes anatômicos da
projeção plantar, opcionalmente refinados pela região de contato observável no vidro
do podoscópio. Nunca são expressos em kPa nem chamados de "pressão".

Todas as zonas são polígonos em mm, editáveis pelo profissional.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import polygon as poly
from .frame import FootFrame


@dataclass
class SupportZone:
    id: str
    label: str
    kind: str
    polygon_mm: np.ndarray
    confidence: float
    color: str


_BANDS = [
    ("heel", "Calcâneo", 0.00, 0.26, "#F97316"),
    ("midfoot", "Mediopé", 0.26, 0.55, "#38BDF8"),
    ("forefoot", "Antepé", 0.55, 0.80, "#FB923C"),
]


def _clip_side(uv_poly: np.ndarray, medial_sign: int, keep_medial: bool) -> np.ndarray:
    """Mantém metade medial ou lateral em relação a v = 0."""
    normal = np.array([0.0, -medial_sign if keep_medial else medial_sign])
    return poly.clip_half_plane(uv_poly, normal, 0.0)


def build_support_zones(contour_mm: np.ndarray, frame: FootFrame, medial_sign: int,
                        toe_cut_t: float = 0.78,
                        contact_confidence: float = 0.0) -> list[SupportZone]:
    uv = frame.to_local(poly.resample_closed(contour_mm, 0.5))
    L = frame.length_mm
    zones: list[SupportZone] = []

    base_conf = 0.55 if contact_confidence <= 0 else min(0.9, 0.5 + contact_confidence)

    for kind, label, t0, t1, color in _BANDS:
        band = poly.clip_band_u(uv, t0 * L, min(t1, toe_cut_t) * L)
        if len(band) < 3:
            continue
        zones.append(SupportZone(
            id=f"zone-{kind}", label=label, kind=kind,
            polygon_mm=frame.to_plane(band), confidence=base_conf, color=color,
        ))

    # Primeiro raio / hálux: metade medial do antepé somada ao lobo do hálux.
    fore = poly.clip_band_u(uv, 0.60 * L, 1.0 * L)
    if len(fore) >= 3:
        first_ray = _clip_side(fore, medial_sign, keep_medial=True)
        if len(first_ray) >= 3:
            zones.append(SupportZone(
                id="zone-first-ray", label="1º raio / hálux", kind="first_ray",
                polygon_mm=frame.to_plane(first_ray),
                confidence=max(0.4, base_conf - 0.1), color="#F59E0B",
            ))
    return zones


def detect_contact_region(rect_bgr: np.ndarray, foot_mask: np.ndarray,
                          px_per_mm: float) -> tuple[np.ndarray, float]:
    """Tenta isolar a região de contato (isquemia/branqueamento no vidro).

    Devolve ``(máscara, confiança)``. Confiança baixa significa que a foto **não**
    permite distinguir contato — e nesse caso o chamador deve informar isso
    explicitamente ao usuário em vez de fingir um mapa de apoio.
    """
    inside = foot_mask > 0
    if int(np.count_nonzero(inside)) < 500:
        return np.zeros_like(foot_mask), 0.0

    lab = cv2.cvtColor(cv2.GaussianBlur(rect_bgr, (0, 0), max(0.8, 0.25 * px_per_mm)),
                       cv2.COLOR_BGR2Lab)
    lightness = lab[:, :, 0].astype(np.float32)
    chroma = np.sqrt((lab[:, :, 1].astype(np.float32) - 128.0) ** 2
                     + (lab[:, :, 2].astype(np.float32) - 128.0) ** 2)

    vals_l = lightness[inside]
    vals_c = chroma[inside]
    # Contato = mais claro E menos saturado que o resto da planta (isquemia).
    l_thr = float(np.percentile(vals_l, 68))
    c_thr = float(np.percentile(vals_c, 32))
    contact = ((lightness >= l_thr) & (chroma <= c_thr) & inside)

    frac = float(np.count_nonzero(contact)) / float(np.count_nonzero(inside))
    l_spread = float(np.percentile(vals_l, 90) - np.percentile(vals_l, 10))
    c_spread = float(np.percentile(vals_c, 90) - np.percentile(vals_c, 10))

    # Só há informação de contato se a planta apresentar contraste interno real.
    confidence = float(np.clip((l_spread - 12.0) / 30.0, 0.0, 1.0)) \
        * float(np.clip((c_spread - 4.0) / 14.0, 0.0, 1.0))
    if not (0.12 <= frac <= 0.82):
        confidence *= 0.3

    mask = (contact.astype(np.uint8)) * 255
    k = max(1, int(round(1.2 * px_per_mm)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask, float(np.clip(confidence, 0.0, 1.0))

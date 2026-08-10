"""Estimativa das cabeças metatarsais M1–M5.

**Aviso metrológico e clínico incorporado ao código:** cabeças metatarsais NÃO são
visíveis em uma fotografia plantar. M1 e M5 são inferidos das proeminências reais do
contorno (as larguras máximas medial e lateral do antepé), o que é uma observação
geométrica legítima da silhueta. M2, M3 e M4 são **interpolados por modelo** e nunca
devem ser apresentados como observados.

Por isso cada ponto carrega ``method`` e ``confidence``, e a revisão manual é
obrigatória antes de qualquer exportação.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import get_settings
from ..geometry import polygon as poly
from ..geometry.frame import FootFrame


@dataclass
class MetatarsalHead:
    index: int          # 0 = M1 … 4 = M5
    point_mm: np.ndarray
    confidence: float
    method: str


def _border_extreme(uv: np.ndarray, frame: FootFrame, band: tuple[float, float],
                    side_sign: int) -> tuple[np.ndarray, float] | None:
    """Ponto de maior afastamento lateral (``side_sign``·v) dentro da faixa ``band``."""
    u_lo = band[0] * frame.length_mm
    u_hi = band[1] * frame.length_mm
    sel = (uv[:, 0] >= u_lo) & (uv[:, 0] <= u_hi)
    if int(np.count_nonzero(sel)) < 3:
        return None
    idx = np.nonzero(sel)[0]
    signed_v = uv[idx, 1] * side_sign
    best = idx[int(np.argmax(signed_v))]
    # Proeminência: quanto o extremo se destaca da média da faixa.
    prominence = float(np.max(signed_v) - np.median(signed_v))
    return uv[best].copy(), prominence


def estimate_metatarsal_heads(contour_mm: np.ndarray, frame: FootFrame,
                              medial_sign: int) -> tuple[list[MetatarsalHead], float]:
    """Devolve ``(M1..M5, confiança_agregada)`` em coordenadas do plano (mm)."""
    g = get_settings().geometry
    dense = poly.resample_closed(contour_mm, g.contour_resample_step_mm)
    uv = frame.to_local(dense)

    m1 = _border_extreme(uv, frame, g.m1_band, medial_sign)
    m5 = _border_extreme(uv, frame, g.m5_band, -medial_sign)
    if m1 is None or m5 is None:
        return [], 0.0

    m1_uv, m1_prom = m1
    m5_uv, m5_prom = m5

    width = abs(float(m1_uv[1] - m5_uv[1]))
    if width < 1e-6:
        return [], 0.0

    # Modelo do arco metatarsal: as cabeças formam uma curva de convexidade distal,
    # com o ápice próximo a M2. Parametrizada em s ∈ [0, 1] de M1 até M5.
    # `bulge` e `skew` são o modelo — explicitados aqui, não escondidos.
    bulge_mm = 0.030 * frame.length_mm
    skew = 0.30           # posição do ápice da protuberância (s ≈ 0,30 ≈ M2)
    s_positions = np.array([0.0, 0.235, 0.470, 0.715, 1.0])

    heads: list[MetatarsalHead] = []
    for k, s in enumerate(s_positions):
        u = float(m1_uv[0] + s * (m5_uv[0] - m1_uv[0]))
        v = float(m1_uv[1] + s * (m5_uv[1] - m1_uv[1]))
        if 0.0 < s < 1.0:
            # Sino assimétrico, nulo nas extremidades (M1 e M5 permanecem observados).
            t = s / skew if s < skew else (1.0 - s) / (1.0 - skew)
            u += bulge_mm * float(np.sin(np.pi * 0.5 * np.clip(t, 0.0, 1.0)) ** 2)
        point = frame.to_plane(np.array([[u, v]]))[0]
        if k == 0:
            conf = float(np.clip(0.35 + 0.65 * min(1.0, m1_prom / (0.06 * frame.length_mm)), 0, 1))
            method = "contour_extreme_medial"
        elif k == 4:
            conf = float(np.clip(0.35 + 0.65 * min(1.0, m5_prom / (0.06 * frame.length_mm)), 0, 1))
            method = "contour_extreme_lateral"
        else:
            conf = 0.40
            method = "interpolated_model"
        heads.append(MetatarsalHead(index=k, point_mm=point, confidence=conf,
                                    method=method))

    aggregate = float(np.mean([h.confidence for h in heads]))
    return heads, aggregate


def metatarsal_line(heads: list[MetatarsalHead]) -> list[np.ndarray]:
    """Segmento M1–M5 — a "linha metatarsal" exibida e usada nas medidas."""
    if len(heads) < 5:
        return []
    return [heads[0].point_mm.copy(), heads[4].point_mm.copy()]

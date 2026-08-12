"""Curvas do arco medial e do arco lateral em **projeção plantar 2D**.

Terminologia: estas curvas descrevem o traçado das bordas medial e lateral entre o
retropé e o antepé. Elas NÃO representam a altura real do arco plantar — isso exige
medição tridimensional ou baropodometria.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import polygon as poly
from .frame import FootFrame


@dataclass
class ArchCurve:
    id: str
    points_mm: np.ndarray
    control_points_mm: np.ndarray
    confidence: float


def _border_points(uv: np.ndarray, frame: FootFrame, side_sign: int,
                   t_lo: float, t_hi: float, samples: int) -> np.ndarray:
    """Borda (máximo de ``side_sign``·v) amostrada ao longo de t."""
    ts = np.linspace(t_lo, t_hi, samples)
    out = []
    for t in ts:
        u = float(t) * frame.length_mm
        vs = poly.crossings_at_u(uv, u)
        if vs.size < 2:
            continue
        v = float(np.max(vs * side_sign)) * side_sign
        out.append([u, v])
    return np.array(out, dtype=np.float64)


def _fit_control_points(points_uv: np.ndarray, n: int = 5) -> np.ndarray:
    """Pontos de controle igualmente espaçados ao longo do arco (edição por spline)."""
    if len(points_uv) < 2:
        return points_uv
    d = np.linalg.norm(np.diff(points_uv, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(d)])
    total = float(cum[-1])
    if total <= 1e-9:
        return points_uv[:1]
    targets = np.linspace(0.0, total, n)
    idx = np.clip(np.searchsorted(cum, targets, side="right") - 1, 0, len(d) - 1)
    seg = np.where(d[idx] > 1e-12, d[idx], 1.0)
    frac = ((targets - cum[idx]) / seg)[:, None]
    return points_uv[idx] + frac * (points_uv[idx + 1] - points_uv[idx])


def build_arches(contour_mm: np.ndarray, frame: FootFrame, medial_sign: int,
                 confidence: float = 0.6) -> tuple[ArchCurve, ArchCurve]:
    uv = frame.to_local(poly.resample_closed(contour_mm, 0.5))

    medial_uv = _border_points(uv, frame, medial_sign, 0.12, 0.80, 48)
    lateral_uv = _border_points(uv, frame, -medial_sign, 0.12, 0.80, 48)

    medial = ArchCurve(
        id="medial",
        points_mm=frame.to_plane(medial_uv) if len(medial_uv) else np.empty((0, 2)),
        control_points_mm=(frame.to_plane(_fit_control_points(medial_uv))
                           if len(medial_uv) else np.empty((0, 2))),
        confidence=confidence,
    )
    lateral = ArchCurve(
        id="lateral",
        points_mm=frame.to_plane(lateral_uv) if len(lateral_uv) else np.empty((0, 2)),
        control_points_mm=(frame.to_plane(_fit_control_points(lateral_uv))
                           if len(lateral_uv) else np.empty((0, 2))),
        confidence=confidence,
    )
    return medial, lateral

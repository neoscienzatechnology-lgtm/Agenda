"""Frame anatômico local do pé (origem no calcâneo, eixo u → dedos).

Duas passagens:
  1. *bootstrap* pelo eixo principal de área + decisão de qual extremo são os dedos;
  2. refino usando o centro do calcâneo e o ápice do 2º pododáctilo (eixo anatômico).

O eixo resultante é editável pelo profissional; toda medida derivada é recalculada.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from ..config import get_settings
from . import polygon as poly


@dataclass
class FootFrame:
    origin: np.ndarray   # (2,) mm — ponto mais posterior do calcâneo sobre o eixo
    u: np.ndarray        # (2,) unitário, calcâneo → dedos
    v: np.ndarray        # (2,) unitário, perpendicular (rot +90°)
    length_mm: float
    medial_sign: int = 1

    @property
    def orientation_deg(self) -> float:
        return float(np.degrees(np.arctan2(self.u[1], self.u[0])))

    def to_local(self, pts) -> np.ndarray:
        return poly.project(pts, self.origin, self.u, self.v)

    def to_plane(self, uv) -> np.ndarray:
        return poly.unproject(uv, self.origin, self.u, self.v)

    def t_of(self, u_mm: float) -> float:
        return u_mm / self.length_mm if self.length_mm > 0 else 0.0

    def u_of_t(self, t: float) -> float:
        return t * self.length_mm


def _rot90(u: np.ndarray) -> np.ndarray:
    return np.array([-u[1], u[0]], dtype=np.float64)


def _frame_from_direction(contour_mm: np.ndarray, u: np.ndarray) -> FootFrame:
    u = np.asarray(u, dtype=np.float64)
    n = float(np.linalg.norm(u))
    u = u / n if n > 1e-12 else np.array([0.0, 1.0])
    v = _rot90(u)
    proj_u = contour_mm @ u
    u_min = float(np.min(proj_u))
    u_max = float(np.max(proj_u))
    length = u_max - u_min
    # Origem: ponto do eixo na altura do extremo posterior, alinhado ao "centro" em v.
    heel_idx = int(np.argmin(proj_u))
    heel_point = contour_mm[heel_idx]
    origin = heel_point - (float(heel_point @ u) - u_min) * u
    # Alinha v = 0 ao centro da secção do calcâneo, tornando o frame independente
    # da posição absoluta do contorno no plano.
    uv_tmp = poly.project(contour_mm, origin, u, v)
    band = poly.clip_band_u(uv_tmp, 0.0, max(1e-3, 0.15 * length))
    if len(band) >= 3:
        v_center = float(poly.centroid(band)[1])
    else:
        v_center = float(np.median(uv_tmp[:, 1]))
    origin = origin + v_center * v
    return FootFrame(origin=origin, u=u, v=v, length_mm=length)


def _toe_peak_count(contour_mm: np.ndarray, u: np.ndarray, length: float) -> int:
    """Quantos ápices prominentes existem no extremo apontado por ``u``."""
    settings = get_settings().geometry
    proj = contour_mm @ u
    top = float(np.max(proj))
    band = top - 0.26 * length
    seq = np.where(proj >= band, proj, band)
    # Duplica o sinal (contorno é circular) e procura picos no trecho central.
    doubled = np.concatenate([seq, seq])
    peaks, _ = find_peaks(
        doubled,
        prominence=settings.toe_peak_min_prominence_mm,
        distance=max(2, int(settings.toe_peak_min_separation_mm
                            / max(settings.contour_resample_step_mm, 1e-6))),
    )
    peaks = peaks[(peaks >= len(seq) // 2) & (peaks < len(seq) // 2 + len(seq))]
    return int(len(peaks))


def bootstrap_frame(contour_mm: np.ndarray) -> tuple[FootFrame, float]:
    """Eixo inicial por momentos de área + orientação calcâneo/dedos.

    Devolve ``(frame, confiança_da_orientação)``.
    """
    contour_mm = poly.as_points(contour_mm)
    u = poly.principal_axis(contour_mm)
    frame_a = _frame_from_direction(contour_mm, u)
    frame_b = _frame_from_direction(contour_mm, -u)

    peaks_a = _toe_peak_count(contour_mm, frame_a.u, frame_a.length_mm)
    peaks_b = _toe_peak_count(contour_mm, frame_b.u, frame_b.length_mm)

    # Pista secundária: o centroide de área fica mais próximo do calcâneo (t < 0.5).
    c = poly.centroid(contour_mm)
    t_a = frame_a.t_of(float((c - frame_a.origin) @ frame_a.u))
    t_b = frame_b.t_of(float((c - frame_b.origin) @ frame_b.u))

    score_a = peaks_a + (0.6 if t_a < 0.5 else 0.0)
    score_b = peaks_b + (0.6 if t_b < 0.5 else 0.0)

    if score_a >= score_b:
        frame, other = frame_a, score_b
        best = score_a
    else:
        frame, other = frame_b, score_a
        best = score_b

    total = best + other
    confidence = float(np.clip((best - other) / total, 0.0, 1.0)) if total > 0 else 0.0
    return frame, confidence


def refine_frame(contour_mm: np.ndarray, frame: FootFrame,
                 second_toe_mm: np.ndarray | None) -> FootFrame:
    """Eixo anatômico: centro do calcâneo → ápice do 2º pododáctilo.

    Se o 2º pododáctilo não foi identificado, mantém o eixo de *bootstrap*.
    """
    if second_toe_mm is None:
        return frame
    uv = frame.to_local(contour_mm)
    heel_band = poly.clip_band_u(uv, 0.0, max(1e-3, 0.12 * frame.length_mm))
    if len(heel_band) < 3:
        return frame
    heel_center = frame.to_plane(poly.centroid(heel_band)[None, :])[0]
    direction = np.asarray(second_toe_mm, dtype=np.float64) - heel_center
    if float(np.linalg.norm(direction)) < 1e-6:
        return frame
    # Rejeita correções absurdas (> 25°): provável erro na detecção do 2º dedo.
    cosang = float(np.clip(np.dot(direction / np.linalg.norm(direction), frame.u), -1, 1))
    if np.degrees(np.arccos(cosang)) > 25.0:
        return frame
    refined = _frame_from_direction(contour_mm, direction)
    refined.medial_sign = frame.medial_sign
    return refined


def frame_from_axis_points(contour_mm: np.ndarray, a_mm, b_mm,
                           medial_sign: int = 1) -> FootFrame:
    """Frame a partir de um eixo definido explicitamente (eixo editado pelo usuário)."""
    a = np.asarray(a_mm, dtype=np.float64)
    b = np.asarray(b_mm, dtype=np.float64)
    d = b - a
    if float(np.linalg.norm(d)) < 1e-9:
        frame, _ = bootstrap_frame(contour_mm)
        frame.medial_sign = medial_sign
        return frame
    frame = _frame_from_direction(poly.as_points(contour_mm), d)
    frame.medial_sign = medial_sign
    return frame


def heel_center(contour_mm: np.ndarray, frame: FootFrame, band_t: float = 0.12) -> np.ndarray:
    uv = frame.to_local(contour_mm)
    band = poly.clip_band_u(uv, 0.0, max(1e-3, band_t * frame.length_mm))
    if len(band) < 3:
        return frame.origin.copy()
    return frame.to_plane(poly.centroid(band)[None, :])[0]


def extreme_point(contour_mm: np.ndarray, frame: FootFrame, *, distal: bool) -> np.ndarray:
    proj = poly.as_points(contour_mm) @ frame.u
    idx = int(np.argmax(proj)) if distal else int(np.argmin(proj))
    return poly.as_points(contour_mm)[idx]

"""Ajuste da homografia a partir de **todos** os pontos de controle disponíveis.

Com um marcador há exatamente 4 correspondências: a homografia passa por elas de
forma exata e o resíduo é sempre zero — ou seja, **não há como medir a qualidade da
calibração**. Com dois ou mais marcadores o sistema fica sobredeterminado: o resíduo
passa a ser um sinal real de erro, em milímetros, e é reportado.

Mais importante que o resíduo é a **cobertura**: dentro do casco convexo dos pontos
de controle a homografia interpola; fora dele, extrapola, e o erro cresce com a
distância. :func:`extrapolation_distance_mm` quantifica isso para que o *quality
gate* possa avisar antes de o profissional confiar em uma medida ruim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..geometry import polygon as poly
from .marker import MarkerDetection
from .target import CalibrationTarget


@dataclass
class HomographyFit:
    homography_image_to_mm: np.ndarray
    control_points_px: np.ndarray
    control_points_mm: np.ndarray
    used_marker_ids: tuple[int, ...]
    residual_rms_mm: float
    residual_max_mm: float
    target_name: str
    exact: bool
    """``True`` quando há apenas 4 pontos: o ajuste é exato e o resíduo não informa nada."""
    warnings: list[str] = field(default_factory=list)

    @property
    def marker_count(self) -> int:
        return len(self.used_marker_ids)

    def control_hull_mm(self) -> np.ndarray:
        if len(self.control_points_mm) < 3:
            return self.control_points_mm
        return poly.convex_hull(self.control_points_mm)

    def coverage_span_mm(self) -> tuple[float, float]:
        pts = self.control_points_mm
        if len(pts) == 0:
            return (0.0, 0.0)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        return (float(hi[0] - lo[0]), float(hi[1] - lo[1]))


class CalibrationFitError(Exception):
    pass


def fit_homography(detection: MarkerDetection,
                   target: CalibrationTarget) -> HomographyFit:
    """Correspondências marcador↔plataforma → homografia imagem(px) → plano(mm)."""
    if not detection.found:
        raise CalibrationFitError("marcador não detectado")

    src: list[np.ndarray] = []
    dst: list[np.ndarray] = []
    used: list[int] = []
    warnings: list[str] = []

    for marker in detection.markers:
        placement = target.placement(marker.marker_id)
        if placement is None:
            continue
        if marker.touches_border:
            warnings.append(
                f"Marcador {marker.marker_id} encosta na borda da foto e foi "
                f"descartado da calibração.")
            continue
        src.append(np.asarray(marker.corners_px, dtype=np.float64))
        dst.append(placement.corners_mm())
        used.append(marker.marker_id)

    if not src:
        raise CalibrationFitError(
            "nenhum marcador do alvo de calibração foi reconhecido na foto")

    src_pts = np.vstack(src)
    dst_pts = np.vstack(dst)

    if len(src_pts) == 4:
        H = cv2.getPerspectiveTransform(src_pts.astype(np.float32),
                                        dst_pts.astype(np.float32)).astype(np.float64)
        exact = True
    else:
        # Mínimos quadrados sobre todos os cantos. Sem RANSAC: os pontos vêm de um
        # detector que já validou o código do marcador, então descartar pontos aqui
        # esconderia erro em vez de medi-lo.
        H, _ = cv2.findHomography(src_pts, dst_pts, method=0)
        if H is None:
            raise CalibrationFitError("ajuste da homografia não convergiu")
        H = H.astype(np.float64)
        exact = False

    projected = cv2.perspectiveTransform(
        src_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    residuals = np.linalg.norm(projected - dst_pts, axis=1)
    rms = float(np.sqrt(np.mean(residuals ** 2)))
    worst = float(np.max(residuals))

    if exact:
        warnings.append(
            "Calibração com um único marcador: a escala é exata sobre ele, mas "
            "extrapolada para o resto da plataforma. Use um alvo com quatro "
            "marcadores para medir os pés por interpolação.")

    return HomographyFit(
        homography_image_to_mm=H,
        control_points_px=src_pts,
        control_points_mm=dst_pts,
        used_marker_ids=tuple(used),
        residual_rms_mm=rms,
        residual_max_mm=worst,
        target_name=target.name,
        exact=exact,
        warnings=warnings,
    )


def extrapolation_distance_mm(fit: HomographyFit, points_mm: np.ndarray) -> float:
    """Maior distância de ``points_mm`` até o casco dos pontos de controle.

    Zero significa que tudo o que foi medido está **dentro** da região coberta pela
    calibração. Valores positivos são a distância em que a homografia está
    extrapolando — a grandeza que prevê o erro de medida.
    """
    pts = poly.as_points(points_mm)
    hull = fit.control_hull_mm()
    if len(pts) == 0 or len(hull) < 3:
        return float("inf")

    a = hull
    b = np.roll(hull, -1, axis=0)
    ab = b - a
    denom = np.sum(ab * ab, axis=1)
    denom = np.where(denom < 1e-12, 1.0, denom)

    worst = 0.0
    for p in pts:
        # Dentro do casco convexo a distância é zero por definição.
        inside = True
        for i in range(len(hull)):
            if np.cross(ab[i], p - a[i]) < -1e-9:
                inside = False
                break
        if inside:
            continue
        t = np.clip(np.sum((p - a) * ab, axis=1) / denom, 0.0, 1.0)
        proj = a + t[:, None] * ab
        worst = max(worst, float(np.min(np.linalg.norm(proj - p, axis=1))))
    return worst

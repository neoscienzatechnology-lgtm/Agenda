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


def fit_free_rectangles(detection: MarkerDetection, model_mm: np.ndarray
                        ) -> tuple[HomographyFit, list[tuple[float, float, float]]]:
    """Homografia a partir de N retângulos **de pose desconhecida** e forma conhecida.

    É o caso do objeto de referência: sabemos que cada objeto mede exatamente
    ``model_mm``, mas não onde ele foi largado sobre a plataforma nem em que ângulo.

    Com **um** retângulo o problema é exatamente determinado (4 pontos, 8 graus de
    liberdade) e vale a mesma ressalva do marcador único: resíduo zero por
    construção, exatidão que decai com a distância.

    Com **dois ou mais** vale a pena resolver junto: as incógnitas passam a ser a
    homografia (8) mais a pose no plano de cada objeto extra (3 cada), contra 8
    resíduos por objeto — sobra informação. O segundo objeto, colocado do outro lado
    dos pés, estende o casco dos pontos de controle e transforma extrapolação em
    interpolação, que é exatamente a correção que o alvo de quatro marcadores faz
    sem exigir impressão.

    O referencial em mm é o do primeiro objeto (origem no seu canto TL), o que fixa
    a liberdade global de similaridade.

    Devolve ``(ajuste, poses)``, com ``poses`` = ``(x_mm, y_mm, rotação_graus)`` de
    cada objeto no referencial resolvido.
    """
    if not detection.found or not detection.markers:
        raise CalibrationFitError("nenhum objeto de referência detectado")

    model = np.asarray(model_mm, dtype=np.float64).reshape(4, 2)
    quads = [np.asarray(m.corners_px, dtype=np.float64).reshape(4, 2)
             for m in detection.markers]
    ids = tuple(int(m.marker_id) for m in detection.markers)
    warnings: list[str] = []

    H0 = cv2.getPerspectiveTransform(quads[0].astype(np.float32),
                                     model.astype(np.float32)).astype(np.float64)
    poses: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    for quad in quads[1:]:
        mapped = cv2.perspectiveTransform(quad.reshape(-1, 1, 2), H0).reshape(-1, 2)
        poses.append(_rigid_from_model(model, mapped))

    if len(quads) == 1:
        H = H0
        exact = True
        warnings.append(
            "Calibração com um único objeto de referência: a escala é exata sobre "
            "ele e extrapolada para o resto da plataforma. Objetos iguais adicionais "
            "reduzem a extrapolação; com quatro, ao redor da área de apoio, ela "
            "desaparece — dois apenas cobrem uma faixa, não uma área.")
    else:
        H, poses = _solve_joint(quads, model, H0, poses)
        exact = False

    src_pts = np.vstack(quads)
    dst_pts = np.vstack([_place(model, p) for p in poses])
    projected = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    residuals = np.linalg.norm(projected - dst_pts, axis=1)

    fit = HomographyFit(
        homography_image_to_mm=H,
        control_points_px=src_pts,
        control_points_mm=dst_pts,
        used_marker_ids=ids,
        residual_rms_mm=float(np.sqrt(np.mean(residuals ** 2))),
        residual_max_mm=float(np.max(residuals)),
        target_name="reference",
        exact=exact,
        warnings=warnings,
    )
    return fit, poses


def _place(model: np.ndarray, pose: tuple[float, float, float]) -> np.ndarray:
    x, y, deg = pose
    a = np.radians(deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return model @ R.T + np.array([x, y], dtype=np.float64)


def _rigid_from_model(model: np.ndarray, observed: np.ndarray
                      ) -> tuple[float, float, float]:
    """Procrustes sem escala: rotação + translação que levam ``model`` a ``observed``."""
    mc, oc = model.mean(axis=0), observed.mean(axis=0)
    A = (model - mc).T @ (observed - oc)
    U, _, Vt = np.linalg.svd(A)
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:                      # nunca espelha o padrão físico
        Vt = Vt.copy()
        Vt[-1] *= -1.0
        R = (U @ Vt).T
    angle = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    t = oc - R @ mc
    return (float(t[0]), float(t[1]), angle)


def _solve_joint(quads: list[np.ndarray], model: np.ndarray, H0: np.ndarray,
                 poses0: list[tuple[float, float, float]]
                 ) -> tuple[np.ndarray, list[tuple[float, float, float]]]:
    """Mínimos quadrados não linear sobre (homografia, poses dos objetos extras)."""
    from scipy.optimize import least_squares

    h0 = (H0 / H0[2, 2]).ravel()[:8]
    extra0 = np.array([v for pose in poses0[1:] for v in pose], dtype=np.float64)
    x0 = np.concatenate([h0, extra0])
    n_extra = len(quads) - 1

    def unpack(x):
        H = np.append(x[:8], 1.0).reshape(3, 3)
        poses = [(0.0, 0.0, 0.0)]
        for k in range(n_extra):
            px, py, pa = x[8 + 3 * k: 11 + 3 * k]
            poses.append((float(px), float(py), float(pa)))
        return H, poses

    def residual(x):
        H, poses = unpack(x)
        out = []
        for quad, pose in zip(quads, poses):
            hom = np.concatenate([quad, np.ones((4, 1))], axis=1) @ H.T
            w = hom[:, 2]
            if np.any(np.abs(w) < 1e-9):
                return np.full(8 * len(quads), 1e6)
            out.append((hom[:, :2] / w[:, None]) - _place(model, pose))
        return np.concatenate(out).ravel()

    try:
        sol = least_squares(residual, x0, method="lm", max_nfev=4000)
    except (ValueError, np.linalg.LinAlgError):
        return H0, poses0
    H, poses = unpack(sol.x)
    if not np.isfinite(H).all():
        return H0, poses0
    return H, poses


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

"""Homografia, retificação e a transformação **pixel → milímetro**.

Esta é a única fonte de escala física do sistema. Todo o restante do software consome
``Rectification.rect_px_to_mm`` — nenhuma outra divisão por ``px_per_mm`` é permitida.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import get_settings
from .marker import MarkerDetection


class CalibrationError(Exception):
    pass


@dataclass
class Rectification:
    """Mapa afim exato entre o raster retificado e o plano físico em mm."""

    homography_image_to_mm: np.ndarray  # 3×3, imagem(px) → plano(mm)
    px_per_mm: float
    origin_mm: np.ndarray               # (2,) canto sup. esq. da janela retificada, em mm
    width_px: int
    height_px: int
    image: np.ndarray                   # raster retificado (BGR)
    warp_image_to_rect: np.ndarray      # 3×3, imagem(px) → retificado(px)
    valid_mask: np.ndarray              # uint8 255 onde havia pixel de origem real
    src_px_per_mm: float = 0.0
    downscaled: bool = False

    # ------------------------------------------------------------------ conversões
    def rect_px_to_mm(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64)
        return self.origin_mm + pts / self.px_per_mm

    def mm_to_rect_px(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64)
        return (pts - self.origin_mm) * self.px_per_mm

    def image_px_to_mm(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, self.homography_image_to_mm)
        return out.reshape(-1, 2)

    @property
    def mm_per_px(self) -> float:
        return 1.0 / self.px_per_mm

    @property
    def area_mm2_per_px(self) -> float:
        return 1.0 / (self.px_per_mm ** 2)

    @property
    def extent_mm(self) -> tuple[float, float, float, float]:
        return (
            float(self.origin_mm[0]),
            float(self.origin_mm[1]),
            float(self.origin_mm[0] + self.width_px / self.px_per_mm),
            float(self.origin_mm[1] + self.height_px / self.px_per_mm),
        )


def marker_model_mm(size_mm: float) -> np.ndarray:
    """Modelo físico do marcador: TL, TR, BR, BL — quadrado exato de ``size_mm``."""
    return np.array(
        [[0.0, 0.0], [size_mm, 0.0], [size_mm, size_mm], [0.0, size_mm]],
        dtype=np.float64,
    )


def compute_homography(detection: MarkerDetection, size_mm: float | None = None
                       ) -> tuple[np.ndarray, float]:
    """H (imagem px → plano mm) e o erro de reprojeção nos 4 cantos."""
    if not detection.found:
        raise CalibrationError("marcador não detectado")
    size_mm = size_mm if size_mm is not None else get_settings().marker_size_mm
    src = detection.corners_px.astype(np.float32)
    dst = marker_model_mm(size_mm).astype(np.float32)
    H = cv2.getPerspectiveTransform(src, dst).astype(np.float64)

    back = cv2.perspectiveTransform(src.reshape(-1, 1, 2).astype(np.float64),
                                    H).reshape(-1, 2)
    residual_mm = float(np.max(np.linalg.norm(back - dst.astype(np.float64), axis=1)))
    # getPerspectiveTransform é exato nos 4 pontos; o resíduo mede estabilidade numérica.
    scale = detection.src_px_per_mm if detection.src_px_per_mm > 0 else 1.0
    return H, residual_mm * scale


def _project_image_bounds_mm(H: np.ndarray, width: int, height: int,
                             samples_per_edge: int = 24) -> np.ndarray | None:
    """Projeta a borda da imagem no plano; descarta pontos além do horizonte."""
    xs = np.linspace(0, width - 1, samples_per_edge)
    ys = np.linspace(0, height - 1, samples_per_edge)
    border = np.concatenate([
        np.stack([xs, np.zeros_like(xs)], axis=1),
        np.stack([xs, np.full_like(xs, height - 1)], axis=1),
        np.stack([np.zeros_like(ys), ys], axis=1),
        np.stack([np.full_like(ys, width - 1), ys], axis=1),
    ]).astype(np.float64)

    homo = np.concatenate([border, np.ones((border.shape[0], 1))], axis=1)
    proj = homo @ H.T
    w = proj[:, 2]
    # Pontos com w ~ 0 estão na linha do horizonte: projetam para o infinito.
    valid = np.abs(w) > 1e-8
    if not np.any(valid):
        return None
    # Mantém apenas o semiplano dominante (mesmo sinal de w da maioria).
    sign = np.sign(w[valid])
    dominant = 1.0 if float(np.sum(sign > 0)) >= float(np.sum(sign < 0)) else -1.0
    valid &= (np.sign(w) == dominant)
    if np.count_nonzero(valid) < 4:
        return None
    pts = proj[valid, :2] / w[valid, None]
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    return pts if len(pts) >= 4 else None


def build_rectification(bgr: np.ndarray, detection: MarkerDetection,
                        H: np.ndarray | None = None) -> Rectification:
    """Retifica a imagem para uma vista plantar ortogonal com escala mm conhecida."""
    settings = get_settings()
    if H is None:
        H, _ = compute_homography(detection)

    height, width = bgr.shape[:2]
    size_mm = settings.marker_size_mm

    projected = _project_image_bounds_mm(H, width, height)
    marker_center = np.array([size_mm / 2.0, size_mm / 2.0])
    half = settings.working_area_mm / 2.0

    if projected is None:
        lo, hi = marker_center - half, marker_center + half
    else:
        lo = projected.min(axis=0)
        hi = projected.max(axis=0)
        # Limita o **tamanho** da janela, não a sua posição: o marcador pode estar em
        # qualquer canto da plataforma, e recortar em torno dele amputaria os pés.
        # O encolhimento acontece em direção ao centro do conteúdo visível.
        center = 0.5 * (lo + hi)
        span = np.minimum(hi - lo, settings.working_area_mm)
        lo = center - span / 2.0
        hi = center + span / 2.0

    # O marcador precisa estar sempre contido na janela retificada.
    lo = np.minimum(lo, np.array([-2.0, -2.0]))
    hi = np.maximum(hi, np.array([size_mm + 2.0, size_mm + 2.0]))

    px_per_mm = float(settings.rectified_px_per_mm)
    span = hi - lo
    if not np.all(np.isfinite(span)) or np.any(span <= 0):
        raise CalibrationError("janela de retificação inválida (plano degenerado)")

    downscaled = False
    needed = np.max(span) * px_per_mm
    if needed > settings.max_rectified_px:
        px_per_mm = settings.max_rectified_px / float(np.max(span))
        downscaled = True

    # Quantiza a origem para um múltiplo exato de 1/px_per_mm: a conversão px↔mm
    # passa a ser exatamente invertível.
    origin_mm = np.floor(lo * px_per_mm) / px_per_mm
    out_w = int(np.ceil((hi[0] - origin_mm[0]) * px_per_mm))
    out_h = int(np.ceil((hi[1] - origin_mm[1]) * px_per_mm))
    out_w = int(np.clip(out_w, 64, settings.max_rectified_px))
    out_h = int(np.clip(out_h, 64, settings.max_rectified_px))

    # mm → px retificado (afim, escala uniforme)
    A = np.array(
        [[px_per_mm, 0.0, -origin_mm[0] * px_per_mm],
         [0.0, px_per_mm, -origin_mm[1] * px_per_mm],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    M = A @ H

    src = bgr
    ratio = detection.src_px_per_mm / px_per_mm if px_per_mm > 0 else 1.0
    if ratio > 1.5:
        # Antialiasing antes de reduzir a amostragem — evita moiré e serrilhado
        # no contorno, sem alterar a escala.
        sigma = 0.45 * (ratio - 1.0)
        ksize = int(2 * round(3 * sigma) + 1)
        if ksize >= 3:
            src = cv2.GaussianBlur(bgr, (ksize, ksize), sigma)
        interp = cv2.INTER_LINEAR
    else:
        interp = cv2.INTER_CUBIC

    rect = cv2.warpPerspective(
        src, M, (out_w, out_h), flags=interp,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )

    # Máscara de validade: distingue "preto real" de "fora da imagem original".
    ones = np.full(bgr.shape[:2], 255, dtype=np.uint8)
    valid = cv2.warpPerspective(
        ones, M, (out_w, out_h), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    valid = cv2.erode(valid, np.ones((3, 3), np.uint8), iterations=1)

    return Rectification(
        homography_image_to_mm=H,
        px_per_mm=px_per_mm,
        origin_mm=origin_mm,
        width_px=out_w,
        height_px=out_h,
        image=rect,
        warp_image_to_rect=M,
        valid_mask=valid,
        src_px_per_mm=detection.src_px_per_mm,
        downscaled=downscaled,
    )


def verify_round_trip(rectification: Rectification) -> tuple[list[float], float]:
    """Re-detecta o marcador na imagem retificada e mede seus lados em mm.

    É a verificação **fim-a-fim** da cadeia metrológica: se a homografia, a escala e a
    reamostragem estiverem corretas, cada lado deve medir 50,00 mm.
    Devolve ``(lados_mm, erro_maximo_mm)``.
    """
    from .marker import detect_marker  # import local evita ciclo

    settings = get_settings()
    det = detect_marker(rectification.image)
    if not det.found:
        return [], float("inf")

    corners_mm = rectification.rect_px_to_mm(det.corners_px)
    sides = [
        float(np.linalg.norm(corners_mm[(i + 1) % 4] - corners_mm[i])) for i in range(4)
    ]
    error = max(abs(s - settings.marker_size_mm) for s in sides)
    return sides, float(error)

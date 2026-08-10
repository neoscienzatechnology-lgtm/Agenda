"""Métricas objetivas de qualidade da captura.

Cada função devolve um número interpretável e independente de resolução, para que os
limiares em ``config.QualityThresholds`` façam sentido em qualquer câmera.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ExposureStats:
    mean_luma: float
    clipped_high_frac: float
    clipped_low_frac: float
    contrast_p5_p95: float


def focus_score(gray: np.ndarray, roi: np.ndarray | None = None) -> float:
    """Variância do laplaciano normalizada pela variância local do sinal.

    A normalização torna a métrica comparável entre cenas claras e escuras — a
    variância crua do laplaciano depende do contraste da cena, não só do foco.
    """
    g = gray.astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    if roi is not None and np.any(roi > 0):
        sel = roi > 0
        lap_var = float(np.var(lap[sel]))
        sig_var = float(np.var(g[sel]))
    else:
        lap_var = float(np.var(lap))
        sig_var = float(np.var(g))
    if sig_var < 1e-6:
        return 0.0
    return float(100.0 * lap_var / (sig_var + 25.0))


def motion_anisotropy(gray: np.ndarray) -> float:
    """0 = gradientes isotrópicos; →1 = borrão direcional (tremido).

    Um borrão de movimento suprime o gradiente na direção do movimento, criando forte
    anisotropia no tensor de estrutura.
    """
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.2)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    jxx = float(np.mean(gx * gx))
    jyy = float(np.mean(gy * gy))
    jxy = float(np.mean(gx * gy))
    trace = jxx + jyy
    if trace < 1e-9:
        return 0.0
    disc = np.sqrt(max(0.0, (jxx - jyy) ** 2 + 4.0 * jxy * jxy))
    l1 = 0.5 * (trace + disc)
    l2 = 0.5 * (trace - disc)
    if l1 < 1e-12:
        return 0.0
    return float(np.clip((l1 - l2) / (l1 + l2), 0.0, 1.0))


def exposure_stats(bgr: np.ndarray, roi: np.ndarray | None = None) -> ExposureStats:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    values = gray[roi > 0] if (roi is not None and np.any(roi > 0)) else gray.ravel()
    if values.size == 0:
        return ExposureStats(0.0, 0.0, 0.0, 0.0)
    v = values.astype(np.float32)
    p5, p95 = np.percentile(v, [5, 95])
    return ExposureStats(
        mean_luma=float(np.mean(v)),
        clipped_high_frac=float(np.mean(v >= 253)),
        clipped_low_frac=float(np.mean(v <= 2)),
        contrast_p5_p95=float(p95 - p5),
    )


def glare_mask_and_fraction(bgr: np.ndarray, roi: np.ndarray | None,
                            px_per_mm: float) -> tuple[np.ndarray, float]:
    """Reflexos especulares: manchas quase brancas, dessaturadas e compactas."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]
    mask = ((v >= 246) & (s <= 40)).astype(np.uint8) * 255
    if roi is not None:
        mask = cv2.bitwise_and(mask, (roi > 0).astype(np.uint8) * 255)

    k = max(1, int(round(0.7 * px_per_mm)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    denom = float(np.count_nonzero(roi > 0)) if roi is not None else float(v.size)
    frac = float(np.count_nonzero(mask)) / max(denom, 1.0)
    return mask, frac


def foreground_background_contrast(bgr: np.ndarray, fg_mask: np.ndarray,
                                   search_mask: np.ndarray) -> float:
    """Separação Lab média entre a região do pé e o fundo dentro da área útil."""
    fg = fg_mask > 0
    bg = (search_mask > 0) & ~fg
    if int(np.count_nonzero(fg)) < 200 or int(np.count_nonzero(bg)) < 200:
        return 0.0
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    mu_f = lab[fg].mean(axis=0)
    mu_b = lab[bg].mean(axis=0)
    sd_f = lab[fg].std(axis=0).mean()
    sd_b = lab[bg].std(axis=0).mean()
    dist = float(np.linalg.norm(mu_f - mu_b))
    # Penaliza cenas em que a variação interna rivaliza com a separação das classes.
    return float(dist / (1.0 + 0.02 * (sd_f + sd_b)))


def marker_roi_mask(shape: tuple[int, int], corners_px: np.ndarray,
                    expand: float = 1.6) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    if corners_px is None or len(corners_px) != 4:
        return mask
    center = corners_px.mean(axis=0)
    pts = ((corners_px - center) * expand + center).astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask

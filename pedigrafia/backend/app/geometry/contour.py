"""Extração, suavização e simplificação do contorno plantar.

A fronteira é obtida por *marching squares* (sub-pixel), nunca por
``cv2.findContours`` (que devolve centros de pixel e introduz viés de meio pixel —
0,08 mm a 6 px/mm, mas 0,25 mm em capturas de baixa resolução).

Dois contornos coexistem:
  * ``high_res_mm`` — detalhado, preservado internamente, base de todas as medidas
    automáticas;
  * ``editable_mm`` — simplificado para uma quantidade útil de nós, com erro
    geométrico verificado (≤ ``simplify_max_error_mm``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage import measure

from ..config import get_settings
from . import polygon as poly


@dataclass
class ContourPair:
    high_res_mm: np.ndarray
    editable_mm: np.ndarray
    simplify_error_mm: float
    smoothing_drift_mm: float
    node_count: int


def largest_subpixel_contour_px(mask: np.ndarray) -> np.ndarray | None:
    """Contorno externo sub-pixel da maior região de ``mask`` (coordenadas x, y)."""
    binary = (mask > 0).astype(np.float32)
    if not np.any(binary):
        return None
    # `find_contours` devolve (linha, coluna); o nível 0,5 cai exatamente na aresta
    # entre pixels de fundo e de objeto.
    contours = measure.find_contours(binary, 0.5)
    if not contours:
        return None
    best = max(contours, key=len)
    return np.stack([best[:, 1], best[:, 0]], axis=1).astype(np.float64)


def all_subpixel_contours_px(mask: np.ndarray) -> list[np.ndarray]:
    binary = (mask > 0).astype(np.float32)
    out = []
    for c in measure.find_contours(binary, 0.5):
        out.append(np.stack([c[:, 1], c[:, 0]], axis=1).astype(np.float64))
    out.sort(key=len, reverse=True)
    return out


def refine_subpixel(contour_px: np.ndarray, score: np.ndarray, level: float,
                    max_shift_px: float = 1.5, samples: int = 25) -> np.ndarray:
    """Reposiciona cada vértice no cruzamento exato de ``score = level``.

    Usa o campo escalar contínuo produzido pelo segmentador em vez da máscara binária.
    A janela de busca precisa ser larga o bastante para alcançar a fronteira
    geométrica real: a máscara binarizada por Otsu pode estar alguns pixels dilatada
    em capturas suavizadas (rotação da câmera, reamostragem, leve desfoque).

    Havendo mais de um cruzamento na janela, escolhe-se o **mais próximo** do vértice
    original — mover para uma borda distante seria trocar de estrutura, não refinar.
    """
    if score is None or contour_px is None or len(contour_px) < 4:
        return contour_px
    h, w = score.shape[:2]
    p = np.asarray(contour_px, dtype=np.float64)

    # Normal aproximada pela diferença central ao longo do contorno.
    nxt = np.roll(p, -1, axis=0)
    prv = np.roll(p, 1, axis=0)
    tangent = nxt - prv
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    norm = np.where(norm < 1e-9, 1.0, norm)
    tangent = tangent / norm
    normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)

    offsets = np.linspace(-max_shift_px, max_shift_px, samples)
    sampled = np.empty((len(p), samples), dtype=np.float64)
    for j, off in enumerate(offsets):
        q = p + normal * off
        xi = np.clip(q[:, 0], 0, w - 1)
        yi = np.clip(q[:, 1], 0, h - 1)
        x0 = np.floor(xi).astype(int)
        y0 = np.floor(yi).astype(int)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)
        fx = xi - x0
        fy = yi - y0
        s = score.astype(np.float64)
        sampled[:, j] = (s[y0, x0] * (1 - fx) * (1 - fy) + s[y0, x1] * fx * (1 - fy)
                         + s[y1, x0] * (1 - fx) * fy + s[y1, x1] * fx * fy)

    out = p.copy()
    rel = sampled - level
    for i in range(len(p)):
        r = rel[i]
        sign_changes = np.nonzero(np.diff(np.sign(r)) != 0)[0]
        if len(sign_changes) == 0:
            continue
        best_shift = None
        for k in sign_changes:
            k = int(k)
            denom = r[k + 1] - r[k]
            if abs(denom) < 1e-12:
                continue
            t = -r[k] / denom
            shift = offsets[k] + t * (offsets[k + 1] - offsets[k])
            if abs(shift) > max_shift_px:
                continue
            if best_shift is None or abs(shift) < abs(best_shift):
                best_shift = shift
        if best_shift is not None:
            out[i] = p[i] + normal[i] * best_shift
    return out


def prepare_contour(contour_mm: np.ndarray) -> ContourPair:
    """Reamostra, suaviza e simplifica — verificando o desvio dimensional em mm."""
    g = get_settings().geometry
    dense = poly.resample_closed(contour_mm, g.contour_resample_step_mm)

    before = _axis_length(dense)
    smoothed = poly.smooth_closed(dense, g.smoothing_window_mm,
                                  g.contour_resample_step_mm)
    after = _axis_length(smoothed)
    drift = abs(after - before)
    if drift > g.max_smoothing_length_drift_mm:
        # A suavização não pode custar milímetros reais: reduz a janela até caber
        # na tolerância, ou desiste dela.
        window = g.smoothing_window_mm
        while window > g.contour_resample_step_mm * 3:
            window *= 0.6
            smoothed = poly.smooth_closed(dense, window, g.contour_resample_step_mm)
            drift = abs(_axis_length(smoothed) - before)
            if drift <= g.max_smoothing_length_drift_mm:
                break
        else:
            smoothed = dense
            drift = 0.0

    editable, err = simplify_to_target(smoothed, g.simplify_target_nodes,
                                       g.simplify_max_error_mm,
                                       g.simplify_min_nodes, g.simplify_max_nodes)
    return ContourPair(high_res_mm=smoothed, editable_mm=editable,
                       simplify_error_mm=err, smoothing_drift_mm=drift,
                       node_count=len(editable))


def simplify_to_target(contour_mm: np.ndarray, target_nodes: int, max_error_mm: float,
                       min_nodes: int, max_nodes: int) -> tuple[np.ndarray, float]:
    """Busca binária no epsilon do RDP: nós próximos do alvo **e** erro dentro do limite."""
    lo, hi = 0.01, 6.0
    best = poly.simplify_closed(contour_mm, lo)
    best_err = poly.max_deviation(contour_mm, best)
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        cand = poly.simplify_closed(contour_mm, mid)
        err = poly.max_deviation(contour_mm, cand)
        if err > max_error_mm or len(cand) > target_nodes:
            hi = mid
        else:
            lo = mid
        if err <= max_error_mm and min_nodes <= len(cand) <= max_nodes:
            best, best_err = cand, err
            if len(cand) <= target_nodes:
                continue
    if len(best) < min_nodes:
        # Densifica preservando a forma: reamostra o contorno simplificado.
        best = poly.resample_closed(best, poly.perimeter(best) / max(min_nodes, 8))
        best_err = poly.max_deviation(contour_mm, best)
    return best, float(best_err)


def _axis_length(contour_mm: np.ndarray) -> float:
    """Comprimento ao longo do eixo principal — métrica sensível usada para o desvio."""
    u = poly.principal_axis(contour_mm)
    proj = np.asarray(contour_mm) @ u
    return float(np.max(proj) - np.min(proj))

"""Posicionamento do pé na folha A4 — **somente isometrias**.

Nenhum fator de escala é aplicado em nenhum ponto deste módulo. A transformação
página é composta por:

1. ``S`` — conversão do referencial do plano (y para baixo, como a câmera vê) para o
   referencial da folha (y para cima). Para captura de podoscópio (``view="below"``)
   isto é uma **reflexão**, e é exatamente o que se quer: o molde impresso precisa ser
   um gabarito físico sobre o qual o pé possa ser apoiado. Para ``view="above"`` a
   conversão é uma rotação de 180°, sem espelhamento.
2. ``R(θ)`` — rotação rígida que alinha o eixo longitudinal do pé à vertical da folha.
3. translação — centraliza horizontalmente e posiciona verticalmente.

``|det(M)| == 1`` e ``‖M·v‖ == ‖v‖`` são verificados em teste: qualquer escala
acidental quebraria a garantia 1:1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import A4_HEIGHT_MM, A4_WIDTH_MM, MM_TO_PT, get_settings


@dataclass
class PagePlacement:
    matrix: np.ndarray        # 2×2, isometria
    offset_mm: np.ndarray     # translação em mm de página
    center_plane_mm: np.ndarray
    rotation_deg: float
    mirrored: bool
    fits_on_page: bool
    content_width_mm: float
    content_height_mm: float
    overflow_mm: float

    def to_page_mm(self, pts_plane_mm) -> np.ndarray:
        p = np.asarray(pts_plane_mm, dtype=np.float64).reshape(-1, 2)
        return (p - self.center_plane_mm) @ self.matrix.T + self.offset_mm

    def to_page_pt(self, pts_plane_mm) -> np.ndarray:
        return self.to_page_mm(pts_plane_mm) * MM_TO_PT


def _view_matrix(view: str) -> tuple[np.ndarray, bool]:
    if view == "below":
        # Reflexão: o gabarito impresso é usado com a face para cima.
        return np.array([[1.0, 0.0], [0.0, -1.0]]), True
    return np.array([[-1.0, 0.0], [0.0, -1.0]]), False


def compute_placement(contour_plane_mm: np.ndarray, axis_a_mm, axis_b_mm,
                      view: str = "below", align_axis: bool | None = None,
                      margin_mm: float | None = None) -> PagePlacement:
    settings = get_settings()
    align = settings.pdf_align_axis_to_page if align_axis is None else align_axis
    margin = settings.pdf_margin_mm if margin_mm is None else margin_mm

    contour = np.asarray(contour_plane_mm, dtype=np.float64).reshape(-1, 2)
    S, mirrored = _view_matrix(view)

    theta = 0.0
    if align:
        a = np.asarray(axis_a_mm, dtype=np.float64)
        b = np.asarray(axis_b_mm, dtype=np.float64)
        u = b - a
        n = float(np.linalg.norm(u))
        if n > 1e-9:
            u_view = S @ (u / n)
            # Ângulo que leva u_view até (0, 1) — dedos apontando para o topo da folha.
            theta = float(np.arctan2(u_view[0], u_view[1]))

    c, s = float(np.cos(theta)), float(np.sin(theta))
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    M = R @ S

    center_plane = contour.mean(axis=0)
    local = (contour - center_plane) @ M.T
    lo, hi = local.min(axis=0), local.max(axis=0)
    width = float(hi[0] - lo[0])
    height = float(hi[1] - lo[1])

    usable_w = A4_WIDTH_MM - 2 * margin
    usable_h = A4_HEIGHT_MM - 2 * margin
    fits = width <= usable_w and height <= usable_h
    overflow = max(0.0, width - usable_w, height - usable_h)

    # Centraliza horizontalmente; verticalmente centraliza na folha inteira para
    # aproveitar a altura. NUNCA reescala — se não couber, o conteúdo sangra e o
    # chamador é avisado por `fits_on_page`.
    offset = np.array([
        A4_WIDTH_MM / 2.0 - 0.5 * (lo[0] + hi[0]),
        A4_HEIGHT_MM / 2.0 - 0.5 * (lo[1] + hi[1]),
    ])

    return PagePlacement(
        matrix=M, offset_mm=offset, center_plane_mm=center_plane,
        rotation_deg=float(np.degrees(theta)), mirrored=mirrored,
        fits_on_page=fits, content_width_mm=width, content_height_mm=height,
        overflow_mm=overflow,
    )


def is_isometry(matrix: np.ndarray, tol: float = 1e-9) -> bool:
    m = np.asarray(matrix, dtype=np.float64)
    return bool(np.allclose(m @ m.T, np.eye(2), atol=tol)
                and abs(abs(float(np.linalg.det(m))) - 1.0) < tol)

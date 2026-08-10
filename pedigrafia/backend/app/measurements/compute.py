"""Cálculo das medidas — **fonte única de verdade dimensional**.

Todas as entradas estão em milímetros no plano físico; todas as saídas também.
Nenhuma conversão de unidade acontece aqui.

Definições (fixadas e documentadas para que sejam reproduzíveis):

``lengthMm``
    Distância do ponto mais posterior do calcâneo ao ponto mais distal dos
    pododáctilos — o diâmetro do fecho convexo da projeção plantar.

    **Por que não a extensão paralela ao eixo:** a largura do pé é ~40 % do
    comprimento, então a extensão paralela ao eixo varia
    ``L·cos δ + W·sen δ`` com o ângulo do eixo: 1° de oscilação do eixo move a
    medida em ~1,8 mm. Como o eixo anatômico é estimado (e editável), essa medida
    não seria reprodutível entre capturas. O diâmetro do fecho convexo é um
    **máximo**, portanto estacionário: perturbações de primeira ordem não o alteram.
    Também é exatamente a definição pedida ("do ponto mais posterior do calcâneo até
    o dedo mais distal"), que não menciona eixo.

``axisParallelLengthMm``
    A extensão paralela ao eixo, mantida como valor auxiliar interno.

``forefootWidthMm``
    Largura máxima da secção transversal na faixa t ∈ [0,55; 0,88] — a "bola" do pé.

``midfootWidthMm``
    Largura **mínima** na faixa t ∈ [0,38; 0,62] — a cintura do mediopé.

``heelWidthMm``
    Largura máxima na faixa t ∈ [0,02; 0,24].

``heelToMetatarsalLineMm``
    Distância perpendicular do ponto mais posterior do calcâneo à reta M1–M5.

``archIndex``
    Ver :mod:`app.measurements.arch_index`.

Este módulo é espelhado em ``frontend/src/geom/measure.ts``; o teste de paridade
compara os dois contra o mesmo *fixture*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import get_settings
from ..geometry import polygon as poly
from ..geometry.frame import FootFrame
from .arch_index import ArchIndexResult, compute_arch_index

_WIDTH_SAMPLES = 121


@dataclass
class Measurements:
    length_mm: float
    forefoot_width_mm: float
    midfoot_width_mm: float
    heel_width_mm: float
    heel_to_metatarsal_line_mm: float
    arch_index: float

    plantar_area_mm2: float
    bbox_width_mm: float
    bbox_length_mm: float
    axis_length_mm: float
    orientation_deg: float
    forefoot_width_at_t: float = 0.0
    midfoot_width_at_t: float = 0.0
    heel_width_at_t: float = 0.0
    arch_index_toe_cut_t: float = 0.0
    arch_index_basis: str = "silhouette"
    arch_areas_mm2: list[float] = field(default_factory=list)
    heel_to_m1_mm: float = 0.0
    heel_to_m5_mm: float = 0.0
    metatarsal_line_length_mm: float = 0.0
    axis_parallel_length_mm: float = 0.0
    warnings: list[str] = field(default_factory=list)


def _extreme_width(uv: np.ndarray, frame: FootFrame, band: tuple[float, float],
                   mode: str) -> tuple[float, float]:
    """Largura extrema (``max``/``min``) na faixa e o ``t`` em que ocorre."""
    ts = np.linspace(band[0], band[1], _WIDTH_SAMPLES)
    widths = np.array([poly.width_at_u(uv, float(t) * frame.length_mm) for t in ts])
    valid = np.isfinite(widths) & (widths > 0)
    if not np.any(valid):
        return 0.0, 0.0
    idx = np.nonzero(valid)[0]
    pick = idx[int(np.argmax(widths[idx]))] if mode == "max" \
        else idx[int(np.argmin(widths[idx]))]
    return float(widths[pick]), float(ts[pick])


def compute_measurements(contour_mm: np.ndarray, frame: FootFrame,
                         m1_mm: np.ndarray | None = None,
                         m5_mm: np.ndarray | None = None,
                         toe_cut_t: float | None = None,
                         arch_basis: str = "silhouette"
                         ) -> tuple[Measurements, ArchIndexResult]:
    g = get_settings().geometry
    dense = poly.resample_closed(contour_mm, 0.5)
    uv = frame.to_local(dense)

    axis_parallel_length = float(np.max(uv[:, 0]) - np.min(uv[:, 0]))
    bbox_width = float(np.max(uv[:, 1]) - np.min(uv[:, 1]))
    # Comprimento oficial: diâmetro do fecho convexo (estacionário, independente do
    # eixo). Ver a docstring do módulo para a justificativa metrológica.
    length, _, _ = poly.max_caliper(dense)

    forefoot, forefoot_t = _extreme_width(uv, frame, g.forefoot_band, "max")
    midfoot, midfoot_t = _extreme_width(uv, frame, g.midfoot_band, "min")
    heel, heel_t = _extreme_width(uv, frame, g.heel_band, "max")

    heel_point = dense[int(np.argmin(uv[:, 0]))]
    warnings: list[str] = []

    heel_to_line = 0.0
    heel_to_m1 = heel_to_m5 = mt_len = 0.0
    if m1_mm is not None and m5_mm is not None:
        m1 = np.asarray(m1_mm, dtype=np.float64)
        m5 = np.asarray(m5_mm, dtype=np.float64)
        mt_len = float(np.linalg.norm(m5 - m1))
        if mt_len > 1e-6:
            heel_to_line = poly.point_line_distance(heel_point, m1, m5)
        heel_to_m1 = float(np.linalg.norm(m1 - heel_point))
        heel_to_m5 = float(np.linalg.norm(m5 - heel_point))
    else:
        warnings.append("cabeças metatarsais ausentes: distância calcâneo→linha "
                        "metatarsal não calculada")

    arch = compute_arch_index(dense, frame, toe_cut_t, arch_basis)
    if arch.warning:
        warnings.append(arch.warning)

    meas = Measurements(
        length_mm=length,
        forefoot_width_mm=forefoot,
        midfoot_width_mm=midfoot,
        heel_width_mm=heel,
        heel_to_metatarsal_line_mm=heel_to_line,
        arch_index=arch.value,
        plantar_area_mm2=float(poly.area(dense)),
        bbox_width_mm=bbox_width,
        bbox_length_mm=axis_parallel_length,
        axis_length_mm=axis_parallel_length,
        orientation_deg=frame.orientation_deg,
        forefoot_width_at_t=forefoot_t,
        midfoot_width_at_t=midfoot_t,
        heel_width_at_t=heel_t,
        arch_index_toe_cut_t=arch.toe_cut_t,
        arch_index_basis=arch.basis,
        arch_areas_mm2=arch.areas_mm2,
        heel_to_m1_mm=heel_to_m1,
        heel_to_m5_mm=heel_to_m5,
        metatarsal_line_length_mm=mt_len,
        axis_parallel_length_mm=axis_parallel_length,
        warnings=warnings,
    )
    return meas, arch


def local_coords(points: dict[str, np.ndarray], frame: FootFrame,
                 length_mm: float) -> list[dict]:
    """Landmarks no frame do pé — exigido pelo requisito de coordenadas relativas."""
    out = []
    for name, p in points.items():
        uv = frame.to_local(np.asarray(p, dtype=np.float64)[None, :])[0]
        out.append({
            "id": name,
            "uMm": float(uv[0]),
            "vMm": float(uv[1]),
            "tPct": float(100.0 * uv[0] / length_mm) if length_mm > 0 else 0.0,
        })
    return out


def shoe_size_check(measured_mm: float, shoe_size: float | None,
                    system: str = "BR") -> dict:
    """Conferência **puramente informativa**. Não altera nenhuma dimensão.

    A faixa esperada vem de tabelas usuais de numeração; a divergência é reportada,
    nunca corrigida. O PDF continua saindo com ``measured_mm``.
    """
    if shoe_size is None:
        return {"provided": None, "system": system, "message": ""}

    # Faixas aproximadas de comprimento do PÉ (não da fôrma) por numeração.
    # BR/EU compartilham a mesma progressão de ~6,67 mm por número.
    if system.upper() in ("BR", "EU"):
        center = 6.6667 * float(shoe_size) - 2.0
    elif system.upper() == "US":
        center = 8.47 * float(shoe_size) + 156.0
    else:
        return {"provided": shoe_size, "system": system,
                "message": "Sistema de numeração desconhecido; conferência ignorada."}

    lo, hi = center - 7.0, center + 7.0
    compatible = lo <= measured_mm <= hi
    if compatible:
        msg = (f"Dimensão compatível com a faixa informada "
               f"({lo:.0f}–{hi:.0f} mm para {shoe_size:g} {system}).")
    else:
        msg = (f"Comprimento medido ({measured_mm:.1f} mm) fora da faixa típica do "
               f"número {shoe_size:g} {system} ({lo:.0f}–{hi:.0f} mm). "
               f"O valor medido pela calibração é o que será usado.")
    return {
        "provided": float(shoe_size), "system": system,
        "expectedRangeMm": [float(lo), float(hi)],
        "measuredMm": float(measured_mm), "compatible": bool(compatible),
        "message": msg,
    }

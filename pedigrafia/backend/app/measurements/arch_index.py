"""Índice geométrico do arco plantar em projeção 2D.

**Nomenclatura honesta.** Isto NÃO é "altura do arco" nem um índice de pressão.
É a razão de áreas de Cavanagh & Rodgers aplicada à **projeção plantar observável**:
a região (excluindo os pododáctilos) é dividida em três faixas longitudinais iguais e
o índice é ``área_média / área_total``.

``basis`` registra sobre o que o índice foi calculado:

* ``silhouette`` — a silhueta do pé. É o padrão quando a foto não permite distinguir
  a área de contato. **Não é comparável** com valores de literatura obtidos por
  pedigrafia de contato/tinta.
* ``contact``    — a região de contato efetivamente identificada (podoscópio com
  isquemia visível). Só é usada quando a confiança da detecção passa do limiar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from ..config import get_settings
from ..geometry import polygon as poly
from ..geometry.frame import FootFrame


@dataclass
class ArchIndexResult:
    value: float
    toe_cut_t: float
    areas_mm2: list[float]
    basis: str
    confidence: float
    method: str
    warning: str = ""


def detect_toe_cut_t(uv: np.ndarray, frame: FootFrame) -> tuple[float, float]:
    """Localiza a linha de corte dos pododáctilos pelo vale do perfil de largura.

    Devolve ``(t, confiança)``. Sem vale claro, cai no valor proporcional padrão e
    devolve confiança baixa — sinal para a UI pedir revisão da linha.
    """
    g = get_settings().geometry
    ts = np.linspace(0.62, 0.94, 65)
    widths = np.array([poly.width_at_u(uv, float(t) * frame.length_mm) for t in ts])
    finite = np.isfinite(widths) & (widths > 0)
    if int(np.count_nonzero(finite)) < 20:
        return g.default_toe_cut_t, 0.15

    ref = float(np.max(widths[finite]))
    if ref <= 0:
        return g.default_toe_cut_t, 0.15

    # Um corte válido exige um **mínimo local** com rebote: a largura precisa cair
    # (fim do antepé) e voltar a subir (base dos pododáctilos abertos). Uma silhueta
    # em que os dedos são contíguos ao antepé simplesmente estreita de forma
    # monótona e NÃO tem vale — nesse caso não se inventa um: usa-se a proporção
    # padrão com confiança baixa, e a linha fica destacada para revisão manual.
    filled = np.where(finite, widths, ref)
    peak_idx = int(np.argmax(filled))
    idx, props = find_peaks(-filled,
                            prominence=g.toe_valley_min_prominence_frac * ref)
    prominences = {int(i): float(p) for i, p in zip(idx, props["prominences"])}
    valleys = [i for i in prominences if i > peak_idx + 2]
    if not valleys:
        return g.default_toe_cut_t, 0.20

    valley_local = max(valleys, key=lambda v: prominences[v])
    depth = (ref - filled[valley_local]) / ref
    after = np.arange(valley_local + 1, len(ts))
    rebound = (float(np.max(filled[after]) - filled[valley_local]) / ref
               if len(after) >= 2 else 0.0)
    if rebound < 0.5 * g.toe_valley_min_prominence_frac:
        return g.default_toe_cut_t, 0.20

    confidence = float(np.clip(0.35 + 2.2 * depth + 1.5 * rebound, 0.0, 0.95))
    return float(ts[valley_local]), confidence


def compute_arch_index(contour_mm: np.ndarray, frame: FootFrame,
                       toe_cut_t: float | None = None,
                       basis: str = "silhouette") -> ArchIndexResult:
    g = get_settings().geometry
    uv = frame.to_local(poly.resample_closed(contour_mm, 0.5))

    if toe_cut_t is None:
        toe_cut_t, cut_conf = detect_toe_cut_t(uv, frame)
    else:
        cut_conf = 1.0   # definido/confirmado pelo profissional

    toe_cut_t = float(np.clip(toe_cut_t, 0.55, 0.97))
    u_cut = toe_cut_t * frame.length_mm
    body = poly.clip_band_u(uv, 0.0, u_cut)
    if len(body) < 3:
        return ArchIndexResult(0.0, toe_cut_t, [], basis, 0.0, "cavanagh_ratio",
                               "região plantar insuficiente para o índice")

    third = u_cut / 3.0
    areas = []
    for k in range(3):
        band = poly.clip_band_u(uv, k * third, (k + 1) * third)
        areas.append(float(poly.area(band)) if len(band) >= 3 else 0.0)

    total = sum(areas)
    value = areas[1] / total if total > 1e-9 else 0.0

    warning = ""
    if basis == "silhouette":
        warning = (
            "Índice calculado sobre a SILHUETA plantar, não sobre a área de contato. "
            "Não é comparável a valores de pedigrafia de contato nem representa "
            "altura de arco."
        )

    return ArchIndexResult(
        value=float(value), toe_cut_t=float(toe_cut_t), areas_mm2=areas,
        basis=basis, confidence=float(np.clip(cut_conf, 0.0, 1.0)),
        method="cavanagh_ratio", warning=warning,
    )

"""Detecção dos ápices dos pododáctilos (T1–T5).

Os ápices são **observáveis** na silhueta plantar: são máximos locais da projeção
longitudinal ao longo do contorno. Por isso recebem confiança alta quando o pico é
proeminente, e baixa quando precisam ser interpolados.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from ..config import get_settings
from ..geometry import polygon as poly
from ..geometry.frame import FootFrame


@dataclass
class ToeApex:
    index: int              # 0 = T1 (hálux) … 4 = T5
    point_mm: np.ndarray
    confidence: float
    method: str
    lobe_width_mm: float = 0.0


def detect_toe_apices(contour_mm: np.ndarray, frame: FootFrame
                      ) -> tuple[list[ToeApex], int, float]:
    """Devolve ``(apices_T1_a_T5, medial_sign, confiança_da_lateralidade)``.

    O hálux é o lobo **mais largo** da região distal; o lado em que ele está define
    qual sentido de ``v`` é medial — base geométrica da classificação D/E.
    """
    g = get_settings().geometry
    step = g.contour_resample_step_mm
    # `resample_closed` garante espaçamento uniforme igual a `step`, então aqui o
    # passo pode ser assumido — ao contrário de `bootstrap_frame`, que recebe o
    # contorno bruto.
    dense = poly.resample_closed(contour_mm, step)
    uv = frame.to_local(dense)
    u = uv[:, 0]
    v = uv[:, 1]

    u_start = g.toe_region_t_start * frame.length_mm
    in_region = u >= u_start
    if int(np.count_nonzero(in_region)) < 12:
        return [], 0, 0.0

    # O contorno é circular: duplica o sinal e procura picos na cópia central para
    # não perder um ápice que caia sobre o ponto inicial.
    seq = np.where(in_region, u, u_start)
    doubled = np.concatenate([seq, seq])
    distance = max(2, int(round(g.toe_peak_min_separation_mm / step)))
    peaks, props = find_peaks(doubled, prominence=g.toe_peak_min_prominence_mm,
                              distance=distance, width=1)
    n = len(seq)
    sel = [(int(p % n), float(props["prominences"][k]),
            float(props["widths"][k]) * step)
           for k, p in enumerate(peaks) if n // 2 <= p < n // 2 + n]

    # Deduplica índices equivalentes vindos da duplicação.
    seen: dict[int, tuple[float, float]] = {}
    for idx, prom, width in sel:
        prev = seen.get(idx)
        if prev is None or prom > prev[0]:
            seen[idx] = (prom, width)
    found = [(idx, p, w) for idx, (p, w) in seen.items()]
    if not found:
        return [], 0, 0.0

    # Mantém no máximo 5 lobos, os mais proeminentes.
    found.sort(key=lambda t: -t[1])
    found = found[:5]

    # O hálux é o lobo mais largo (e tipicamente o mais distal).
    def hallux_score(entry) -> float:
        idx, prom, width = entry
        return width * 1.0 + 0.35 * float(u[idx])

    hallux = max(found, key=hallux_score)
    hallux_v = float(v[hallux[0]])
    others_v = [float(v[i]) for i, _, _ in found if i != hallux[0]]
    mean_other_v = float(np.mean(others_v)) if others_v else 0.0
    medial_sign = 1 if hallux_v > mean_other_v else -1

    # Confiança da lateralidade: separação do hálux em relação aos demais lobos,
    # normalizada pela largura do antepé.
    forefoot_span = float(np.max(v) - np.min(v)) if len(v) else 1.0
    separation = abs(hallux_v - mean_other_v) / max(forefoot_span, 1e-6)
    width_ratio = hallux[2] / max(np.mean([w for _, _, w in found]), 1e-6)
    lat_conf = float(np.clip(0.55 * min(1.0, separation / 0.30)
                             + 0.45 * min(1.0, (width_ratio - 1.0) / 0.8), 0.0, 1.0))
    if len(found) < 3:
        lat_conf *= 0.6

    # Ordena medial → lateral no sentido definido por medial_sign.
    found.sort(key=lambda t: -medial_sign * float(v[t[0]]))

    apices: list[ToeApex] = []
    max_prom = max(p for _, p, _ in found)
    for k, (idx, prom, width) in enumerate(found[:5]):
        conf = float(np.clip(0.35 + 0.65 * (prom / max(max_prom, 1e-6)), 0.0, 1.0))
        apices.append(ToeApex(index=k, point_mm=dense[idx].copy(), confidence=conf,
                              method="contour_peak", lobe_width_mm=width))

    if len(apices) < 5:
        apices = _interpolate_missing(apices, dense, frame, medial_sign)
    return apices[:5], medial_sign, lat_conf


def _interpolate_missing(apices: list[ToeApex], dense: np.ndarray, frame: FootFrame,
                         medial_sign: int) -> list[ToeApex]:
    """Completa T1–T5 quando a silhueta não separa todos os pododáctilos.

    Os pontos criados aqui são **estimados** e recebem confiança baixa: a UI precisa
    destacá-los para revisão.
    """
    if not apices:
        return apices
    uv = frame.to_local(dense)
    known_uv = frame.to_local(np.array([a.point_mm for a in apices]))

    # Modelo proporcional do arco digital: v uniformemente distribuído entre os
    # extremos observados, u seguindo a curva quadrática ajustada aos observados.
    v_known = known_uv[:, 1] * medial_sign      # crescente = medial → lateral
    u_known = known_uv[:, 0]
    order = np.argsort(-v_known)
    v_sorted = v_known[order]
    u_sorted = u_known[order]

    forefoot_v = uv[:, 1] * medial_sign
    v_med = float(np.max(forefoot_v))
    v_lat = float(np.min(forefoot_v))
    # Posições relativas canônicas dos ápices T1..T5 na largura do antepé.
    canonical = np.array([0.86, 0.62, 0.44, 0.27, 0.11])
    targets_v = v_lat + canonical * (v_med - v_lat)

    if len(v_sorted) >= 3:
        coeffs = np.polyfit(v_sorted, u_sorted, 2)
    elif len(v_sorted) == 2:
        coeffs = np.polyfit(v_sorted, u_sorted, 1)
    else:
        coeffs = np.array([u_sorted[0]])

    out: list[ToeApex] = []
    for k in range(5):
        tv = float(targets_v[k])
        near = [a for a, kv in zip(apices, v_known)
                if abs(kv - tv) < 0.09 * abs(v_med - v_lat)]
        if near:
            best = max(near, key=lambda a: a.confidence)
            if not any(o.point_mm is best.point_mm for o in out):
                out.append(ToeApex(index=k, point_mm=best.point_mm,
                                   confidence=best.confidence, method=best.method,
                                   lobe_width_mm=best.lobe_width_mm))
                continue
        u_est = float(np.polyval(coeffs, tv))
        p = frame.to_plane(np.array([[u_est, tv * medial_sign]]))[0]
        out.append(ToeApex(index=k, point_mm=p, confidence=0.25,
                           method="interpolated_model"))
    return out

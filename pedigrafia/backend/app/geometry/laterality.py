"""Classificação pé direito / pé esquerdo.

**Nunca** decidida pela posição na imagem. A decisão vem de geometria anatômica:

1. em qual lado do eixo longitudinal está o hálux (lobo digital mais largo);
2. de que lado está a concavidade do arco (a borda medial é côncava, a lateral é
   convexa) — pista independente que confirma ou contradiz a primeira;
3. em capturas bilaterais, a coerência entre os dois pés (as bordas mediais se
   voltam uma para a outra) é usada como verificação cruzada.

O resultado é sempre corrigível manualmente na revisão.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..synth.foot_shape import laterality_for
from . import polygon as poly
from .frame import FootFrame


@dataclass
class LateralityResult:
    laterality: str
    confidence: float
    medial_sign: int
    method: str
    cues: dict


def arch_concavity_sign(contour_mm: np.ndarray, frame: FootFrame) -> tuple[int, float]:
    """Sinal de ``v`` do lado côncavo (medial) e a força da evidência.

    Compara, no mediopé, o afastamento de cada borda em relação à corda que liga o
    retropé ao antepé: a borda medial recua (arco), a lateral não.
    """
    uv = frame.to_local(poly.resample_closed(contour_mm, 0.75))
    L = frame.length_mm
    if L <= 0:
        return 1, 0.0

    def border_profile(side: int) -> np.ndarray:
        """Perfil |v| máximo do lado ``side`` em função de t."""
        ts = np.linspace(0.18, 0.78, 40)
        out = []
        for t in ts:
            u = t * L
            vs = poly.crossings_at_u(uv, u)
            if vs.size < 2:
                out.append(np.nan)
                continue
            out.append(float(np.max(vs * side)))
        return np.array(out)

    ts = np.linspace(0.18, 0.78, 40)
    scores = {}
    for side in (1, -1):
        prof = border_profile(side)
        ok = np.isfinite(prof)
        if int(np.count_nonzero(ok)) < 12:
            return 1, 0.0
        # Corda entre retropé (t≈0,20) e antepé (t≈0,75).
        a_idx = int(np.argmin(np.abs(ts - 0.22)))
        b_idx = int(np.argmin(np.abs(ts - 0.74)))
        chord = np.interp(ts, [ts[a_idx], ts[b_idx]], [prof[a_idx], prof[b_idx]])
        deficit = chord - prof            # positivo = recuo (concavidade)
        mid = (ts >= 0.34) & (ts <= 0.60) & ok
        scores[side] = float(np.mean(deficit[mid])) if np.any(mid) else 0.0

    medial = 1 if scores[1] >= scores[-1] else -1
    diff = abs(scores[1] - scores[-1])
    strength = float(np.clip(diff / (0.030 * L), 0.0, 1.0))
    return medial, strength


def classify(contour_mm: np.ndarray, frame: FootFrame, hallux_sign: int,
             hallux_confidence: float, view: str = "below") -> LateralityResult:
    arch_sign, arch_strength = arch_concavity_sign(contour_mm, frame)

    # Combinação ponderada das duas pistas independentes.
    vote = hallux_confidence * hallux_sign + 0.8 * arch_strength * arch_sign
    medial_sign = 1 if vote >= 0 else -1
    agreement = hallux_sign == arch_sign

    total = hallux_confidence + 0.8 * arch_strength
    confidence = abs(vote) / total if total > 1e-6 else 0.0
    if agreement:
        confidence = min(1.0, confidence * 1.15)
    else:
        confidence *= 0.55

    return LateralityResult(
        laterality=laterality_for(medial_sign, view),
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        medial_sign=medial_sign,
        method="hallux_side+arch_concavity",
        cues={
            "halluxSign": int(hallux_sign),
            "halluxConfidence": float(hallux_confidence),
            "archConcavitySign": int(arch_sign),
            "archConcavityStrength": float(arch_strength),
            "agreement": bool(agreement),
            "view": view,
        },
    )


def reconcile_pair(results: list[LateralityResult], centroids_mm: list[np.ndarray],
                   view: str) -> list[LateralityResult]:
    """Verificação cruzada em captura bilateral.

    As bordas mediais de ambos os pés se voltam uma para a outra. Se as duas
    classificações forem iguais (dois "direitos"), a de menor confiança é invertida e
    ambas são rebaixadas — o profissional precisa confirmar.
    """
    if len(results) != 2:
        return results

    expected = []
    order = np.argsort([float(c[0]) for c in centroids_mm])
    # Pé mais à esquerda na imagem tem a borda medial apontando para +x.
    for rank, idx in enumerate(order):
        expected.append((idx, 1 if rank == 0 else -1))

    consistent = all(results[i].medial_sign == sign for i, sign in expected)
    if consistent:
        for r in results:
            r.confidence = float(min(1.0, r.confidence * 1.2 + 0.08))
            r.cues["bilateralCheck"] = "consistent"
        return results

    if results[0].laterality == results[1].laterality:
        weaker = 0 if results[0].confidence <= results[1].confidence else 1
        r = results[weaker]
        r.medial_sign = -r.medial_sign
        r.laterality = laterality_for(r.medial_sign, view)
        r.method += "+bilateral_correction"
        for res in results:
            res.confidence = float(min(res.confidence, 0.55))
            res.cues["bilateralCheck"] = "corrected_duplicate"
        return results

    for i, sign in expected:
        results[i].cues["bilateralCheck"] = "inconsistent_positions"
        results[i].confidence = float(min(results[i].confidence, 0.7))
        del sign
    return results

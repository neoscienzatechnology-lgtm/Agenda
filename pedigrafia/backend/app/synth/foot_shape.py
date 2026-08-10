"""Contorno plantar paramétrico canônico — **verdade de referência** dos testes.

O polígono é construído analiticamente (sem imagem), com cinco pododáctilos, arco
medial côncavo, proeminência da 1ª cabeça metatarsal, borda lateral e calcâneo.

Normalização: a escala do polígono é resolvida iterativamente para que o
**comprimento medido pelo próprio pipeline** (extensão paralela ao eixo anatômico)
valha exatamente ``length_mm``. Assim, ``foot_polygon(length_mm=265)`` produz uma
figura cujo comprimento verdadeiro é 265,000 mm, e o teste metrológico compara contra
esse valor sem circularidade: o que está sob teste é a cadeia imagem → mm, não a
definição de comprimento.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

# ------------------------------------------------------------------ modelo anatômico
# Coordenadas (u, w): u = 0 no calcâneo, 1 na ponta do hálux;
# w > 0 = lado MEDIAL, w < 0 = lado LATERAL. Frações do comprimento do pé.

_HEEL = [
    (0.030, 0.100), (0.012, 0.070), (0.003, 0.035), (0.000, 0.000),
    (0.003, -0.038), (0.013, -0.072), (0.032, -0.102),
]

_LATERAL = [
    (0.090, -0.132), (0.170, -0.152), (0.260, -0.166), (0.360, -0.178),
    (0.450, -0.190), (0.545, -0.202), (0.625, -0.208), (0.700, -0.202),
    (0.778, -0.188),
]

_MEDIAL = [  # da base do hálux de volta ao calcâneo
    (0.755, 0.196), (0.700, 0.190), (0.620, 0.160), (0.520, 0.120),
    (0.420, 0.098), (0.330, 0.100), (0.240, 0.118), (0.160, 0.135),
    (0.090, 0.128),
]

# (apex_u, base_u, center_w, half_w) — ordem medial → lateral.
# Os intervalos em w se SOBREPÕEM de propósito: pododáctilos reais se tocam, e a
# silhueta resultante tem entalhes interdigitais rasos e resolvíveis, em vez de
# fendas de largura zero que nenhuma câmera (nem rasterizador) consegue reproduzir.
_TOES = [
    (1.000, 0.800, 0.118, 0.078),   # T1 — hálux
    (0.978, 0.800, 0.030, 0.045),   # T2
    (0.948, 0.795, -0.038, 0.042),  # T3
    (0.910, 0.788, -0.098, 0.040),  # T4
    (0.862, 0.778, -0.150, 0.038),  # T5
]

TOE_APEX_UW = [(t[0], t[2]) for t in _TOES]
"""Ápices T1..T5 em (u, w) do modelo canônico, ordem medial → lateral."""

M1_UW = (0.755, 0.196)
M5_UW = (0.625, -0.208)

_TOE_W_MEDIAL = _TOES[0][2] + _TOES[0][3]     # +0.196
_TOE_W_LATERAL = _TOES[-1][2] - _TOES[-1][3]  # -0.188


def _toe_union_u(w: float) -> float:
    """Silhueta dos pododáctilos: envoltória superior das calotas elípticas."""
    best = min(t[1] for t in _TOES)
    for apex_u, base_u, c_w, hw in _TOES:
        d = (w - c_w) / hw
        if abs(d) <= 1.0:
            best = max(best, base_u + (apex_u - base_u) * float(np.sqrt(1.0 - d * d)))
    return best


def _toe_profile(samples: int = 320) -> list[tuple[float, float]]:
    """Perfil lateral → medial da região dos pododáctilos."""
    ws = np.linspace(_TOE_W_LATERAL, _TOE_W_MEDIAL, samples)
    return [(_toe_union_u(float(w)), float(w)) for w in ws]


def _canonical_uw() -> np.ndarray:
    pts: list[tuple[float, float]] = []
    pts.extend(_HEEL)
    pts.extend(_LATERAL)
    pts.extend(_toe_profile())
    pts.extend(_MEDIAL)
    return np.array(pts, dtype=np.float64)


def _densify(uw: np.ndarray, step: float = 0.004) -> np.ndarray:
    """Interpola linearmente para obter um contorno denso (evita viés de amostragem)."""
    closed = np.vstack([uw, uw[:1]])
    out = []
    for i in range(len(closed) - 1):
        a, b = closed[i], closed[i + 1]
        d = float(np.linalg.norm(b - a))
        n = max(1, int(np.ceil(d / step)))
        for k in range(n):
            out.append(a + (b - a) * (k / n))
    return np.array(out, dtype=np.float64)


@lru_cache(maxsize=1)
def _base_uw() -> np.ndarray:
    """Polígono canônico denso em unidades de projeto (``u``-extent ≈ 1,005)."""
    return _densify(_canonical_uw())


def _measure_scaled(scale: float) -> float:
    """Comprimento que o **pipeline** mede para o canônico reescalado por ``scale``.

    A normalização é resolvida iterativamente para tolerar qualquer não-linearidade
    da definição de comprimento em relação à escala.
    """
    from ..geometry.polygon import max_caliper

    # Mesma definição do pipeline: diâmetro do fecho convexo.
    length, _, _ = max_caliper(_base_uw() * scale)
    return float(length)


@lru_cache(maxsize=64)
def _scale_for_length(length_mm: float) -> float:
    """Escala tal que o comprimento medido pelo pipeline seja exatamente ``length_mm``."""
    scale = float(length_mm)
    for _ in range(6):
        measured = _measure_scaled(scale)
        if measured <= 0:
            raise RuntimeError("normalização do contorno canônico falhou")
        correction = length_mm / measured
        scale *= correction
        if abs(correction - 1.0) < 1e-12:
            break
    return scale


def canonical_scale_factor(length_mm: float = 260.0) -> float:
    """Razão entre o comprimento medido e o ``u``-extent de projeto (documentação)."""
    scale = _scale_for_length(length_mm)
    span = float(np.ptp(_base_uw()[:, 0])) * scale
    return length_mm / span if span > 0 else 1.0


def foot_polygon(length_mm: float, laterality: str = "right", view: str = "below",
                 *, center_mm: tuple[float, float] = (0.0, 0.0),
                 rotation_deg: float = 0.0) -> np.ndarray:
    """Contorno plantar em mm, no plano físico.

    ``length_mm`` é o comprimento verdadeiro (paralelo ao eixo anatômico).
    O sentido medial/lateral segue a convenção de ``view``:
    ``below`` (podoscópio) espelha horizontalmente em relação a ``above``.
    """
    if laterality not in ("left", "right"):
        raise ValueError("laterality deve ser 'left' ou 'right'")
    if view not in ("below", "above"):
        raise ValueError("view deve ser 'below' ou 'above'")

    uw = _base_uw() * _scale_for_length(float(length_mm))
    medial_sign = medial_sign_for(laterality, view)

    # u aponta para os dedos = -y do plano (dedos para cima na imagem);
    # w medial mapeia para x com o sinal de `medial_sign`.
    x = uw[:, 1] * medial_sign
    y = -uw[:, 0]
    pts = np.stack([x, y], axis=1)

    theta = np.radians(rotation_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts = pts @ R.T
    pts = pts - pts.mean(axis=0) + np.asarray(center_mm, dtype=np.float64)
    return pts


def medial_sign_for(laterality: str, view: str) -> int:
    """+1 se a borda medial aponta para +x da imagem.

    Podoscópio (câmera **sob** o vidro, dedos para cima): o pé direito aparece com o
    hálux à direita. Visto de cima, o espelhamento inverte o sinal.
    """
    return 1 if (view == "below") == (laterality == "right") else -1


def laterality_for(medial_sign: int, view: str) -> str:
    """Inversa de :func:`medial_sign_for` — usada pela classificação D/E."""
    if view == "below":
        return "right" if medial_sign > 0 else "left"
    return "left" if medial_sign > 0 else "right"


def expected_landmarks_mm(length_mm: float, laterality: str = "right",
                          view: str = "below", *,
                          center_mm: tuple[float, float] = (0.0, 0.0),
                          rotation_deg: float = 0.0) -> dict[str, np.ndarray]:
    """Landmarks verdadeiros do modelo, no mesmo referencial de :func:`foot_polygon`."""
    # Os landmarks canônicos estão em unidades de projeto; convertem-se com o mesmo
    # fator de normalização aplicado ao polígono.
    factor = _scale_for_length(float(length_mm))

    medial_sign = medial_sign_for(laterality, view)
    theta = np.radians(rotation_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    poly_pts = foot_polygon(length_mm, laterality, view,
                            center_mm=center_mm, rotation_deg=rotation_deg)
    raw = _base_uw() * factor
    raw_xy = np.stack([raw[:, 1] * medial_sign, -raw[:, 0]], axis=1) @ R.T
    offset = np.asarray(center_mm, float) - raw_xy.mean(axis=0)
    assert np.allclose(raw_xy + offset, poly_pts, atol=1e-9)

    def place(uw: tuple[float, float]) -> np.ndarray:
        u, w = uw[0] * factor, uw[1] * factor
        p = np.array([w * medial_sign, -u]) @ R.T
        return p + offset

    out = {f"T{i + 1}": place(TOE_APEX_UW[i]) for i in range(5)}
    out["M1"] = place(M1_UW)
    out["M5"] = place(M5_UW)
    out["heel"] = place((0.0, 0.0))
    return out


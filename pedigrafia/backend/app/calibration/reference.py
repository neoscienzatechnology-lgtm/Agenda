"""Referências de dimensão padronizada — calibrar sem imprimir marcador.

Por que isto existe
-------------------
O alvo ArUco de quatro marcadores é o método mais exato do sistema, mas exige
imprimir e colar folhas. Um objeto retangular de dimensão **normalizada** já
resolve o mesmo problema físico: ele traz para dentro da foto uma medida real,
que é a única coisa que uma fotografia sozinha não tem.

O que serve e o que não serve
-----------------------------
Serve qualquer **retângulo plano de dimensão padronizada**: os quatro cantos dão
quatro correspondências px↔mm, exatamente como um marcador. Cartão ID-1 e folha
A4 estão no catálogo.

**Não serve moeda.** A projeção de um círculo é uma elipse, e sem os parâmetros
intrínsecos da câmera a elipse só permite recuperar a escala assumindo que o plano
está frontal. O erro dessa suposição é ``d · sen(θ) / Z``: com a moeda a 150 mm do
pé, inclinação de apenas 3° e câmera a 600 mm, dá 1,3 % — 3,4 mm em um pé de
265 mm. E uma inclinação de 3° é indistinguível pela razão dos eixos da elipse
(cos 3° = 0,9986). Ou seja: a moeda não erra pouco, ela erra sem avisar. Por isso
o sistema não a aceita.

**Não serve régua.** Os quatro cantos de uma régua são quase colineares em uma das
direções; a homografia fica malcondicionada nessa direção justamente onde ela
precisaria corrigir a perspectiva.

Identificação x medição
-----------------------
A razão largura/altura do retângulo original é recuperável da foto (é o resultado
clássico de Zhang & He, 2007: a ortogonalidade dos lados fixa a distância focal, e
daí sai a razão). Aqui ela é usada **apenas para identificar/conferir** o objeto.
A medição vem sempre da dimensão declarada no catálogo — nunca de algo estimado a
partir da imagem.

Essa recuperação degenera quando o eixo de inclinação da câmera coincide com um
lado do retângulo (um par de lados continua paralelo na imagem, o ponto de fuga vai
ao infinito). O módulo mede esse malcondicionamento propagando ruído de 1 px pelos
cantos, e se recusa a **identificar** sozinho quando a razão não discrimina — em
vez de chutar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import get_settings
from .marker import DetectedMarker, MarkerDetection, estimate_tilt_deg

REFERENCE_ID_BASE = 900
"""Ids sintéticos dos retângulos de referência (900, 901, …).

Não colidem com os ids do alvo ArUco (0–3 no ``board4``) porque `DICT_4X4_50` só
vai até 49. Servem para que o resto do pipeline — máscara de exclusão, verificação
de ida-e-volta, serialização — trate a referência como trata um marcador."""

_UNIT_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                        dtype=np.float32)


# --------------------------------------------------------------------------- catálogo


@dataclass(frozen=True)
class ReferenceObject:
    """Objeto retangular de dimensão normalizada usado como padrão de comprimento."""

    key: str
    label: str
    width_mm: float
    height_mm: float
    tolerance_mm: float
    """Tolerância dimensional do próprio objeto, segundo a norma.

    Este é um erro que **nenhum** processamento de imagem consegue remover: se a
    folha foi cortada 2 mm maior, toda a escala sai 1 % maior. É reportado ao
    profissional como incerteza de escala."""
    corner_radius_mm: float
    standard: str
    note: str = ""

    @property
    def aspect(self) -> float:
        """Razão largura/altura do objeto físico."""
        return self.width_mm / self.height_mm

    @property
    def model_mm(self) -> np.ndarray:
        """Cantos no referencial do próprio objeto: TL, TR, BR, BL."""
        w, h = self.width_mm, self.height_mm
        return np.array([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]], dtype=np.float64)

    @property
    def scale_tolerance_rel(self) -> float:
        """Incerteza relativa de escala imposta pela tolerância do objeto."""
        return float(self.tolerance_mm / min(self.width_mm, self.height_mm))

    @property
    def span_mm(self) -> float:
        return float(math.hypot(self.width_mm, self.height_mm))


CARD_ID1 = ReferenceObject(
    key="card",
    label="Cartão (crédito/débito/RG novo)",
    width_mm=85.60, height_mm=53.98,
    # ISO/IEC 7810 ID-1 admite 85,47–85,72 mm e 53,92–54,03 mm.
    tolerance_mm=0.13,
    corner_radius_mm=3.18,
    standard="ISO/IEC 7810 ID-1",
    note=("Dimensão muito bem controlada (±0,15 %), mas objeto pequeno: a "
          "calibração fica exata sobre o cartão e extrapolada sobre o pé."),
)

SHEET_A4 = ReferenceObject(
    key="a4",
    label="Folha A4",
    width_mm=210.0, height_mm=297.0,
    # ISO 216: ±2 mm para dimensões entre 150 e 600 mm.
    tolerance_mm=2.0,
    corner_radius_mm=0.0,
    standard="ISO 216",
    note=("Cobre bem a área de apoio, mas o corte do papel tem tolerância de "
          "±2 mm — quase 1 % de incerteza de escala, que nenhum algoritmo remove."),
)

SHEET_A5 = ReferenceObject(
    key="a5",
    label="Folha A5",
    width_mm=148.0, height_mm=210.0,
    tolerance_mm=2.0,
    corner_radius_mm=0.0,
    standard="ISO 216",
    note="Meio-termo entre o cartão e a folha A4.",
)

CATALOGUE: dict[str, ReferenceObject] = {
    r.key: r for r in (CARD_ID1, SHEET_A4, SHEET_A5)
}


def parse_custom_reference(spec: str) -> ReferenceObject:
    """``"85.6x53.98"`` ou ``"85.6x53.98@0.1"`` (tolerância) → objeto de referência.

    Para uma placa cortada sob medida. As dimensões devem ser **medidas com
    paquímetro**, não copiadas do desenho: elas passam a definir a métrica de todo
    o sistema.
    """
    text = spec.strip().lower().replace(",", ".")
    tol = 0.5
    if "@" in text:
        text, tol_text = text.split("@", 1)
        tol = float(tol_text)
    parts = text.replace("×", "x").split("x")
    if len(parts) != 2:
        raise ValueError(f"referência personalizada inválida: {spec!r} (use LxA em mm)")
    w, h = float(parts[0]), float(parts[1])
    if not (10.0 <= w <= 1000.0 and 10.0 <= h <= 1000.0):
        raise ValueError("dimensões de referência fora da faixa plausível (10–1000 mm)")
    return ReferenceObject(
        key="custom", label=f"Retângulo personalizado {w:g} × {h:g} mm",
        width_mm=w, height_mm=h, tolerance_mm=tol, corner_radius_mm=0.0,
        standard="declarado pelo operador",
        note="Dimensões declaradas manualmente — confira com paquímetro.",
    )


def resolve_reference(key: str) -> ReferenceObject:
    k = (key or "").strip().lower()
    if k in CATALOGUE:
        return CATALOGUE[k]
    if "x" in k or "×" in k:
        return parse_custom_reference(k)
    if k == "custom":
        settings = get_settings()
        if not settings.reference_custom_mm:
            raise ValueError("PEDIGRAFIA_REFERENCE_CUSTOM_MM não definido")
        return parse_custom_reference(settings.reference_custom_mm)
    raise ValueError(f"referência desconhecida: {key!r}")


def catalogue_for(spec: str | None = None) -> list[ReferenceObject]:
    """Objetos que a detecção deve considerar, conforme a configuração."""
    settings = get_settings()
    value = (spec if spec is not None else settings.reference_object).strip().lower()
    if value in ("", "auto"):
        items = list(CATALOGUE.values())
        if settings.reference_custom_mm:
            items.append(parse_custom_reference(settings.reference_custom_mm))
        return items
    if value in ("off", "none", "aruco"):
        return []
    return [resolve_reference(value)]


# ------------------------------------------------------- razão de aspecto projetiva


def rectangle_aspect(quad_px: np.ndarray, image_size: tuple[int, int]
                     ) -> tuple[float, float]:
    """Razão largura/altura do retângulo original e a distância focal implícita.

    ``quad_px`` são os cantos em ordem cíclica; a "largura" é o lado ``q0→q1``.

    Sejam ``g1, g2`` as duas primeiras colunas da homografia que leva o quadrado
    unitário ao quadrilátero observado, com a origem no ponto principal. Se o
    objeto é um retângulo, ``K⁻¹g1`` e ``K⁻¹g2`` são as direções ortogonais dos seus
    lados, com módulos proporcionais a largura e altura. Daí:

    * ortogonalidade ``g1ᵀ ω g2 = 0`` fixa ``f`` (``ω = diag(1/f², 1/f², 1)``);
    * ``L/A = ‖K⁻¹g1‖ / ‖K⁻¹g2‖``.

    Devolve ``(razão, f_px)``; ``(nan, nan)`` quando a configuração é degenerada
    (vista frontal ou eixo de inclinação paralelo a um lado — nesses casos um par
    de lados continua paralelo na imagem e ``f`` não é observável).
    """
    w, h = image_size
    q = (np.asarray(quad_px, dtype=np.float64) - np.array([w / 2.0, h / 2.0]))
    try:
        G = cv2.getPerspectiveTransform(_UNIT_SQUARE,
                                        q.astype(np.float32)).astype(np.float64)
    except cv2.error:
        return float("nan"), float("nan")

    g1, g2 = G[:, 0], G[:, 1]
    scale = max(float(np.linalg.norm(g1[:2])), float(np.linalg.norm(g2[:2])), 1e-9)
    # Condicionamento: g1[2] e g2[2] são as componentes que carregam a perspectiva.
    # Quando qualquer uma some, o par de lados correspondente continua paralelo na
    # imagem e não há informação de fuga para extrair.
    if abs(g1[2] * g2[2]) * scale < 1e-7:
        return float("nan"), float("nan")

    f2 = -(g1[0] * g2[0] + g1[1] * g2[1]) / (g1[2] * g2[2])
    if not np.isfinite(f2) or f2 <= 0.0:
        return float("nan"), float("nan")

    n1 = (g1[0] ** 2 + g1[1] ** 2) / f2 + g1[2] ** 2
    n2 = (g2[0] ** 2 + g2[1] ** 2) / f2 + g2[2] ** 2
    if n2 <= 0.0 or not np.isfinite(n1 / n2):
        return float("nan"), float("nan")
    return float(math.sqrt(n1 / n2)), float(math.sqrt(f2))


def aspect_with_uncertainty(quad_px: np.ndarray, image_size: tuple[int, int],
                            jitter_px: float = 1.0, trials: int = 48,
                            seed: int = 20240517) -> tuple[float, float, float]:
    """``(razão, desvio-padrão da razão, f_px)`` propagando ruído nos cantos.

    O desvio-padrão é o que decide se a razão pode ser usada para **identificar** o
    objeto. Perto da degenerescência ele explode, e o sistema passa a exigir que o
    operador declare qual referência usou em vez de adivinhar.
    """
    base, focal = rectangle_aspect(quad_px, image_size)
    if not np.isfinite(base):
        return float("nan"), float("inf"), float("nan")

    rng = np.random.default_rng(seed)
    quad = np.asarray(quad_px, dtype=np.float64)
    samples: list[float] = []
    for _ in range(trials):
        noisy = quad + rng.normal(0.0, jitter_px, quad.shape)
        ratio, _ = rectangle_aspect(noisy, image_size)
        if np.isfinite(ratio):
            samples.append(ratio)
    if len(samples) < trials // 2:
        # Metade das perturbações caiu na degenerescência: a razão não é utilizável.
        return base, float("inf"), focal
    return base, float(np.std(samples)), focal


# ------------------------------------------------------------- detecção de retângulos


def _order_quad(quad: np.ndarray) -> np.ndarray:
    """Ordena os cantos no sentido horário da imagem (y para baixo), começando no
    canto mais próximo da origem — mesma convenção de ``cv2.aruco``."""
    c = quad.mean(axis=0)
    ang = np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0])
    q = quad[np.argsort(ang)]
    start = int(np.argmin(q[:, 0] + q[:, 1]))
    return np.roll(q, -start, axis=0)


def _binarizations(gray: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    med = float(np.median(gray))
    lo = int(max(0.0, 0.62 * med))
    hi = int(min(255.0, 1.30 * med))
    edges = cv2.Canny(gray, lo, hi, L2gradient=True)
    out.append(cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1))

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    out.append(otsu)
    out.append(cv2.bitwise_not(otsu))
    for block in (31, 71):
        ad = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, block, 7)
        out.append(ad)
        out.append(cv2.bitwise_not(ad))
    return out


def _quad_candidates(bgr: np.ndarray, max_side: int = 1400) -> list[np.ndarray]:
    """Quadriláteros convexos plausíveis, em coordenadas da imagem original."""
    h, w = bgr.shape[:2]
    scale = 1.0
    work = bgr
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        work = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 45, 45)
    area_img = float(gray.shape[0] * gray.shape[1])
    min_area = 0.0012 * area_img
    max_area = 0.62 * area_img

    found: list[np.ndarray] = []
    seen: set[tuple] = set()
    for binary in _binarizations(gray):
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = abs(cv2.contourArea(cnt))
            if area < min_area or area > max_area:
                continue
            peri = cv2.arcLength(cnt, True)
            if peri <= 0:
                continue
            for eps in (0.010, 0.020, 0.032, 0.045):
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                if len(approx) != 4 or not cv2.isContourConvex(approx):
                    continue
                quad = _order_quad(approx.reshape(4, 2).astype(np.float64))
                if not _plausible_quad(quad):
                    continue
                key = (int(round(quad[:, 0].mean() / 8.0)),
                       int(round(quad[:, 1].mean() / 8.0)),
                       int(round(math.sqrt(max(area, 1.0)) / 6.0)))
                if key in seen:
                    break
                seen.add(key)
                found.append(quad / scale)
                break
    return found


def _plausible_quad(quad: np.ndarray) -> bool:
    """Descarta quadriláteros que nenhum retângulo plano poderia gerar."""
    sides = [float(np.linalg.norm(quad[(i + 1) % 4] - quad[i])) for i in range(4)]
    if min(sides) < 18.0:
        return False
    for i in range(4):
        a = quad[(i - 1) % 4] - quad[i]
        b = quad[(i + 1) % 4] - quad[i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            return False
        ang = math.degrees(math.acos(
            float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))
        # Um retângulo fotografado com inclinação ≤ 45° não produz ângulos assim.
        if ang < 42.0 or ang > 138.0:
            return False
    # Lados opostos de um retângulo não podem diferir arbitrariamente sob perspectiva.
    for i in (0, 1):
        a, b = sides[i], sides[i + 2]
        if max(a, b) / max(min(a, b), 1e-6) > 2.6:
            return False
    return True


def _fit_edge_line(gray: np.ndarray, p: np.ndarray, q: np.ndarray,
                   trim: float, samples: int = 72
                   ) -> tuple[np.ndarray, float] | None:
    """Reta subpixel da aresta ``p→q``, por máximo de gradiente perpendicular.

    Ajustar a **reta** e depois intersectar é o que recupera o canto ideal de um
    cartão, cujos cantos físicos são arredondados (r = 3,18 mm no ID-1) e não
    existem como ponto na imagem. ``trim`` descarta as extremidades da aresta,
    onde estão o arredondamento e a influência dos cantos vizinhos.
    """
    d = q - p
    length = float(np.linalg.norm(d))
    if length < 24.0:
        return None
    u = d / length
    n = np.array([-u[1], u[0]], dtype=np.float64)
    band = float(np.clip(length * 0.035, 3.0, 16.0))

    ts = np.linspace(trim, 1.0 - trim, samples)
    offs = np.arange(-band, band + 1e-9, 0.5)
    base = p[None, :] + ts[:, None] * d[None, :]
    pts = base[:, None, :] + offs[None, :, None] * n[None, None, :]

    prof = cv2.remap(gray, pts[:, :, 0].astype(np.float32),
                     pts[:, :, 1].astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE).astype(np.float64)
    grad = np.abs(np.gradient(prof, axis=1))
    k = np.argmax(grad, axis=1)
    rows = np.arange(grad.shape[0])
    peak = grad[rows, k]

    interior = (k > 0) & (k < grad.shape[1] - 1) & (peak > 1.5)
    if np.count_nonzero(interior) < 8:
        return None
    ki = np.clip(k, 1, grad.shape[1] - 2)
    y0, y1, y2 = grad[rows, ki - 1], grad[rows, ki], grad[rows, ki + 1]
    denom = (y0 - 2.0 * y1 + y2)
    delta = np.where(np.abs(denom) > 1e-9, 0.5 * (y0 - y2) / np.where(
        np.abs(denom) > 1e-9, denom, 1.0), 0.0)
    delta = np.clip(delta, -1.0, 1.0)
    step = float(offs[1] - offs[0])
    s = offs[ki] + delta * step

    edge = base + s[:, None] * n[None, :]
    edge = edge[interior]

    line = _total_least_squares_line(edge)
    if line is None:
        return None
    # Uma passada de rejeição de outliers (reflexo, sombra, textura do vidro).
    resid = np.abs(edge @ line[:2] - line[2])
    keep = resid < max(1.0, 2.5 * float(np.median(resid)) + 0.35)
    if np.count_nonzero(keep) >= 8:
        refit = _total_least_squares_line(edge[keep])
        if refit is not None:
            line = refit
            resid = np.abs(edge[keep] @ line[:2] - line[2])
    return line, float(np.sqrt(np.mean(resid ** 2)))


def _total_least_squares_line(points: np.ndarray) -> np.ndarray | None:
    """Reta ``[nx, ny, c]`` com ``nx·x + ny·y = c`` e ``‖n‖ = 1``."""
    if len(points) < 3:
        return None
    centroid = points.mean(axis=0)
    centred = points - centroid
    try:
        _, _, vh = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    normal = vh[1]
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        return None
    normal = normal / norm
    return np.array([normal[0], normal[1], float(np.dot(normal, centroid))])


def _intersect(l1: np.ndarray, l2: np.ndarray) -> np.ndarray | None:
    A = np.array([l1[:2], l2[:2]], dtype=np.float64)
    det = float(np.linalg.det(A))
    if abs(det) < 1e-9:
        return None
    return np.linalg.solve(A, np.array([l1[2], l2[2]], dtype=np.float64))


def refine_quad(gray: np.ndarray, quad: np.ndarray,
                corner_radius_px: float = 0.0) -> tuple[np.ndarray, float]:
    """Cantos subpixel por interseção das quatro retas das arestas."""
    sides = [float(np.linalg.norm(quad[(i + 1) % 4] - quad[i])) for i in range(4)]
    lines: list[np.ndarray | None] = []
    rms: list[float] = []
    for i in range(4):
        # Descarta pelo menos o dobro do raio de arredondamento em cada ponta.
        trim = float(np.clip(
            max(0.10, 2.2 * corner_radius_px / max(sides[i], 1.0)), 0.10, 0.33))
        got = _fit_edge_line(gray, quad[i], quad[(i + 1) % 4], trim)
        lines.append(got[0] if got else None)
        rms.append(got[1] if got else float("nan"))

    if any(line is None for line in lines):
        return quad, float("nan")

    refined = []
    for i in range(4):
        pt = _intersect(lines[(i - 1) % 4], lines[i])
        refined.append(quad[i] if pt is None else pt)
    refined_arr = np.array(refined, dtype=np.float64)

    # Segurança: o refino corrige fração de pixel; deslocamento grande é sintoma de
    # aresta ajustada em outra borda (moldura, sombra) e é descartado.
    if float(np.max(np.linalg.norm(refined_arr - quad, axis=1))) > 0.10 * min(sides):
        return quad, float(np.nanmean(rms))
    return refined_arr, float(np.nanmean(rms))


# --------------------------------------------------------------------- resultado


@dataclass
class ReferenceCandidate:
    reference: ReferenceObject
    corners_px: np.ndarray
    """(4, 2) já reordenados: mapeiam para TL, TR, BR, BL do modelo em mm."""
    aspect_measured: float
    aspect_sigma: float
    aspect_error_rel: float
    focal_px: float
    edge_rms_px: float
    area_px: float
    touches_border: bool
    identifiable: bool
    """``True`` quando a razão de aspecto é bem-condicionada o bastante para
    distinguir este objeto dos demais do catálogo."""
    confidence: float

    @property
    def side_lengths_px(self) -> tuple[float, ...]:
        return tuple(float(np.linalg.norm(self.corners_px[(i + 1) % 4]
                                          - self.corners_px[i])) for i in range(4))


@dataclass
class ReferenceDetection:
    detection: MarkerDetection
    reference: ReferenceObject | None = None
    candidates: list[ReferenceCandidate] = field(default_factory=list)
    ambiguous: bool = False
    reason: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.detection.found


def _assign_orientation(quad: np.ndarray, ref: ReferenceObject,
                        image_size: tuple[int, int]
                        ) -> tuple[np.ndarray, float, float, float]:
    """Escolhe qual par de lados é a largura declarada.

    Devolve ``(quad_reordenado, razão, sigma, f_px)``. A escolha entre as duas
    hipóteses é robusta mesmo com a razão ruidosa: cartão em pé e cartão deitado
    diferem por um fator 2,5, muito acima do ruído.
    """
    best = None
    for shift in (0, 1):
        rotated = np.roll(quad, -shift, axis=0)
        ratio, sigma, focal = aspect_with_uncertainty(rotated, image_size)
        if not np.isfinite(ratio):
            # Sem razão projetiva utilizável: usa a razão observada na imagem, que
            # na degenerescência é próxima da verdadeira (vista quase frontal).
            sides = [float(np.linalg.norm(rotated[(i + 1) % 4] - rotated[i]))
                     for i in range(4)]
            ratio = (sides[0] + sides[2]) / max(sides[1] + sides[3], 1e-9)
            sigma = float("inf")
            focal = float("nan")
        err = abs(ratio - ref.aspect) / ref.aspect
        if best is None or err < best[0]:
            best = (err, rotated, ratio, sigma, focal)
    assert best is not None
    return best[1], best[2], best[3], best[4]


def _touches_frame(quad: np.ndarray, width: int, height: int,
                   margin: float = 2.0) -> bool:
    return bool(
        np.any(quad[:, 0] < margin) or np.any(quad[:, 1] < margin)
        or np.any(quad[:, 0] > width - 1 - margin)
        or np.any(quad[:, 1] > height - 1 - margin)
    )


def _discrimination_gap(ref: ReferenceObject,
                        catalogue: list[ReferenceObject]) -> float:
    """Menor separação relativa entre a razão deste objeto e a dos demais.

    Considera as **duas** orientações de cada concorrente: um objeto deitado e o
    mesmo objeto em pé produzem razões recíprocas, e a foto não diz qual é qual.
    Por isso a folha A4 e a A5 são indistinguíveis por forma — toda a série ISO A
    compartilha a razão √2 — e o sistema precisa perguntar em vez de adivinhar.
    """
    gaps = [min(abs(ref.aspect - other.aspect),
                abs(ref.aspect - 1.0 / other.aspect)) / ref.aspect
            for other in catalogue if other.key != ref.key]
    return min(gaps) if gaps else float("inf")


def detect_reference(bgr: np.ndarray, *, declared: str | None = None,
                     max_objects: int = 4) -> ReferenceDetection:
    """Localiza objeto(s) de referência e devolve uma ``MarkerDetection`` equivalente.

    ``declared`` é a chave do catálogo informada pelo operador. Quando ausente, o
    módulo tenta identificar pela razão de aspecto e **recusa** se a razão não
    discriminar — adivinhar entre um cartão e uma folha A4 erraria a escala por um
    fator 2,45 sem deixar rastro.
    """
    settings = get_settings()
    height, width = bgr.shape[:2]
    enabled = catalogue_for()
    # `PEDIGRAFIA_REFERENCE_OBJECT=off` desliga o caminho inteiro, inclusive para
    # quem declarou explicitamente: é uma decisão da implantação, não do operador.
    catalogue = ([resolve_reference(declared)] if declared and enabled else enabled)
    if not catalogue:
        return ReferenceDetection(
            detection=MarkerDetection(
                found=False, corners_px=np.zeros((4, 2)),
                reason=("Calibração por objeto de referência está desativada nesta "
                        "instalação. Use o alvo impresso.")),
            reason="disabled")

    quads = _quad_candidates(bgr)
    if not quads:
        return ReferenceDetection(
            detection=MarkerDetection(
                found=False, corners_px=np.zeros((4, 2)),
                reason="Nenhum retângulo de referência reconhecido na foto."),
            reason="no_quad")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    full_cat = catalogue_for("auto")
    candidates: list[ReferenceCandidate] = []

    for quad in quads:
        first_side_px = float(np.linalg.norm(quad[1] - quad[0]))
        for ref in catalogue:
            # Estimativa grosseira do arredondamento em pixels: qual lado é a largura
            # ainda não foi decidido, então isto pode errar pelo fator da razão de
            # aspecto. Serve só para dimensionar o `trim` das arestas, que é limitado
            # a [0,10 ; 0,33] — a imprecisão não sai daí.
            radius_px = ref.corner_radius_mm * first_side_px / max(ref.width_mm, 1e-6)
            refined, rms = refine_quad(gray, quad, radius_px)
            ordered, ratio, sigma, focal = _assign_orientation(
                refined, ref, (width, height))
            err = abs(ratio - ref.aspect) / ref.aspect

            # Tolerância honesta: três desvios-padrão da própria medida, com piso
            # que absorve erro de localização de canto e **teto**, porque uma
            # incerteza enorme não pode virar licença para aceitar qualquer coisa.
            # Quando o sigma estoura, o quadrilátero ainda passa pelo teto (é só um
            # filtro de sanidade), mas `identifiable` fica falso e o modo automático
            # se recusa a escolher sozinho.
            tol = float(np.clip(3.0 * sigma, 0.025, 0.12)) if np.isfinite(sigma) \
                else 0.12
            if err > tol:
                continue
            # Distância focal implausível denuncia um quadrilátero que não é a
            # projeção de um retângulo (é a única checagem NÃO circular disponível).
            if np.isfinite(focal):
                fr = focal / max(width, height)
                if not (0.35 <= fr <= 6.0):
                    continue

            gap = _discrimination_gap(ref, full_cat)
            identifiable = bool(np.isfinite(sigma) and 3.0 * sigma < 0.5 * gap)

            area = float(abs(cv2.contourArea(ordered.astype(np.float32))))
            touches = _touches_frame(ordered, width, height)
            sides = [float(np.linalg.norm(ordered[(i + 1) % 4] - ordered[i]))
                     for i in range(4)]
            conf = (0.40 * float(np.clip(1.0 - err / max(tol, 1e-6), 0.0, 1.0))
                    + 0.30 * float(np.clip(min(sides) / 90.0, 0.0, 1.0))
                    + 0.30 * float(np.clip(1.0 - (rms if np.isfinite(rms) else 1.0)
                                           / 1.2, 0.0, 1.0)))
            if touches:
                conf *= 0.45
            candidates.append(ReferenceCandidate(
                reference=ref, corners_px=ordered, aspect_measured=ratio,
                aspect_sigma=sigma, aspect_error_rel=err, focal_px=focal,
                edge_rms_px=rms, area_px=area, touches_border=touches,
                identifiable=identifiable, confidence=float(np.clip(conf, 0.0, 1.0)),
            ))

    if not candidates:
        names = ", ".join(r.label for r in catalogue)
        return ReferenceDetection(
            detection=MarkerDetection(
                found=False, corners_px=np.zeros((4, 2)),
                reason=(f"Nenhum retângulo compatível com {names} foi encontrado. "
                        f"Deixe o objeto inteiro, plano e bem visível na foto.")),
            reason="no_match")

    warnings: list[str] = []
    if declared is None:
        # Identificar sozinho só é aceitável quando a forma **prova** de qual objeto
        # se trata. Confundir um cartão com uma folha A4 erra a escala em 2,45× e
        # produz um molde coerente, bonito e completamente errado — exatamente o
        # tipo de falha silenciosa que este sistema existe para não cometer.
        keys = {c.reference.key for c in candidates}
        best = max(candidates, key=lambda c: c.confidence)
        if len(keys) > 1 or not best.identifiable:
            options = ", ".join(sorted(r.key for r in catalogue))
            return ReferenceDetection(
                detection=MarkerDetection(
                    found=False, corners_px=np.zeros((4, 2)),
                    reason=("Encontrei um retângulo, mas a forma não prova qual "
                            "objeto é: a razão largura/altura observada é "
                            "compatível com mais de uma referência. Informe qual "
                            f"foi usada ({options}). Confundir cartão com folha A4 "
                            "erraria a escala em 2,45× sem nenhum outro sintoma.")),
                candidates=candidates, ambiguous=True, reason="ambiguous")
        chosen_ref = best.reference
    else:
        chosen_ref = resolve_reference(declared)
    same = [c for c in candidates if c.reference.key == chosen_ref.key]
    same.sort(key=lambda c: (-c.confidence, -c.area_px))

    # Objetos distintos: descarta quadriláteros que se sobrepõem (contorno interno
    # e externo da mesma borda aparecem como dois candidatos).
    picked: list[ReferenceCandidate] = []
    for cand in same:
        if any(_overlaps(cand.corners_px, other.corners_px) for other in picked):
            continue
        picked.append(cand)
        if len(picked) >= max_objects:
            break

    if len(picked) > 1:
        warnings.append(
            f"{len(picked)} objetos de referência detectados: a calibração usa "
            f"todos, o que reduz a extrapolação sobre os pés.")
        # O primeiro objeto ancora o referencial em mm. Qual deles é isso não afeta a
        # exatidão (é apenas a fixação do calibre), mas afeta a reprodutibilidade: com
        # a ordem vinda da confiança, duas execuções da mesma foto poderiam devolver
        # frames diferentes. Ordem espacial estável resolve.
        picked.sort(key=lambda c: (round(float(c.corners_px[:, 1].mean()), 1),
                                   round(float(c.corners_px[:, 0].mean()), 1)))

    markers = tuple(
        DetectedMarker(marker_id=REFERENCE_ID_BASE + i, corners_px=c.corners_px,
                       side_lengths_px=c.side_lengths_px,
                       touches_border=c.touches_border)
        for i, c in enumerate(picked)
    )
    primary = picked[0]
    sides = primary.side_lengths_px
    src_px_per_mm = float(np.mean([
        sides[0] / chosen_ref.width_mm, sides[2] / chosen_ref.width_mm,
        sides[1] / chosen_ref.height_mm, sides[3] / chosen_ref.height_mm,
    ]))
    # Um retângulo não tem "lados iguais": o skew relevante é a divergência entre
    # lados OPOSTOS, que é o que a perspectiva produz.
    skew = max(abs(sides[0] - sides[2]) / max(sides[0] + sides[2], 1e-9),
               abs(sides[1] - sides[3]) / max(sides[1] + sides[3], 1e-9)) * 2.0
    angle_dev = _max_angle_deviation(primary.corners_px)
    tilt = estimate_tilt_deg(primary.corners_px, bgr.shape[:2],
                             settings.marker_size_mm,
                             model_mm=chosen_ref.model_mm)

    detection = MarkerDetection(
        found=True, corners_px=primary.corners_px,
        marker_id=REFERENCE_ID_BASE, dictionary=f"reference:{chosen_ref.key}",
        side_lengths_px=sides, src_px_per_mm=src_px_per_mm, skew=float(skew),
        angle_deviation_deg=angle_dev, tilt_deg=tilt,
        confidence=primary.confidence, touches_border=primary.touches_border,
        markers=markers, source="reference",
    )
    return ReferenceDetection(detection=detection, reference=chosen_ref,
                              candidates=picked, warnings=warnings)


def reference_target(ref: ReferenceObject,
                     poses: list[tuple[float, float, float]] | None = None):
    """Alvo de calibração equivalente para um ou mais objetos de referência.

    ``poses`` vem do ajuste conjunto (:func:`app.calibration.fit.fit_free_rectangles`)
    e traz a posição resolvida de cada objeto no referencial do primeiro. Sem ele,
    assume um único objeto na origem.
    """
    from .target import CalibrationTarget, MarkerPlacement

    placements = poses or [(0.0, 0.0, 0.0)]
    markers = tuple(
        MarkerPlacement(REFERENCE_ID_BASE + i, (float(x), float(y)),
                        ref.width_mm, ref.height_mm, float(deg))
        for i, (x, y, deg) in enumerate(placements)
    )
    count = len(markers)
    plural = "objetos" if count > 1 else "objeto"
    return CalibrationTarget(
        name=f"reference:{ref.key}" + (f"×{count}" if count > 1 else ""),
        dictionary="", markers=markers, kind="reference", reference=ref,
        description=(f"{count} {plural} de referência — {ref.label}, "
                     f"{ref.width_mm:g} × {ref.height_mm:g} mm ({ref.standard})."),
    )


def verify_reference_round_trip(rectification, target) -> tuple[list[float], float]:
    """Re-mede cada objeto de referência na imagem retificada.

    Confere lados e diagonais contra as dimensões declaradas. O que este teste
    valida é a **retificação** (homografia, reamostragem, quantização da origem) e,
    com dois objetos, a consistência do ajuste conjunto; ele **não** valida a
    identificação do objeto — se o operador declarou "cartão" e fotografou outra
    coisa retangular, a medida sai coerente e errada. Essa é a diferença honesta
    em relação ao marcador ArUco, cujo código é verificável em si.

    A busca é feita na vizinhança esperada de cada objeto. Sem isso, o próprio
    contorno da região válida da imagem retificada — que é um quadrilátero, porque
    é a projeção do sensor — competiria como candidato.
    """
    ref = target.reference
    if ref is None:
        return [], float("inf")

    image = rectification.image
    height, width = image.shape[:2]
    diagonal = math.hypot(ref.width_mm, ref.height_mm)
    measured: list[float] = []
    errors: list[float] = []

    for placement in target.markers:
        expected_px = np.asarray(
            rectification.mm_to_rect_px(placement.corners_mm()), dtype=np.float64)
        found_px = _redetect_near(image, expected_px, ref, (width, height))
        if found_px is None:
            return [], float("inf")

        corners_mm = np.asarray(rectification.rect_px_to_mm(found_px),
                                dtype=np.float64).reshape(4, 2)
        expected = [ref.width_mm, ref.height_mm, ref.width_mm, ref.height_mm]
        for i in range(4):
            side = float(np.linalg.norm(corners_mm[(i + 1) % 4] - corners_mm[i]))
            measured.append(side)
            errors.append(abs(side - expected[i]))
        for a, b in ((0, 2), (1, 3)):
            got = float(np.linalg.norm(corners_mm[a] - corners_mm[b]))
            measured.append(got)
            errors.append(abs(got - diagonal))

    if not errors:
        return [], float("inf")
    return measured, float(max(errors))


def _redetect_near(image: np.ndarray, expected_px: np.ndarray, ref: ReferenceObject,
                   size: tuple[int, int]) -> np.ndarray | None:
    """Reencontra o objeto perto de onde ele deveria estar, na imagem retificada.

    Primeiro tenta um recorte ao redor da posição esperada — é o caminho barato e
    imune ao contorno da região válida da retificada, que é um quadrilátero e
    competiria como candidato. Se o recorte não resolver (tipicamente porque a folga
    até a borda da janela retificada é menor que a margem pedida, e aí o objeto
    encosta na borda do recorte), repete na imagem inteira e escolhe o candidato mais
    próximo da posição esperada.
    """
    width, height = size
    side_px = float(np.min(np.linalg.norm(
        np.roll(expected_px, -1, axis=0) - expected_px, axis=1)))
    centre = expected_px.mean(axis=0)

    margin = 0.30 * side_px
    lo = np.maximum(expected_px.min(axis=0) - margin, 0.0).astype(int)
    hi = np.minimum(expected_px.max(axis=0) + margin,
                    [width - 1, height - 1]).astype(int)
    if hi[0] - lo[0] >= 40 and hi[1] - lo[1] >= 40:
        det = detect_reference(image[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1],
                               declared=ref.key, max_objects=1)
        if det.found:
            return det.detection.corners_px + lo

    det = detect_reference(image, declared=ref.key, max_objects=4)
    if not det.found:
        return None
    near = [c for c in det.candidates
            if float(np.linalg.norm(c.corners_px.mean(axis=0) - centre)) < 0.5 * side_px]
    if not near:
        return None
    return min(near, key=lambda c: float(
        np.linalg.norm(c.corners_px.mean(axis=0) - centre))).corners_px


def _overlaps(a: np.ndarray, b: np.ndarray) -> bool:
    """Dois quadriláteros compartilham a mesma região da imagem?"""
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    ra = float(np.mean(np.linalg.norm(a - ca, axis=1)))
    rb = float(np.mean(np.linalg.norm(b - cb, axis=1)))
    return float(np.linalg.norm(ca - cb)) < 0.75 * (ra + rb)


def _max_angle_deviation(quad: np.ndarray) -> float:
    worst = 0.0
    for i in range(4):
        a = quad[(i - 1) % 4] - quad[i]
        b = quad[(i + 1) % 4] - quad[i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 90.0
        cos = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
        worst = max(worst, abs(math.degrees(math.acos(cos)) - 90.0))
    return float(worst)

"""Releitura programática do PDF gerado — verificação metrológica independente.

Decodifica o *content stream*, interpreta a pilha de estados gráficos (``q``/``Q``),
a matriz corrente (``cm``) e a cor de traço, e devolve os caminhos já em
**milímetros de página**. É assim que o teste comprova que 265,0 mm de pé viraram
265,0 mm de papel — sem confiar no código que escreveu o arquivo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from pypdf import PdfReader

from ..config import A4_HEIGHT_PT, A4_WIDTH_PT, PT_TO_MM

_TOKEN = re.compile(rb"""
    (?P<num>[-+]?\d*\.?\d+)
  | (?P<name>/[^\s/\[\]()<>{}]+)
  | (?P<str>\((?:\\.|[^()\\])*\))
  | (?P<op>[A-Za-z'"*]+)
  | (?P<arr>[\[\]])
""", re.VERBOSE)


@dataclass
class PdfPath:
    points_mm: np.ndarray
    stroke_rgb: tuple[float, float, float] | None
    closed: bool
    subpath_count: int = 1


@dataclass
class PdfPageInfo:
    media_box_pt: tuple[float, float, float, float]
    paths: list[PdfPath] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def media_box_mm(self) -> tuple[float, float]:
        _, _, w, h = self.media_box_pt
        return (w * PT_TO_MM, h * PT_TO_MM)

    def is_a4(self, tol_pt: float = 0.05) -> bool:
        x0, y0, x1, y1 = self.media_box_pt
        return (abs(x0) < tol_pt and abs(y0) < tol_pt
                and abs(x1 - A4_WIDTH_PT) < tol_pt
                and abs(y1 - A4_HEIGHT_PT) < tol_pt)

    def paths_with_color(self, rgb, tol: float = 0.006) -> list[PdfPath]:
        out = []
        for p in self.paths:
            if p.stroke_rgb is None:
                continue
            if all(abs(a - b) <= tol for a, b in zip(p.stroke_rgb, rgb)):
                out.append(p)
        return out


def _bezier(p0, p1, p2, p3, n: int = 16) -> list[np.ndarray]:
    ts = np.linspace(0, 1, n + 1)[1:]
    out = []
    for t in ts:
        mt = 1 - t
        out.append(mt ** 3 * p0 + 3 * mt ** 2 * t * p1 + 3 * mt * t ** 2 * p2
                   + t ** 3 * p3)
    return out


def parse_page(reader: PdfReader, index: int = 0) -> PdfPageInfo:
    page = reader.pages[index]
    box = page.mediabox
    media = (float(box.left), float(box.bottom), float(box.right), float(box.top))
    data = page.get_contents().get_data()

    ctm_stack: list[np.ndarray] = []
    ctm = np.eye(3, dtype=np.float64)
    color_stack: list[tuple | None] = []
    stroke_rgb: tuple[float, float, float] | None = None

    operands: list = []
    paths: list[PdfPath] = []
    current: list[np.ndarray] = []
    current_closed = False
    subpaths = 0
    start_pt: np.ndarray | None = None
    last: np.ndarray | None = None

    def apply(p: np.ndarray) -> np.ndarray:
        v = np.array([p[0], p[1], 1.0])
        out = ctm @ v
        return out[:2]

    def flush(stroked: bool) -> None:
        nonlocal current, current_closed, subpaths, start_pt, last
        if stroked and len(current) >= 2:
            pts = np.array(current, dtype=np.float64) * PT_TO_MM
            paths.append(PdfPath(points_mm=pts, stroke_rgb=stroke_rgb,
                                 closed=current_closed,
                                 subpath_count=max(1, subpaths)))
        current = []
        current_closed = False
        subpaths = 0
        start_pt = None
        last = None

    for m in _TOKEN.finditer(data):
        if m.group("num") is not None:
            operands.append(float(m.group("num")))
            continue
        if m.group("name") is not None or m.group("str") is not None \
                or m.group("arr") is not None:
            operands.append(m.group(0))
            continue
        op = m.group("op").decode("latin1")

        if op == "q":
            ctm_stack.append(ctm.copy())
            color_stack.append(stroke_rgb)
        elif op == "Q":
            if ctm_stack:
                ctm = ctm_stack.pop()
            if color_stack:
                stroke_rgb = color_stack.pop()
        elif op == "cm" and len(operands) >= 6:
            a, b, cc, d, e, f = [float(x) for x in operands[-6:]]
            mat = np.array([[a, cc, e], [b, d, f], [0, 0, 1]], dtype=np.float64)
            ctm = ctm @ mat
        elif op == "RG" and len(operands) >= 3:
            stroke_rgb = tuple(float(x) for x in operands[-3:])
        elif op == "G" and len(operands) >= 1:
            g = float(operands[-1])
            stroke_rgb = (g, g, g)
        elif op == "m" and len(operands) >= 2:
            p = apply(np.array([float(operands[-2]), float(operands[-1])]))
            current.append(p)
            start_pt = p
            last = p
            subpaths += 1
        elif op == "l" and len(operands) >= 2:
            p = apply(np.array([float(operands[-2]), float(operands[-1])]))
            current.append(p)
            last = p
        elif op in ("c", "v", "y") and last is not None:
            if op == "c" and len(operands) >= 6:
                raw = [float(x) for x in operands[-6:]]
                p1 = apply(np.array(raw[0:2]))
                p2 = apply(np.array(raw[2:4]))
                p3 = apply(np.array(raw[4:6]))
            elif op == "v" and len(operands) >= 4:
                raw = [float(x) for x in operands[-4:]]
                p1 = last
                p2 = apply(np.array(raw[0:2]))
                p3 = apply(np.array(raw[2:4]))
            elif op == "y" and len(operands) >= 4:
                raw = [float(x) for x in operands[-4:]]
                p1 = apply(np.array(raw[0:2]))
                p3 = apply(np.array(raw[2:4]))
                p2 = p3
            else:
                operands = []
                continue
            current.extend(_bezier(last, p1, p2, p3))
            last = p3
        elif op == "re" and len(operands) >= 4:
            x, y, w, h = [float(v) for v in operands[-4:]]
            corners = [apply(np.array(c)) for c in
                       ((x, y), (x + w, y), (x + w, y + h), (x, y + h))]
            current.extend(corners)
            current_closed = True
            subpaths += 1
            last = corners[0]
        elif op == "h":
            if start_pt is not None:
                current_closed = True
                last = start_pt
        elif op in ("S", "s"):
            if op == "s":
                current_closed = True
            flush(True)
        elif op in ("f", "F", "f*", "B", "B*", "b", "b*"):
            flush(op in ("B", "B*", "b", "b*"))
        elif op == "n":
            flush(False)
        operands = []

    meta = {}
    try:
        info = reader.metadata or {}
        meta = {str(k): str(v) for k, v in info.items()}
    except Exception:
        meta = {}

    return PdfPageInfo(media_box_pt=media, paths=paths, metadata=meta)


def inspect_pdf(data: bytes, index: int = 0) -> PdfPageInfo:
    import io

    return parse_page(PdfReader(io.BytesIO(data)), index)


def path_axis_length_mm(path: PdfPath, direction: np.ndarray | None = None) -> float:
    """Extensão do caminho ao longo de ``direction`` (padrão: eixo maior da página)."""
    pts = path.points_mm
    if direction is None:
        direction = np.array([0.0, 1.0])
    d = np.asarray(direction, dtype=np.float64)
    d = d / max(float(np.linalg.norm(d)), 1e-12)
    proj = pts @ d
    return float(np.max(proj) - np.min(proj))


def path_max_caliper_mm(path: PdfPath) -> float:
    """Diâmetro do caminho — mesma definição de comprimento usada nas medidas.

    Independe de como o pé ficou girado na página, o que torna a verificação do PDF
    imune à convenção de posicionamento.
    """
    from ..geometry.polygon import max_caliper

    length, _, _ = max_caliper(path.points_mm)
    return float(length)


def path_bbox_mm(path: PdfPath) -> tuple[float, float, float, float]:
    pts = path.points_mm
    return (float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()))

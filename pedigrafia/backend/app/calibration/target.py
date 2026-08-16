"""Alvo de calibração: onde cada marcador está, fisicamente, na plataforma.

Por que isto existe
-------------------
Um único marcador de 50 × 50 mm ancora a escala em **um ponto**. A homografia
estimada a partir dele é exata sobre o próprio marcador e vai perdendo exatidão
conforme se afasta: qualquer erro angular residual na localização dos quatro cantos
é multiplicado pelo braço de alavanca.

Isso foi medido (ver ``docs/METROLOGY_CALIBRATION.md``): com a câmera inclinada 10° e
o marcador a ~350 mm dos pododáctilos, o ápice do hálux saiu 2,2 mm deslocado,
enquanto o calcâneo — próximo do marcador — ficou exato em 0,1 mm. Movendo o marcador
para perto dos dedos, o erro migrou para o calcâneo. É extrapolação, não segmentação.

A correção é distribuir pontos de controle pela área útil, de modo que a região dos
pés fique **interpolada** entre marcadores em vez de extrapolada a partir de um só.

Cada alvo é apenas uma tabela de "marcador de id N tem o canto superior esquerdo em
(x, y) mm e lado L mm". Nada além disso define a métrica do sistema.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import get_settings


@dataclass(frozen=True)
class MarkerPlacement:
    """Posição física de um marcador no plano da plataforma."""

    marker_id: int
    origin_mm: tuple[float, float]
    size_mm: float

    def corners_mm(self) -> np.ndarray:
        """Cantos TL, TR, BR, BL — mesma ordem que o detector do OpenCV devolve."""
        x, y = self.origin_mm
        s = self.size_mm
        return np.array([[x, y], [x + s, y], [x + s, y + s], [x, y + s]],
                        dtype=np.float64)


@dataclass(frozen=True)
class CalibrationTarget:
    name: str
    dictionary: str
    markers: tuple[MarkerPlacement, ...] = field(default_factory=tuple)
    description: str = ""

    @property
    def ids(self) -> set[int]:
        return {m.marker_id for m in self.markers}

    @property
    def is_single(self) -> bool:
        return len(self.markers) <= 1

    def placement(self, marker_id: int) -> MarkerPlacement | None:
        for m in self.markers:
            if m.marker_id == marker_id:
                return m
        return None

    def all_corners_mm(self) -> np.ndarray:
        if not self.markers:
            return np.empty((0, 2))
        return np.vstack([m.corners_mm() for m in self.markers])

    def span_mm(self) -> tuple[float, float]:
        pts = self.all_corners_mm()
        if len(pts) == 0:
            return (0.0, 0.0)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        return (float(hi[0] - lo[0]), float(hi[1] - lo[1]))


def single_marker_target(size_mm: float | None = None,
                         dictionary: str | None = None,
                         marker_id: int = 0) -> CalibrationTarget:
    """Alvo de um marcador só, com origem no próprio marcador.

    Mantido para compatibilidade e para quem já tem só um marcador colado. Produz
    resultados corretos perto do marcador e **degrada com a distância** — o sistema
    avisa quando os pés caem fora da região coberta.
    """
    settings = get_settings()
    return CalibrationTarget(
        name="single",
        dictionary=dictionary or settings.marker_dictionary,
        markers=(MarkerPlacement(marker_id, (0.0, 0.0),
                                 size_mm or settings.marker_size_mm),),
        description="Marcador único de 50 × 50 mm (origem no marcador).",
    )


def board4_target(size_mm: float | None = None, dictionary: str | None = None,
                  area_mm: tuple[float, float] = (320.0, 480.0),
                  inset_mm: float = 0.0) -> CalibrationTarget:
    """Quatro marcadores nos cantos de uma área útil — layout **recomendado**.

    Os pés ficam dentro do quadrilátero formado pelos marcadores, portanto a
    homografia interpola em vez de extrapolar. A origem em mm é o canto superior
    esquerdo da área útil.
    """
    settings = get_settings()
    s = size_mm or settings.marker_size_mm
    w, h = area_mm
    return CalibrationTarget(
        name="board4",
        dictionary=dictionary or settings.marker_dictionary,
        markers=(
            MarkerPlacement(0, (inset_mm, inset_mm), s),
            MarkerPlacement(1, (w - inset_mm - s, inset_mm), s),
            MarkerPlacement(2, (w - inset_mm - s, h - inset_mm - s), s),
            MarkerPlacement(3, (inset_mm, h - inset_mm - s), s),
        ),
        description=(f"Quatro marcadores de {s:.0f} mm nos cantos de uma área de "
                     f"{w:.0f} × {h:.0f} mm (ids 0–3)."),
    )


def load_target_from_json(path: str | Path) -> CalibrationTarget:
    """Alvo personalizado, para plataformas com geometria própria.

    Formato::

        {
          "name": "minha-plataforma",
          "dictionary": "DICT_4X4_50",
          "markers": [
            {"id": 0, "originMm": [0, 0],     "sizeMm": 50},
            {"id": 1, "originMm": [310, 0],   "sizeMm": 50}
          ]
        }

    As posições devem ser **medidas com paquímetro na plataforma real**, não copiadas
    do desenho: é esta tabela que define a métrica de todo o sistema.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    markers = tuple(
        MarkerPlacement(int(m["id"]), (float(m["originMm"][0]), float(m["originMm"][1])),
                        float(m["sizeMm"]))
        for m in data["markers"]
    )
    if not markers:
        raise ValueError("alvo de calibração sem marcadores")
    return CalibrationTarget(
        name=str(data.get("name", "custom")),
        dictionary=str(data.get("dictionary", get_settings().marker_dictionary)),
        markers=markers,
        description=str(data.get("description", "Alvo personalizado.")),
    )


def resolve_target(spec: str | None = None) -> CalibrationTarget:
    """Resolve o alvo configurado (``PEDIGRAFIA_TARGET``).

    Valores: ``auto`` (padrão), ``single``, ``board4`` ou o caminho de um JSON.
    ``auto`` decide pelo que aparece na foto — ver :func:`select_target_for_detection`.
    """
    settings = get_settings()
    value = (spec if spec is not None else settings.calibration_target).strip()
    lowered = value.lower()
    if lowered in ("", "auto", "single"):
        return single_marker_target()
    if lowered == "board4":
        return board4_target()
    return load_target_from_json(value)


def select_target_for_detection(detected_ids: list[int],
                                spec: str | None = None) -> CalibrationTarget:
    """Escolhe o alvo em modo ``auto`` a partir dos ids realmente detectados.

    Se pelo menos dois marcadores do layout de tabuleiro aparecerem, usa o
    tabuleiro (interpolação). Caso contrário cai no marcador único, ancorando a
    origem no marcador que foi visto.
    """
    settings = get_settings()
    value = (spec if spec is not None else settings.calibration_target).strip().lower()
    if value not in ("", "auto"):
        return resolve_target(spec)

    board = board4_target()
    matched = [i for i in detected_ids if i in board.ids]
    if len(set(matched)) >= 2:
        return board
    chosen = detected_ids[0] if detected_ids else 0
    return single_marker_target(marker_id=chosen)

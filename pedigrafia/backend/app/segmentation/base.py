"""Interface de segmentação — plugável por design.

Trocar o modelo (clássico → YOLO-seg → U-Net fine-tuned → SAM) não deve exigir
alteração em nenhuma outra camada. O contrato é: dada a imagem **retificada** e o
contexto métrico, devolver uma máscara binária no mesmo raster.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class SegmentationContext:
    """Tudo que um segmentador precisa saber sobre a métrica da cena."""

    px_per_mm: float
    origin_mm: np.ndarray
    valid_mask: np.ndarray
    """255 onde o raster retificado tem conteúdo real da foto original."""
    exclude_mask: np.ndarray
    """255 onde o segmentador NÃO deve procurar pé (marcador, zona de silêncio)."""
    min_area_mm2: float
    max_area_mm2: float

    def mm2_to_px(self, area_mm2: float) -> float:
        return area_mm2 * (self.px_per_mm ** 2)

    def mm_to_px(self, mm: float) -> float:
        return mm * self.px_per_mm


@dataclass
class SegmentationResult:
    mask: np.ndarray                 # uint8 {0, 255}, mesmo tamanho da imagem retificada
    confidence: float
    method: str
    notes: list[str] = field(default_factory=list)
    debug: dict[str, np.ndarray] = field(default_factory=dict)
    score_field: np.ndarray | None = None
    """Campo escalar contínuo em que a máscara foi limiarizada.

    Quando presente, permite refinar a fronteira em precisão sub-pixel em vez de
    aceitar a discretização da máscara. Modelos neurais devem expor aqui o mapa de
    probabilidade."""
    score_threshold: float | None = None


@runtime_checkable
class Segmenter(Protocol):
    name: str

    def segment(self, rect_bgr: np.ndarray,
                ctx: SegmentationContext) -> SegmentationResult:
        ...


class SegmenterUnavailable(RuntimeError):
    """Modelo pedido não pôde ser carregado — o chamador deve cair no clássico."""

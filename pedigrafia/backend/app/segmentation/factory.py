"""Registro e resolução do segmentador ativo.

Trocar de modelo é uma variável de ambiente:

    PEDIGRAFIA_SEGMENTER=onnx PEDIGRAFIA_SEG_MODEL=/models/pes-unet.onnx

Se o modelo pedido não puder ser carregado, o sistema **registra o motivo** e cai no
segmentador clássico — nunca falha silenciosamente nem devolve resultado vazio.
"""

from __future__ import annotations

import logging
from typing import Callable

from ..config import get_settings
from .base import Segmenter, SegmenterUnavailable
from .classical import ClassicalSegmenter

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, Callable[[], Segmenter]] = {}


def register(name: str, builder: Callable[[], Segmenter]) -> None:
    _REGISTRY[name] = builder


def available() -> list[str]:
    return sorted(_REGISTRY)


def _build_onnx() -> Segmenter:
    from .onnx_model import OnnxSegmenter  # import tardio: onnxruntime é opcional

    settings = get_settings()
    return OnnxSegmenter(settings.seg_model_path)


register("classical", ClassicalSegmenter)
register("onnx", _build_onnx)


def get_segmenter(name: str | None = None) -> Segmenter:
    settings = get_settings()
    wanted = (name or settings.segmenter or "classical").strip().lower()
    builder = _REGISTRY.get(wanted)
    if builder is None:
        logger.warning("segmentador '%s' desconhecido; usando clássico", wanted)
        return ClassicalSegmenter()
    try:
        return builder()
    except SegmenterUnavailable as exc:
        logger.warning("segmentador '%s' indisponível (%s); usando clássico",
                       wanted, exc)
    except Exception as exc:  # pragma: no cover - proteção de inicialização
        logger.warning("falha ao construir segmentador '%s' (%s); usando clássico",
                       wanted, exc)
    return ClassicalSegmenter()

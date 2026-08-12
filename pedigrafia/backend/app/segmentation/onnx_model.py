"""Adaptador para qualquer modelo de segmentação exportado em ONNX.

Aceita saídas no formato ``(1, 1, H, W)`` (logits/probabilidade) ou ``(1, C, H, W)``
(multi-classe, canal ``foot_class_index``). Isso cobre U-Net, DeepLab, SegFormer e a
cabeça de segmentação do YOLO após pós-processamento — basta exportar para ONNX.

A implementação vive em ``ml-or-vision/pedigrafia_vision`` para manter o backend livre
de dependências de ML; aqui há apenas a ponte.
"""

from __future__ import annotations

import numpy as np

from .base import SegmentationContext, SegmentationResult, SegmenterUnavailable


class OnnxSegmenter:
    name = "onnx"

    def __init__(self, model_path: str, *, input_size: int = 512,
                 foot_class_index: int = 1, threshold: float = 0.5) -> None:
        if not model_path:
            raise SegmenterUnavailable("PEDIGRAFIA_SEG_MODEL não configurado")
        try:
            from pedigrafia_vision.onnx_runner import OnnxMaskRunner
        except ImportError as exc:  # pragma: no cover - depende de instalação opcional
            raise SegmenterUnavailable(
                "pacote pedigrafia_vision/onnxruntime não instalado") from exc

        self._runner = OnnxMaskRunner(
            model_path, input_size=input_size,
            foot_class_index=foot_class_index, threshold=threshold,
        )
        self.name = f"onnx:{self._runner.model_name}"

    def segment(self, rect_bgr: np.ndarray,
                ctx: SegmentationContext) -> SegmentationResult:
        prob = self._runner.run(rect_bgr)
        mask = (prob >= self._runner.threshold).astype(np.uint8) * 255

        search = np.bitwise_and(ctx.valid_mask, np.bitwise_not(ctx.exclude_mask))
        mask[search == 0] = 0

        inside = mask > 0
        confidence = float(np.mean(prob[inside])) if np.any(inside) else 0.0
        return SegmentationResult(mask=mask, confidence=confidence, method=self.name,
                                  score_field=prob,
                                  score_threshold=self._runner.threshold)

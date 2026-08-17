"""Camada de visão/ML plugável do Pedigrafia Digital."""

from .onnx_runner import OnnxMaskRunner

__all__ = ["OnnxMaskRunner"]

# `dataset`, `train`, `export_onnx` e `benchmark` são importados sob demanda: eles
# dependem de extras (torch) que o backend de produção não precisa ter.

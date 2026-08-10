"""Execução de modelos de segmentação em ONNX.

Isolado do backend de propósito: o servidor de API não depende de nenhuma biblioteca
de ML. Instale este pacote (``pip install -e ./ml-or-vision``) somente quando for usar
um modelo neural.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

try:  # cv2 é a única dependência pesada compartilhada
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore


class OnnxMaskRunner:
    """Executa um modelo de segmentação e devolve um mapa de probabilidade [0, 1]."""

    def __init__(self, model_path: str, *, input_size: int = 512,
                 foot_class_index: int = 1, threshold: float = 0.5,
                 mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
                 std: tuple[float, float, float] = (0.229, 0.224, 0.225)) -> None:
        if cv2 is None:
            raise ImportError("opencv é necessário para o OnnxMaskRunner")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("onnxruntime não instalado") from exc

        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.model_name = os.path.basename(model_path)
        self.input_size = int(input_size)
        self.foot_class_index = int(foot_class_index)
        self.threshold = float(threshold)
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size),
                             interpolation=cv2.INTER_AREA)
        x = resized.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        return np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))

    @staticmethod
    def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / np.sum(e, axis=axis, keepdims=True)

    def _postprocess(self, raw: Any, out_shape: tuple[int, int]) -> np.ndarray:
        arr = np.asarray(raw)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 2:
            prob = arr
        elif arr.shape[0] == 1:
            prob = arr[0]
            if prob.min() < 0.0 or prob.max() > 1.0:
                prob = self._sigmoid(prob)
        else:
            probs = self._softmax(arr, axis=0)
            idx = min(self.foot_class_index, probs.shape[0] - 1)
            prob = probs[idx]
        prob = np.clip(prob.astype(np.float32), 0.0, 1.0)
        return cv2.resize(prob, (out_shape[1], out_shape[0]),
                          interpolation=cv2.INTER_LINEAR)

    def run(self, bgr: np.ndarray) -> np.ndarray:
        x = self._preprocess(bgr)
        outputs = self.session.run(None, {self.input_name: x})
        return self._postprocess(outputs[0], bgr.shape[:2])

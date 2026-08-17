"""Exporta o checkpoint treinado para ONNX — o formato que o backend consome.

    python -m pedigrafia_vision.export_onnx --checkpoint modelo.pt --out modelo.onnx

Depois:

    export PEDIGRAFIA_SEGMENTER=onnx
    export PEDIGRAFIA_SEG_MODEL=/caminho/modelo.onnx

O backend passa a usar o modelo sem nenhuma outra alteração; se ele não carregar, o
sistema registra o motivo e volta ao segmentador clássico.

A saída exportada é a **probabilidade** (após sigmoide), não a máscara binarizada: é
dela que o refino sub-pixel do contorno extrai a isolinha de 50 %.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .train import build_unet


def export(checkpoint: str, out: str, size: int | None = None, opset: int = 17) -> str:
    import torch
    import torch.nn as nn

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    base = int(ckpt.get("base", 32))
    input_size = int(size or ckpt.get("size", 384))

    model = build_unet(base)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    class WithSigmoid(nn.Module):
        """Exporta a PROBABILIDADE, não o logit — contrato do OnnxMaskRunner."""

        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return torch.sigmoid(self.net(x))

    wrapped = WithSigmoid(model)
    dummy = torch.zeros(1, 3, input_size, input_size)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped, dummy, str(out_path),
        input_names=["image"], output_names=["probability"],
        dynamic_axes={"image": {0: "batch"}, "probability": {0: "batch"}},
        opset_version=opset,
    )
    print(f"exportado: {out_path}  (entrada {input_size}×{input_size})")
    print("Confirme o ganho em MILÍMETROS antes de adotar:")
    print(f"  python -m pedigrafia_vision.benchmark --model {out_path}")
    return str(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="modelo.onnx")
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()
    export(args.checkpoint, args.out, args.size, args.opset)


if __name__ == "__main__":
    main()

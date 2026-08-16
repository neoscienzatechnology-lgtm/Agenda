"""Treino da segmentação plantar (U-Net) com perda concentrada na borda.

    python -m pedigrafia_vision.train --data ./dataset --epochs 40 --out modelo.pt

Requer o extra ``train``::

    pip install -e ./ml-or-vision[train]

Três decisões que valem mais que a arquitetura
----------------------------------------------
1. **Rótulos suaves na borda.** A máscara é gravada com anti-aliasing, e o alvo é a
   fração de cobertura, não 0/1. Assim a probabilidade de saída fica proporcional à
   cobertura do pixel, e a isolinha de 50 % coincide com a borda física — que é o que
   o refino sub-pixel do contorno consome em produção.
2. **Peso extra na faixa da borda.** Perda uniforme otimiza área; nós precisamos de
   fronteira. Sem isso o modelo melhora o IoU e piora o milímetro.
3. **Critério de aceite em milímetros**, nunca IoU. Ver ``benchmark.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .dataset import edge_weight_map, load_pairs


def _import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PyTorch não instalado. Use: pip install -e ./ml-or-vision[train]") from exc
    return torch, nn, F


def build_unet(base: int = 32, in_ch: int = 3):
    """U-Net pequena, suficiente para uma classe e viável em CPU."""
    torch, nn, _ = _import_torch()

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1, self.e2, self.e3, self.e4 = (
                block(in_ch, base), block(base, base * 2),
                block(base * 2, base * 4), block(base * 4, base * 8))
            self.pool = nn.MaxPool2d(2)
            self.u3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
            self.d3 = block(base * 8, base * 4)
            self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
            self.d2 = block(base * 4, base * 2)
            self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
            self.d1 = block(base * 2, base)
            self.head = nn.Conv2d(base, 1, 1)

        def forward(self, x):
            c1 = self.e1(x)
            c2 = self.e2(self.pool(c1))
            c3 = self.e3(self.pool(c2))
            c4 = self.e4(self.pool(c3))
            x = self.d3(torch.cat([self.u3(c4), c3], dim=1))
            x = self.d2(torch.cat([self.u2(x), c2], dim=1))
            x = self.d1(torch.cat([self.u1(x), c1], dim=1))
            return self.head(x)

    return UNet()


class FootDataset:
    """Carrega pares, redimensiona e devolve tensores com o mapa de pesos."""

    def __init__(self, root: str | Path, split: str, size: int = 384):
        import cv2

        self.pairs = load_pairs(root, split)
        self.size = size
        self._cv2 = cv2

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        torch, _, _ = _import_torch()
        cv2 = self._cv2
        img_path, mask_path = self.pairs[index]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        # INTER_AREA preserva a fração de cobertura — é isso que faz o rótulo ser
        # suave na borda em vez de 0/1.
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_AREA)

        weights = edge_weight_map(mask)
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / \
            np.array([0.229, 0.224, 0.225], np.float32)
        return (torch.from_numpy(np.transpose(x, (2, 0, 1))),
                torch.from_numpy((mask.astype(np.float32) / 255.0)[None]),
                torch.from_numpy(weights[None]))


def train(data: str, out: str, epochs: int = 40, batch: int = 4, size: int = 384,
          lr: float = 3e-4, base: int = 32) -> dict:
    torch, nn, F = _import_torch()
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet(base).to(device)
    train_loader = DataLoader(FootDataset(data, "train", size), batch_size=batch,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(FootDataset(data, "val", size), batch_size=batch,
                            num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    def loss_fn(logits, target, weights):
        bce = F.binary_cross_entropy_with_logits(logits, target, weight=weights)
        prob = torch.sigmoid(logits)
        num = 2.0 * (prob * target).sum(dim=(1, 2, 3)) + 1.0
        den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1.0
        return bce + (1.0 - (num / den)).mean()

    history = []
    best = float("inf")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y, w in train_loader:
            x, y, w = x.to(device), y.to(device), w.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y, w)
            loss.backward()
            opt.step()
            total += float(loss.item())
        train_loss = total / max(1, len(train_loader))

        model.eval()
        vtotal = 0.0
        with torch.no_grad():
            for x, y, w in val_loader:
                x, y, w = x.to(device), y.to(device), w.to(device)
                vtotal += float(loss_fn(model(x), y, w).item())
        val_loss = vtotal / max(1, len(val_loader))
        sched.step()
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss})
        print(f"época {epoch + 1}/{epochs}  treino={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best:
            best = val_loss
            torch.save({"state_dict": model.state_dict(), "base": base, "size": size},
                       out_path)

    report = {"best_val_loss": best, "epochs": epochs, "history": history,
              "checkpoint": str(out_path)}
    Path(str(out_path) + ".json").write_text(json.dumps(report, indent=1),
                                             encoding="utf-8")
    print("\nATENÇÃO: a perda de validação NÃO é o critério de aceite. "
          "Rode `python -m pedigrafia_vision.benchmark` e compare o erro em MILÍMETROS.")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="modelo.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base", type=int, default=32)
    args = ap.parse_args()
    train(args.data, args.out, args.epochs, args.batch, args.size, args.lr, args.base)


if __name__ == "__main__":
    main()

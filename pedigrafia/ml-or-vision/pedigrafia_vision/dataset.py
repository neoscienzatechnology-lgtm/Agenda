"""Conjuntos de dados para treinar a segmentação plantar.

Duas fontes, mesmo formato de saída:

``synthetic``
    Gerado pelo próprio ``app.synth``: cenas com marcador, iluminação, ruído e
    inclinação variáveis, e **máscara exata por construção**. Serve para pré-treino e
    para verificar que o arcabouço funciona, não para prometer desempenho em foto real.

``annotated``
    Fotos reais anotadas. As máscaras devem ser desenhadas **no raster retificado**
    (não na foto original), porque é nesse espaço que o modelo roda em produção.

Formato em disco (idêntico nos dois casos)::

    dataset/
      train/images/0001.png     imagem retificada (BGR)
      train/masks/0001.png      máscara uint8: 0 = fundo, 255 = pé
      val/images/…  val/masks/…
      meta.json                 px/mm, alvo de calibração, procedência

A máscara é gravada com **anti-aliasing**: o valor do pixel é a fração de cobertura.
Isso não é cosmético — é o que permite ao modelo aprender a borda em sub-pixel, e é o
que o refino do contorno consome em produção (isolinha de 50 %).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore


@dataclass
class DatasetMeta:
    name: str
    px_per_mm: float
    samples: int
    source: str
    notes: str = ""


def _require_cv2() -> None:
    if cv2 is None:
        raise ImportError("opencv é necessário para gerar/ler o conjunto de dados")


def _write_pair(root: Path, split: str, index: int, image: np.ndarray,
                mask: np.ndarray) -> None:
    img_dir = root / split / "images"
    msk_dir = root / split / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir.mkdir(parents=True, exist_ok=True)
    name = f"{index:05d}.png"
    cv2.imwrite(str(img_dir / name), image)
    cv2.imwrite(str(msk_dir / name), mask)


def generate_synthetic_dataset(out_dir: str | Path, *, samples: int = 400,
                               val_fraction: float = 0.15, seed: int = 7,
                               px_per_mm: float = 6.0) -> DatasetMeta:
    """Gera pares (imagem retificada, máscara) a partir do gerador de cenas.

    Roda o **pipeline real** até a retificação, de modo que as imagens tenham
    exatamente a distribuição que o modelo verá em produção — inclusive os artefatos
    da reamostragem. A máscara vem do contorno de verdade projetado no mesmo raster.
    """
    _require_cv2()
    import sys

    backend = Path(__file__).resolve().parents[2] / "backend"
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))

    from app.calibration.fit import fit_homography
    from app.calibration.homography import build_rectification
    from app.calibration.marker import detect_marker
    from app.calibration.target import board4_target
    from app.synth import generator as G

    root = Path(out_dir)
    rng = np.random.default_rng(seed)
    written = 0
    val_cut = int(samples * (1.0 - val_fraction))

    for i in range(samples):
        left = float(rng.uniform(200.0, 300.0))
        right = left + float(rng.uniform(-6.0, 6.0))
        spec = G.board_scene(
            left, right,
            tilt_deg=float(rng.uniform(0.0, 26.0)),
            azimuth_deg=float(rng.uniform(0.0, 360.0)),
            roll_deg=float(rng.uniform(-20.0, 20.0)),
            camera_distance_mm=float(rng.uniform(500.0, 1100.0)),
            noise_sigma=float(rng.uniform(1.0, 7.0)),
            glare=float(rng.choice([0.0, 0.0, 0.15, 0.35])),
            jpeg_quality=int(rng.choice([0, 0, 70, 88])),
            background=str(rng.choice(["light", "light", "dark"])),
            skin_bgr=tuple(int(v) for v in rng.integers(
                low=[90, 110, 140], high=[175, 200, 232])),
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        try:
            scene = G.render_scene(spec)
            det = detect_marker(scene.image_bgr)
            if not det.found:
                continue
            fit = fit_homography(det, board4_target())
            rect = build_rectification(scene.image_bgr, det,
                                       fit.homography_image_to_mm,
                                       fit.control_points_mm)
        except Exception:
            continue

        mask = np.zeros(rect.image.shape[:2], dtype=np.uint8)
        shift = 4
        for truth in scene.truth_contours_mm:
            pts = np.round(rect.mm_to_rect_px(truth) * (1 << shift)).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255, lineType=cv2.LINE_AA, shift=shift)

        split = "train" if written < val_cut else "val"
        _write_pair(root, split, written, rect.image, mask)
        written += 1

    meta = DatasetMeta(
        name=root.name, px_per_mm=px_per_mm, samples=written, source="synthetic",
        notes=("Máscaras exatas por construção. NÃO substituem fotos reais: a "
               "distribuição de pele, sombra, sujeira e patologia não está "
               "representada."),
    )
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta.json").write_text(json.dumps(asdict(meta), indent=1),
                                    encoding="utf-8")
    return meta


def load_pairs(root: str | Path, split: str = "train") -> list[tuple[Path, Path]]:
    """Lista os pares (imagem, máscara) de um split, conferindo o pareamento."""
    base = Path(root) / split
    images = sorted((base / "images").glob("*.png"))
    pairs: list[tuple[Path, Path]] = []
    for img in images:
        mask = base / "masks" / img.name
        if not mask.is_file():
            raise FileNotFoundError(f"máscara ausente para {img}")
        pairs.append((img, mask))
    if not pairs:
        raise FileNotFoundError(f"nenhum par encontrado em {base}")
    return pairs


def edge_weight_map(mask: np.ndarray, band_px: int = 3,
                    edge_weight: float = 5.0) -> np.ndarray:
    """Peso por pixel que concentra a perda **na borda**.

    É a borda que vira milímetro. Uma perda uniforme otimiza a área (IoU) e é quase
    indiferente a um deslocamento de um pixel na fronteira — exatamente o erro que
    importa aqui.
    """
    _require_cv2()
    binary = (mask > 127).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    dil = cv2.dilate(binary, kernel, iterations=band_px)
    ero = cv2.erode(binary, kernel, iterations=band_px)
    band = (dil - ero).astype(bool)
    weights = np.ones(mask.shape, dtype=np.float32)
    weights[band] = edge_weight
    return weights

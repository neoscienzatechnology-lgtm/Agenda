"""Critério de aceite do modelo: erro em **milímetros**, através do pipeline real.

    python -m pedigrafia_vision.benchmark --scenes 60
    python -m pedigrafia_vision.benchmark --scenes 60 --model modelo.onnx

Por que não IoU
---------------
IoU é quase insensível ao que importa aqui. Um deslocamento de um pixel em toda a
fronteira de um pé de 265 mm muda o IoU na terceira casa decimal e muda o comprimento
em ~0,3 mm — que é o que vai para o molde. Um modelo pode subir o IoU e piorar a peça.

Este *benchmark* roda o pipeline inteiro (detecção do alvo → homografia → segmentação
→ contorno → medidas) contra cenas de geometria conhecida e reporta o erro nas
grandezas que o profissional usa: comprimento, larguras e desvio do contorno.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def _prepare_path() -> None:
    backend = Path(__file__).resolve().parents[2] / "backend"
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))


def run_benchmark(scenes: int = 40, seed: int = 3, model: str | None = None,
                  use_board: bool = True) -> dict:
    _prepare_path()
    if model:
        os.environ["PEDIGRAFIA_SEGMENTER"] = "onnx"
        os.environ["PEDIGRAFIA_SEG_MODEL"] = model
        from app import config

        config.get_settings.cache_clear()

    from app.geometry import polygon as poly
    from app.image_io import DecodedImage
    from app.pipeline import PipelineBlocked, analyze
    from app.synth import generator as G

    rng = np.random.default_rng(seed)
    length_errors: list[float] = []
    forefoot_errors: list[float] = []
    contour_p95: list[float] = []
    blocked = 0

    for _ in range(scenes):
        left = float(rng.uniform(210.0, 295.0))
        right = left + float(rng.uniform(-5.0, 5.0))
        kwargs = dict(
            tilt_deg=float(rng.uniform(0.0, 24.0)),
            azimuth_deg=float(rng.uniform(0.0, 360.0)),
            roll_deg=float(rng.uniform(-15.0, 15.0)),
            camera_distance_mm=float(rng.uniform(520.0, 1050.0)),
            noise_sigma=float(rng.uniform(1.0, 6.0)),
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        spec = (G.board_scene(left, right, **kwargs) if use_board
                else G.bilateral_scene(left, right, **kwargs))
        scene = G.render_scene(spec)
        try:
            result = analyze(DecodedImage(scene.image_bgr, 1, scene.image_bgr.shape[1],
                                          scene.image_bgr.shape[0]))
        except PipelineBlocked:
            blocked += 1
            continue

        feet = sorted(result.feet, key=lambda f: float(f.component.centroid_mm[0]))
        for foot, truth in zip(feet, scene.truth_contours_mm):
            truth_len, _, _ = poly.max_caliper(truth)
            got_len, _, _ = poly.max_caliper(foot.contour_high_res_mm)
            length_errors.append(got_len - truth_len)

            reference = poly.resample_closed(truth, 1.5)
            distances = np.array([
                float(np.min(np.linalg.norm(foot.contour_high_res_mm - p, axis=1)))
                for p in reference
            ])
            contour_p95.append(float(np.percentile(distances, 95)))

            truth_width = float(np.max(truth[:, 0]) - np.min(truth[:, 0]))
            got_width = float(np.max(foot.contour_high_res_mm[:, 0])
                              - np.min(foot.contour_high_res_mm[:, 0]))
            forefoot_errors.append(got_width - truth_width)

    def stats(values: list[float]) -> dict:
        if not values:
            return {"n": 0}
        arr = np.array(values)
        return {
            "n": int(arr.size),
            "meanMm": round(float(np.mean(arr)), 4),
            "stdMm": round(float(np.std(arr)), 4),
            "maxAbsMm": round(float(np.max(np.abs(arr))), 4),
            "p95AbsMm": round(float(np.percentile(np.abs(arr), 95)), 4),
        }

    return {
        "segmenter": model or "classical",
        "target": "board4" if use_board else "single",
        "scenes": scenes,
        "blocked": blocked,
        "length": stats(length_errors),
        "bboxWidth": stats(forefoot_errors),
        "contourP95": stats(contour_p95),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--model", default=None, help="caminho de um .onnx")
    ap.add_argument("--single-marker", action="store_true",
                    help="usa o alvo de marcador único em vez do tabuleiro")
    args = ap.parse_args()
    report = run_benchmark(args.scenes, args.seed, args.model,
                           use_board=not args.single_marker)
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

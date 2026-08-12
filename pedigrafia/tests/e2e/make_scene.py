"""Gera a cena sintética usada pelo teste end-to-end do navegador."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.synth import generator as G  # noqa: E402

# JPEG de alta qualidade: a cena PNG tem 12 MB e o upload dominava o tempo do
# teste. A compressão não altera a medida (verificado em test_pipeline_metrology).
out = ROOT / "tests" / "e2e" / "scene.jpg"
out.write_bytes(G.render_scene(G.bilateral_scene(258.0, 261.0)).encode_jpeg(94))
print(f"cena escrita: {out}")

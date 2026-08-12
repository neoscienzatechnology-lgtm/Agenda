"""Verificação metrológica independente de um PDF baixado pelo navegador.

Uso: python verify_pdf.py <arquivo.pdf> <comprimento_mm_exibido>
Saída: JSON com o comprimento realmente contido no arquivo.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.pdf.builder import CONTOUR_RGB  # noqa: E402
from app.pdf.inspect import inspect_pdf, path_max_caliper_mm  # noqa: E402

pdf_path, declared = sys.argv[1], float(sys.argv[2])
info = inspect_pdf(Path(pdf_path).read_bytes())
paths = info.paths_with_color(CONTOUR_RGB)
measured = path_max_caliper_mm(max(paths, key=lambda p: len(p.points_mm))) if paths else 0.0

print(json.dumps({
    "isA4": info.is_a4(),
    "mediaBoxMm": [round(v, 4) for v in info.media_box_mm],
    "declaredMm": declared,
    "measuredMm": round(measured, 4),
    "errorMm": round(measured - declared, 4),
    "pathCount": len(paths),
}))

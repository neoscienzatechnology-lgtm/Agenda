"""Gera o *fixture* de paridade entre a medição em Python e em TypeScript.

Executar: ``python tests/make_parity_fixture.py``

O arquivo resultante (``tests/fixtures/measurement_parity.json``) é consumido por
``tests/backend/test_parity.py`` (pytest) e por
``frontend/src/geom/__tests__/parity.test.ts`` (vitest). Se as duas implementações
divergirem, os dois testes falham.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.geometry.frame import frame_from_axis_points  # noqa: E402
from app.measurements.compute import compute_measurements  # noqa: E402
from app.synth import foot_shape as FS  # noqa: E402

CASES = [
    {"lengthMm": 200.0, "laterality": "right", "rotationDeg": 0.0, "toeCutT": 0.78},
    {"lengthMm": 240.0, "laterality": "left", "rotationDeg": 7.5, "toeCutT": 0.74},
    {"lengthMm": 265.0, "laterality": "right", "rotationDeg": -12.0, "toeCutT": 0.80},
    {"lengthMm": 285.0, "laterality": "left", "rotationDeg": 23.0, "toeCutT": 0.86},
]


def build_case(spec: dict) -> dict:
    contour = FS.foot_polygon(spec["lengthMm"], spec["laterality"], "below",
                              center_mm=(31.7, -84.3),
                              rotation_deg=spec["rotationDeg"])
    # Reduz a densidade para um contorno "editável" realista, ainda fiel à forma.
    from app.geometry import polygon as poly

    contour = poly.simplify_closed(poly.resample_closed(contour, 1.0), 0.25)

    lm = FS.expected_landmarks_mm(spec["lengthMm"], spec["laterality"], "below",
                                  center_mm=(31.7, -84.3),
                                  rotation_deg=spec["rotationDeg"])
    from app.geometry.frame import bootstrap_frame, refine_frame

    frame0, _ = bootstrap_frame(contour)
    frame0 = refine_frame(contour, frame0, lm["T2"])
    uv = frame0.to_local(contour)
    a = frame0.to_plane(np.array([[float(np.min(uv[:, 0])), 0.0]]))[0]
    b = frame0.to_plane(np.array([[float(np.max(uv[:, 0])), 0.0]]))[0]

    frame = frame_from_axis_points(contour, a, b, 1)
    meas, _ = compute_measurements(contour, frame, lm["M1"], lm["M5"],
                                   spec["toeCutT"], "silhouette")

    return {
        "name": f"{spec['laterality']}-{spec['lengthMm']:.0f}mm-rot{spec['rotationDeg']:g}",
        "input": {
            "contourMm": [{"x": float(p[0]), "y": float(p[1])} for p in contour],
            "axis": {
                "aMm": {"x": float(a[0]), "y": float(a[1])},
                "bMm": {"x": float(b[0]), "y": float(b[1])},
            },
            "m1Mm": {"x": float(lm["M1"][0]), "y": float(lm["M1"][1])},
            "m5Mm": {"x": float(lm["M5"][0]), "y": float(lm["M5"][1])},
            "toeCutT": spec["toeCutT"],
        },
        "expected": {
            "lengthMm": meas.length_mm,
            "forefootWidthMm": meas.forefoot_width_mm,
            "midfootWidthMm": meas.midfoot_width_mm,
            "heelWidthMm": meas.heel_width_mm,
            "heelToMetatarsalLineMm": meas.heel_to_metatarsal_line_mm,
            "archIndex": meas.arch_index,
            "plantarAreaMm2": meas.plantar_area_mm2,
            "bboxWidthMm": meas.bbox_width_mm,
            "forefootWidthAtT": meas.forefoot_width_at_t,
            "midfootWidthAtT": meas.midfoot_width_at_t,
            "heelWidthAtT": meas.heel_width_at_t,
            "archAreasMm2": list(meas.arch_areas_mm2),
            "heelToM1Mm": meas.heel_to_m1_mm,
            "heelToM5Mm": meas.heel_to_m5_mm,
            "metatarsalLineLengthMm": meas.metatarsal_line_length_mm,
            "frameOrigin": {"x": float(frame.origin[0]), "y": float(frame.origin[1])},
            "frameU": {"x": float(frame.u[0]), "y": float(frame.u[1])},
            "frameLengthMm": frame.length_mm,
        },
    }


def main() -> None:
    out = {
        "generatedBy": "tests/make_parity_fixture.py",
        "toleranceMm": 1e-6,
        "cases": [build_case(spec) for spec in CASES],
    }
    path = ROOT / "tests" / "fixtures" / "measurement_parity.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"escrito: {path} ({len(out['cases'])} casos)")


if __name__ == "__main__":
    main()

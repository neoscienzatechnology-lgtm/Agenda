"""Recomputação autoritativa a partir da geometria revisada.

O editor do frontend recalcula medidas localmente para feedback instantâneo, mas o
valor que vai para o PDF é **sempre** este: recomputado no servidor, a partir da
geometria em mm enviada pelo profissional.
"""

from __future__ import annotations

import numpy as np

from .. import schemas as S
from ..geometry import polygon as poly
from ..geometry.frame import frame_from_axis_points
from ..measurements.compute import compute_measurements, local_coords
from .serialize import frame_to_schema, measurements_to_schema, to_np


class ReviewValidationError(ValueError):
    pass


def validate_reviewed(foot: S.ReviewedFoot) -> np.ndarray:
    contour = to_np(foot.contourMm)
    if len(contour) < 3:
        raise ReviewValidationError("O contorno precisa de pelo menos 3 pontos.")
    if len(contour) > 4000:
        raise ReviewValidationError("Contorno com número de pontos excessivo.")
    if not np.all(np.isfinite(contour)):
        raise ReviewValidationError("O contorno contém coordenadas inválidas.")
    if poly.area(contour) <= 1.0:
        raise ReviewValidationError("O contorno revisado tem área nula.")
    # Auto-interseção invalidaria área, índice de arco e o próprio molde.
    if len(contour) <= 400 and not poly.is_simple(contour):
        raise ReviewValidationError(
            "O contorno revisado se autointersecta. Corrija os nós cruzados.")

    ids = {lm.id for lm in foot.landmarks}
    missing = [k for k in ("M1", "M5") if k not in ids]
    if missing:
        raise ReviewValidationError(
            f"Landmarks obrigatórios ausentes: {', '.join(missing)}.")
    return contour


def landmark_map(foot: S.ReviewedFoot) -> dict[str, np.ndarray]:
    return {lm.id: np.array([lm.positionMm.x, lm.positionMm.y], dtype=np.float64)
            for lm in foot.landmarks}


def recompute(foot: S.ReviewedFoot, view: str = "below") -> S.MeasureResponse:
    contour = validate_reviewed(foot)
    lms = landmark_map(foot)

    medial_sign = 1
    if "M1" in lms and "M5" in lms:
        axis_a = np.array([foot.axis.aMm.x, foot.axis.aMm.y])
        axis_b = np.array([foot.axis.bMm.x, foot.axis.bMm.y])
        tmp = frame_from_axis_points(contour, axis_a, axis_b, 1)
        v_m1 = float(tmp.to_local(lms["M1"][None, :])[0, 1])
        v_m5 = float(tmp.to_local(lms["M5"][None, :])[0, 1])
        medial_sign = 1 if v_m1 >= v_m5 else -1

    frame = frame_from_axis_points(
        contour, np.array([foot.axis.aMm.x, foot.axis.aMm.y]),
        np.array([foot.axis.bMm.x, foot.axis.bMm.y]), medial_sign)

    meas, _ = compute_measurements(contour, frame, lms.get("M1"), lms.get("M5"),
                                   foot.toeCutT, "silhouette")

    mt_line: list[S.PointMm] = []
    if "M1" in lms and "M5" in lms:
        mt_line = [S.PointMm(x=round(float(lms["M1"][0]), 4),
                             y=round(float(lms["M1"][1]), 4)),
                   S.PointMm(x=round(float(lms["M5"][0]), 4),
                             y=round(float(lms["M5"][1]), 4))]

    return S.MeasureResponse(
        measurements=measurements_to_schema(meas),
        frame=frame_to_schema(frame),
        landmarksFootFrame=[S.LocalCoord(**c)
                            for c in local_coords(lms, frame, meas.length_mm)],
        metatarsalLineMm=mt_line,
        warnings=list(meas.warnings),
    )

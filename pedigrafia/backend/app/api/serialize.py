"""Conversão entre os objetos do pipeline e os contratos Pydantic.

Toda geometria sai em **milímetros**. Não há nenhum campo em pixels no contrato de
saída além da posição dos cantos do marcador na foto original (informativo/debug).
"""

from __future__ import annotations

import numpy as np

from .. import schemas as S
from ..calibration.homography import Rectification
from ..calibration.marker import MarkerDetection
from ..geometry.frame import FootFrame
from ..pipeline import FootResult, PipelineResult, foot_landmark_dict, foot_local_coords
from ..quality.gate import QualityReport

TOE_LABELS = ["Hálux (T1)", "2º pododáctilo (T2)", "3º pododáctilo (T3)",
              "4º pododáctilo (T4)", "5º pododáctilo (T5)"]
MT_LABELS = ["1ª cabeça metatarsal (M1)", "2ª cabeça metatarsal (M2)",
             "3ª cabeça metatarsal (M3)", "4ª cabeça metatarsal (M4)",
             "5ª cabeça metatarsal (M5)"]


def pts(arr) -> list[S.PointMm]:
    a = np.asarray(arr, dtype=np.float64).reshape(-1, 2)
    return [S.PointMm(x=round(float(p[0]), 4), y=round(float(p[1]), 4)) for p in a]


def to_np(points: list[S.PointMm]) -> np.ndarray:
    return np.array([[p.x, p.y] for p in points], dtype=np.float64)


def quality_to_schema(report: QualityReport) -> S.CaptureQuality:
    return S.CaptureQuality(
        score=round(report.score(), 1),
        passed=report.passed,
        summary=report.summary(),
        blockers=list(report.blockers),
        checks=[S.QualityCheck(
            id=c.id, label=c.label, passed=c.passed, score=round(c.score, 4),
            value=(round(float(c.value), 4) if c.value is not None else None),
            threshold=(round(float(c.threshold), 4)
                       if c.threshold is not None else None),
            hint=c.hint, severity=c.severity,
        ) for c in report.checks],
    )


def marker_to_schema(marker: MarkerDetection, round_trip_sides: list[float],
                     round_trip_error: float) -> S.MarkerInfo:
    from ..config import get_settings

    return S.MarkerInfo(
        detected=marker.found,
        dictionary=marker.dictionary,
        markerId=marker.marker_id,
        sizeMm=get_settings().marker_size_mm,
        cornersPx=[S.PointPx(x=round(float(p[0]), 3), y=round(float(p[1]), 3))
                   for p in np.asarray(marker.corners_px).reshape(-1, 2)],
        confidence=round(marker.confidence, 4),
        srcPxPerMm=round(marker.src_px_per_mm, 4),
        skew=round(marker.skew, 5),
        tiltDeg=round(marker.tilt_deg, 3),
        roundTripSideMm=[round(float(s), 4) for s in round_trip_sides],
        roundTripErrorMm=(round(float(round_trip_error), 4)
                          if np.isfinite(round_trip_error) else -1.0),
    )


def rectification_to_schema(rect: Rectification) -> S.RectificationInfo:
    return S.RectificationInfo(
        pxPerMm=round(rect.px_per_mm, 6),
        originMm=S.PointMm(x=round(float(rect.origin_mm[0]), 4),
                           y=round(float(rect.origin_mm[1]), 4)),
        widthPx=int(rect.width_px), heightPx=int(rect.height_px),
        homographyImageToMm=[round(float(v), 10)
                             for v in rect.homography_image_to_mm.ravel()],
    )


def frame_to_schema(frame: FootFrame) -> S.FootFrame:
    return S.FootFrame(
        originMm=S.PointMm(x=round(float(frame.origin[0]), 4),
                           y=round(float(frame.origin[1]), 4)),
        axisUMm=S.PointMm(x=round(float(frame.u[0]), 8),
                          y=round(float(frame.u[1]), 8)),
        axisVMm=S.PointMm(x=round(float(frame.v[0]), 8),
                          y=round(float(frame.v[1]), 8)),
        medialSign=int(frame.medial_sign),
        orientationDeg=round(frame.orientation_deg, 4),
    )


def axis_to_schema(frame: FootFrame, contour_mm: np.ndarray) -> S.AxisSpec:
    uv = frame.to_local(contour_mm)
    a = frame.to_plane(np.array([[float(np.min(uv[:, 0])), 0.0]]))[0]
    b = frame.to_plane(np.array([[float(np.max(uv[:, 0])), 0.0]]))[0]
    return S.AxisSpec(
        aMm=S.PointMm(x=round(float(a[0]), 4), y=round(float(a[1]), 4)),
        bMm=S.PointMm(x=round(float(b[0]), 4), y=round(float(b[1]), 4)),
        angleDeg=round(frame.orientation_deg, 4),
    )


def measurements_to_schema(m) -> S.FootMeasurements:
    return S.FootMeasurements(
        lengthMm=round(m.length_mm, 2),
        forefootWidthMm=round(m.forefoot_width_mm, 2),
        midfootWidthMm=round(m.midfoot_width_mm, 2),
        heelWidthMm=round(m.heel_width_mm, 2),
        heelToMetatarsalLineMm=round(m.heel_to_metatarsal_line_mm, 2),
        archIndex=round(m.arch_index, 4),
        plantarAreaMm2=round(m.plantar_area_mm2, 1),
        bboxWidthMm=round(m.bbox_width_mm, 2),
        bboxLengthMm=round(m.bbox_length_mm, 2),
        axisLengthMm=round(m.axis_length_mm, 2),
        orientationDeg=round(m.orientation_deg, 3),
        forefootWidthAtT=round(m.forefoot_width_at_t, 4),
        midfootWidthAtT=round(m.midfoot_width_at_t, 4),
        heelWidthAtT=round(m.heel_width_at_t, 4),
        archIndexToeCutT=round(m.arch_index_toe_cut_t, 4),
        archIndexBasis=m.arch_index_basis,
        archAreasMm2=[round(float(a), 1) for a in m.arch_areas_mm2],
        heelToM1Mm=round(m.heel_to_m1_mm, 2),
        heelToM5Mm=round(m.heel_to_m5_mm, 2),
        metatarsalLineLengthMm=round(m.metatarsal_line_length_mm, 2),
    )


def landmarks_to_schema(foot: FootResult) -> list[S.LandmarkPoint]:
    out: list[S.LandmarkPoint] = []
    for i, t in enumerate(foot.toes[:5]):
        out.append(S.LandmarkPoint(
            id=f"T{i + 1}", label=TOE_LABELS[i],
            positionMm=S.PointMm(x=round(float(t.point_mm[0]), 3),
                                 y=round(float(t.point_mm[1]), 3)),
            confidence=round(float(t.confidence), 3),
            estimated=t.method != "contour_peak", method=t.method,
        ))
    for i, h in enumerate(foot.heads[:5]):
        out.append(S.LandmarkPoint(
            id=f"M{i + 1}", label=MT_LABELS[i],
            positionMm=S.PointMm(x=round(float(h.point_mm[0]), 3),
                                 y=round(float(h.point_mm[1]), 3)),
            confidence=round(float(h.confidence), 3),
            estimated=True, method=h.method,
        ))
    lm = foot_landmark_dict(foot)
    for key, label, method in (("H", "Centro do calcâneo", "heel_region_centroid"),
                               ("heelPosterior", "Ponto mais posterior do calcâneo",
                                "contour_extreme")):
        p = lm.get(key)
        if p is None:
            continue
        out.append(S.LandmarkPoint(
            id=key, label=label,
            positionMm=S.PointMm(x=round(float(p[0]), 3), y=round(float(p[1]), 3)),
            confidence=0.9, estimated=(key == "H"), method=method,
        ))
    return out


def arch_to_schema(arch, arch_id: str) -> S.ArchCurve:
    return S.ArchCurve(
        id=arch_id, pointsMm=pts(arch.points_mm),
        controlPointsMm=pts(arch.control_points_mm),
        confidence=round(float(arch.confidence), 3),
    )


def foot_to_schema(foot: FootResult, index: int) -> S.FootAnalysis:
    conf = foot.confidence
    return S.FootAnalysis(
        id=f"foot-{index}",
        laterality=foot.laterality,
        lateralityConfidence=round(float(foot.laterality_confidence), 3),
        lateralityMethod=foot.laterality_method,
        contourMm=pts(foot.contour_editable_mm),
        contourHighResMm=pts(foot.contour_high_res_mm),
        frame=frame_to_schema(foot.frame),
        axis=axis_to_schema(foot.frame, foot.contour_high_res_mm),
        landmarks=landmarks_to_schema(foot),
        landmarksFootFrame=[S.LocalCoord(**c) for c in foot_local_coords(foot)],
        metatarsalLineMm=pts(foot.metatarsal_line_mm) if foot.metatarsal_line_mm else [],
        medialArch=arch_to_schema(foot.medial_arch, "medial"),
        lateralArch=arch_to_schema(foot.lateral_arch, "lateral"),
        supportZones=[S.SupportZone(
            id=z.id, label=z.label, kind=z.kind, polygonMm=pts(z.polygon_mm),
            confidence=round(float(z.confidence), 3), color=z.color,
        ) for z in foot.support_zones],
        callosityHints=[S.CallosityHint(
            id=c.id, polygonMm=pts(c.polygon_mm),
            centroidMm=S.PointMm(x=round(float(c.centroid_mm[0]), 3),
                                 y=round(float(c.centroid_mm[1]), 3)),
            areaMm2=round(float(c.area_mm2), 2),
            confidence=round(float(c.confidence), 3),
        ) for c in foot.callosities],
        measurements=measurements_to_schema(foot.measurements),
        confidence=S.FootConfidence(
            segmentation=round(float(conf.get("segmentation", 0)), 3),
            laterality=round(float(conf.get("laterality", 0)), 3),
            marker=round(float(conf.get("marker", 0)), 3),
            metatarsal=round(float(conf.get("metatarsal", 0)), 3),
            supportArea=round(float(conf.get("supportArea", 0)), 3),
            toes=round(float(conf.get("toes", 0)), 3),
            archIndex=round(float(conf.get("archIndex", 0)), 3),
        ),
        warnings=list(foot.warnings),
        toeCutT=round(float(foot.toe_cut_t), 4),
    )


def result_to_response(result: PipelineResult, session_id: str, created_at: float,
                       expires_at: float, rectified_url: str,
                       shoe_size: float | None = None,
                       shoe_system: str = "BR") -> S.AnalyzeResponse:
    from ..measurements.compute import shoe_size_check

    feet = [foot_to_schema(f, i) for i, f in enumerate(result.feet)]
    check = None
    if shoe_size is not None and feet:
        longest = max(f.measurements.lengthMm for f in feet)
        check = S.ShoeSizeCheck(**shoe_size_check(longest, shoe_size, shoe_system))

    return S.AnalyzeResponse(
        sessionId=session_id, createdAt=created_at, expiresAt=expires_at,
        view=result.view,
        captureQuality=quality_to_schema(result.quality),
        marker=marker_to_schema(result.marker, result.round_trip_sides_mm,
                                result.round_trip_error_mm),
        rectification=rectification_to_schema(result.rectification),
        rectifiedImageUrl=rectified_url,
        feet=feet, footCount=len(feet), shoeSizeCheck=check,
        warnings=list(result.warnings),
        timingsMs=result.timings_ms,
    )

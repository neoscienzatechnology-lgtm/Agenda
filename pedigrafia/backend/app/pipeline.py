"""Orquestração do pipeline de visão — as 15 etapas, em ordem explícita.

Cada etapa é um módulo isolado; aqui só existe sequenciamento, medição de tempo e
montagem do contrato de resposta. Nenhuma regra geométrica vive neste arquivo.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .calibration.fit import (CalibrationFitError, HomographyFit,
                              extrapolation_distance_mm, fit_homography)
from .calibration.homography import (Rectification, build_rectification,
                                     verify_round_trip)
from .calibration.marker import MarkerDetection, detect_marker
from .calibration.target import CalibrationTarget, select_target_for_detection
from .callosity.detect import detect_callosities
from .config import get_settings
from .geometry import polygon as poly
from .geometry.arches import build_arches
from .geometry.contour import prepare_contour
from .geometry.frame import FootFrame, bootstrap_frame, refine_frame, heel_center
from .geometry.laterality import classify, reconcile_pair
from .geometry.separation import FootComponent, split_feet
from .geometry.support_zones import build_support_zones, detect_contact_region
from .image_io import DecodedImage
from .landmarks.metatarsals import estimate_metatarsal_heads, metatarsal_line
from .landmarks.toes import detect_toe_apices
from .measurements.compute import Measurements, compute_measurements, local_coords
from .quality.gate import QualityReport, evaluate_capture, evaluate_geometry
from .quality.metrics import glare_mask_and_fraction
from .segmentation.base import SegmentationContext, SegmentationResult
from .segmentation.factory import get_segmenter

PIPELINE_VERSION = "1.0.0"


class PipelineBlocked(Exception):
    """A captura reprovou no quality gate — não há resultado a produzir."""

    def __init__(self, report: QualityReport, marker: MarkerDetection):
        super().__init__("captura reprovada no quality gate")
        self.report = report
        self.marker = marker


@dataclass
class FootResult:
    component: FootComponent
    frame: FootFrame
    laterality: str
    laterality_confidence: float
    laterality_method: str
    laterality_cues: dict
    contour_editable_mm: np.ndarray
    contour_high_res_mm: np.ndarray
    toes: list
    heads: list
    metatarsal_line_mm: list
    medial_arch: object
    lateral_arch: object
    support_zones: list
    callosities: list
    measurements: Measurements
    toe_cut_t: float
    confidence: dict
    warnings: list[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    marker: MarkerDetection
    calibration: HomographyFit
    target: CalibrationTarget
    extrapolation_mm: float
    rectification: Rectification
    segmentation: SegmentationResult
    quality: QualityReport
    feet: list[FootResult]
    round_trip_sides_mm: list[float]
    round_trip_error_mm: float
    timings_ms: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    view: str = "below"


def _marker_exclusion_mask(rect: Rectification, target: CalibrationTarget,
                           pad_mm: float = 18.0) -> np.ndarray:
    """Região onde não se procura pé: cada marcador do alvo mais a zona de silêncio."""
    mask = np.zeros(rect.image.shape[:2], dtype=np.uint8)
    for placement in target.markers:
        x, y = placement.origin_mm
        s = placement.size_mm
        box = np.array([[x - pad_mm, y - pad_mm],
                        [x + s + pad_mm, y - pad_mm],
                        [x + s + pad_mm, y + s + pad_mm],
                        [x - pad_mm, y + s + pad_mm]], dtype=np.float64)
        cv2.fillPoly(mask, [rect.mm_to_rect_px(box).astype(np.int32)], 255)
    return mask


def analyze(decoded: DecodedImage, *, view: str | None = None,
            detect_callosity: bool = True) -> PipelineResult:
    settings = get_settings()
    view = view or settings.default_view
    timings: dict[str, float] = {}
    warnings: list[str] = []

    def stage(name: str, fn):
        t0 = time.perf_counter()
        out = fn()
        timings[name] = round((time.perf_counter() - t0) * 1000.0, 2)
        return out

    # Etapa 3 — detectar os marcadores de 50 × 50 mm.
    marker = stage("marker", lambda: detect_marker(decoded.bgr))
    target = select_target_for_detection(marker.detected_ids)

    # Quality gate, parte 1: sem marcador válido o pipeline não continua.
    report = stage("quality_capture", lambda: evaluate_capture(decoded, marker))
    if not report.passed:
        raise PipelineBlocked(report, marker)

    # Etapas 4–6 — homografia, retificação e a escala px → mm.
    # A homografia usa TODOS os cantos de TODOS os marcadores do alvo: com quatro
    # marcadores a região dos pés fica interpolada, e não extrapolada a partir de um
    # ponto só (causa medida do viés de 2 mm — ver docs/METROLOGY_CALIBRATION.md).
    try:
        fit = stage("calibrate", lambda: fit_homography(marker, target))
    except CalibrationFitError as exc:
        raise PipelineBlocked(_calibration_blocked(marker, str(exc)), marker) from exc
    warnings.extend(fit.warnings)

    rect: Rectification = stage("rectify", lambda: build_rectification(
        decoded.bgr, marker, fit.homography_image_to_mm, fit.control_points_mm))
    if rect.downscaled:
        warnings.append(
            "Área útil muito grande: a imagem retificada foi reamostrada em "
            f"{rect.px_per_mm:.2f} px/mm. A escala física continua exata."
        )

    sides, rt_error = stage("round_trip", lambda: verify_round_trip(rect, target))
    if not np.isfinite(rt_error):
        warnings.append("Não foi possível re-detectar o marcador na imagem retificada.")

    # Etapa 7 — segmentar a região dos pés.
    exclude = _marker_exclusion_mask(rect, target)
    ctx = SegmentationContext(
        px_per_mm=rect.px_per_mm, origin_mm=rect.origin_mm,
        valid_mask=rect.valid_mask, exclude_mask=exclude,
        min_area_mm2=settings.geometry.min_foot_area_mm2,
        max_area_mm2=settings.geometry.max_foot_area_mm2,
    )
    segmenter = get_segmenter()
    seg: SegmentationResult = stage("segment",
                                    lambda: segmenter.segment(rect.image, ctx))
    warnings.extend(seg.notes)

    # Etapas 8–9 — nenhum / um / dois pés, com separação de componentes.
    components = stage("separate", lambda: split_feet(
        seg.mask, rect, seg.score_field, seg.score_threshold))

    # Quanto a calibração está extrapolando sobre a região realmente medida.
    if components:
        measured_pts = np.vstack([c.contour_mm for c in components])
        extrapolation_mm = extrapolation_distance_mm(fit, measured_pts)
    else:
        extrapolation_mm = 0.0

    search_mask = cv2.bitwise_and(rect.valid_mask, cv2.bitwise_not(exclude))
    report = stage("quality_geometry", lambda: evaluate_geometry(
        report, rect.image, seg.mask, search_mask, components,
        rect.px_per_mm, rt_error, extrapolation_mm, fit))
    if not report.passed:
        raise PipelineBlocked(report, marker)

    glare_mask, _ = glare_mask_and_fraction(rect.image, search_mask, rect.px_per_mm)

    feet: list[FootResult] = []
    lat_results = []
    for comp in components:
        foot = _analyze_foot(comp, rect, view, glare_mask, detect_callosity,
                             lat_results)
        foot.confidence["segmentation"] = float(seg.confidence)
        foot.confidence["marker"] = float(marker.confidence)
        feet.append(foot)

    # Etapa 10 (continuação) — verificação cruzada bilateral.
    if len(feet) == 2:
        reconciled = reconcile_pair(lat_results,
                                    [f.component.centroid_mm for f in feet], view)
        for f, r in zip(feet, reconciled):
            f.laterality = r.laterality
            f.laterality_confidence = r.confidence
            f.laterality_method = r.method
            f.laterality_cues = r.cues
            f.confidence["laterality"] = r.confidence

    return PipelineResult(
        marker=marker, calibration=fit, target=target,
        extrapolation_mm=float(extrapolation_mm),
        rectification=rect, segmentation=seg, quality=report,
        feet=feet, round_trip_sides_mm=list(sides), round_trip_error_mm=float(rt_error),
        timings_ms=timings, warnings=warnings, view=view,
    )


def _calibration_blocked(marker: MarkerDetection, reason: str) -> QualityReport:
    """Relatório de bloqueio quando a calibração não pode sequer ser montada."""
    from .quality.gate import Check

    report = QualityReport()
    report.add(Check(
        id="calibration_fit", label="Calibração do alvo", passed=False, score=0.0,
        severity="blocker", group="marker",
        hint=f"Não foi possível calibrar a partir dos marcadores: {reason}",
    ))
    return report


def _analyze_foot(comp: FootComponent, rect: Rectification, view: str,
                  glare_mask: np.ndarray, detect_callosity: bool,
                  lat_results: list) -> FootResult:
    settings = get_settings()
    foot_warnings: list[str] = []

    # Etapas 11–12 — contorno completo, suavizado e simplificado (com verificação).
    pair = prepare_contour(comp.contour_mm)
    contour = pair.high_res_mm
    if pair.smoothing_drift_mm > settings.geometry.max_smoothing_length_drift_mm:
        foot_warnings.append("Suavização reduzida para preservar a dimensão física.")

    # Eixo em duas passagens: momentos de área → refino anatômico pelo 2º pododáctilo.
    frame, orient_conf = bootstrap_frame(contour)
    toes, hallux_sign, hallux_conf = detect_toe_apices(contour, frame)
    if len(toes) >= 2:
        frame = refine_frame(contour, frame, toes[1].point_mm)
        toes, hallux_sign, hallux_conf = detect_toe_apices(contour, frame)
    if len(toes) < 5:
        foot_warnings.append(
            f"Apenas {len(toes)} pododáctilo(s) separável(is) na silhueta; "
            "os demais foram estimados e precisam de revisão.")

    # Etapa 10 — lateralidade por geometria anatômica.
    lat = classify(contour, frame, hallux_sign, hallux_conf, view)
    frame.medial_sign = lat.medial_sign
    lat_results.append(lat)

    # Etapa 13 — landmarks estimados.
    heads, mt_conf = estimate_metatarsal_heads(contour, frame, lat.medial_sign)
    mt_line = metatarsal_line(heads)
    if heads:
        foot_warnings.append(
            "M2, M3 e M4 são ESTIMADOS por modelo — não são visíveis na fotografia "
            "plantar. Confirme ou corrija antes de exportar.")

    medial_arch, lateral_arch = build_arches(contour, frame, lat.medial_sign,
                                             confidence=0.55)

    contact_mask, contact_conf = detect_contact_region(
        rect.image, comp.mask, rect.px_per_mm)
    if contact_conf < 0.35:
        foot_warnings.append(
            "A fotografia não permite distinguir a área de contato com validade; "
            "as zonas exibidas são recortes anatômicos, não medição de apoio.")

    # Etapa 14 — medidas.
    m1 = heads[0].point_mm if len(heads) == 5 else None
    m5 = heads[4].point_mm if len(heads) == 5 else None
    meas, arch = compute_measurements(contour, frame, m1, m5, None, "silhouette")
    foot_warnings.extend(meas.warnings)

    zones = build_support_zones(contour, frame, lat.medial_sign, arch.toe_cut_t,
                                contact_conf)

    callosities = []
    if detect_callosity:
        callosities = detect_callosities(rect.image, comp.mask, rect.px_per_mm,
                                         rect.origin_mm, glare_mask)

    confidence = {
        "segmentation": 0.0,   # preenchido pelo chamador com a confiança global
        "laterality": lat.confidence,
        "marker": 0.0,
        "metatarsal": mt_conf,
        "supportArea": float(max(0.25, contact_conf)),
        "toes": float(np.mean([t.confidence for t in toes])) if toes else 0.0,
        "archIndex": arch.confidence,
        "orientation": orient_conf,
    }

    return FootResult(
        component=comp, frame=frame, laterality=lat.laterality,
        laterality_confidence=lat.confidence, laterality_method=lat.method,
        laterality_cues=lat.cues,
        contour_editable_mm=pair.editable_mm, contour_high_res_mm=contour,
        toes=toes, heads=heads, metatarsal_line_mm=mt_line,
        medial_arch=medial_arch, lateral_arch=lateral_arch,
        support_zones=zones, callosities=callosities, measurements=meas,
        toe_cut_t=arch.toe_cut_t, confidence=confidence, warnings=foot_warnings,
    )


def foot_landmark_dict(foot: FootResult) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for i, t in enumerate(foot.toes[:5]):
        out[f"T{i + 1}"] = t.point_mm
    for i, h in enumerate(foot.heads[:5]):
        out[f"M{i + 1}"] = h.point_mm
    out["H"] = heel_center(foot.contour_high_res_mm, foot.frame)
    out["heelPosterior"] = foot.contour_high_res_mm[
        int(np.argmin(poly.project(foot.contour_high_res_mm, foot.frame.origin,
                                   foot.frame.u, foot.frame.v)[:, 0]))]
    return out


def foot_local_coords(foot: FootResult) -> list[dict]:
    return local_coords(foot_landmark_dict(foot), foot.frame,
                        foot.measurements.length_mm)

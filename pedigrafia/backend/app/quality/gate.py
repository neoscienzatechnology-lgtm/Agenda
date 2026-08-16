"""Quality gate: score 0–100 e verificações bloqueantes.

Nenhuma análise final é liberada com foto inadequada. O gate roda em duas etapas:

* :func:`evaluate_capture` — antes da segmentação (marcador, resolução, foco, tremor,
  exposição, reflexos, perspectiva). Um bloqueio aqui aborta o pipeline.
* :func:`evaluate_geometry` — depois da segmentação (pés presentes, inteiros, com
  calcâneo e pododáctilos visíveis, contraste real, sem corte).

Todos os limiares vêm de ``config.QualityThresholds`` e são configuráveis por env.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..calibration.marker import MarkerDetection
from ..config import get_settings
from ..image_io import DecodedImage
from . import metrics


@dataclass
class Check:
    id: str
    label: str
    passed: bool
    score: float          # 0..1
    value: float | None = None
    threshold: float | None = None
    hint: str = ""
    severity: str = "warning"   # blocker | warning | info
    group: str = "misc"


@dataclass
class QualityReport:
    checks: list[Check] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)

    def add(self, check: Check) -> None:
        self.checks.append(check)
        if not check.passed and check.severity == "blocker":
            self.blockers.append(check.hint or check.label)

    @property
    def passed(self) -> bool:
        return not self.blockers

    def score(self) -> float:
        weights = get_settings().quality.weights
        total_w = 0.0
        acc = 0.0
        by_group: dict[str, list[float]] = {}
        for c in self.checks:
            by_group.setdefault(c.group, []).append(c.score)
        for group, scores in by_group.items():
            w = weights.get(group, 2.0)
            acc += w * float(np.mean(scores))
            total_w += w
        if total_w <= 0:
            return 0.0
        raw = 100.0 * acc / total_w
        if self.blockers:
            # Um bloqueio impede que o score sugira que "está quase bom".
            raw = min(raw, 49.0)
        return float(np.clip(raw, 0.0, 100.0))

    def summary(self) -> str:
        if self.blockers:
            return "Captura inadequada — refaça a foto."
        s = self.score()
        if s >= 85:
            return "Captura excelente."
        if s >= 70:
            return "Captura adequada."
        return "Captura no limite — considere refazer para maior precisão."


def _ramp(value: float, bad: float, good: float) -> float:
    """Normaliza para 0..1 entre ``bad`` e ``good`` (aceita good < bad)."""
    if good == bad:
        return 1.0 if value >= good else 0.0
    t = (value - bad) / (good - bad)
    return float(np.clip(t, 0.0, 1.0))


def evaluate_capture(decoded: DecodedImage, marker: MarkerDetection) -> QualityReport:
    q = get_settings().quality
    report = QualityReport()
    bgr = decoded.bgr
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # ------------------------------------------------- referência física de escala
    is_reference = marker.source == "reference"
    noun = "Objeto de referência" if is_reference else "Marcador de referência"
    fallback_hint = (
        marker.reason or
        "Marcador de 50 × 50 mm não encontrado. Mantenha-o inteiro e visível.")
    report.add(Check(
        id="marker_detected", label=f"{noun} detectado",
        passed=marker.found, score=1.0 if marker.found else 0.0,
        severity="blocker" if q.require_marker else "warning", group="marker",
        hint="" if marker.found else fallback_hint,
    ))
    if not marker.found:
        report.add(Check(
            id="marker_corners", label="Quatro cantos visíveis",
            passed=False, score=0.0, severity="blocker", group="marker",
            hint=("Sem uma referência de dimensão conhecida não existe escala "
                  "física: uma fotografia sozinha não contém tamanho. Use o alvo "
                  "impresso ou deixe um objeto padronizado (cartão, folha A4) "
                  "inteiro e no mesmo plano dos pés.")))
        return report

    report.add(Check(
        id="marker_corners", label=f"Quatro cantos {'do objeto' if is_reference else 'do marcador'} visíveis",
        passed=True, score=1.0, severity="info", group="marker",
    ))
    report.add(Check(
        id="marker_not_clipped", label=f"{noun} inteiro no enquadramento",
        passed=not marker.touches_border,
        score=0.0 if marker.touches_border else 1.0,
        severity="blocker" if q.require_marker_inside_frame else "warning",
        group="marker",
        hint=f"{noun} encosta na borda da foto; afaste-se ou recentralize.",
    ))
    report.add(Check(
        id="marker_skew", label=f"{noun} sem deformação excessiva",
        passed=marker.skew <= q.max_marker_skew,
        score=_ramp(marker.skew, q.max_marker_skew, 0.02),
        value=marker.skew, threshold=q.max_marker_skew,
        severity="blocker", group="marker",
        hint=f"{noun} muito distorcido: fotografe mais de frente.",
    ))

    # ---------------------------------------------------------------- resolução
    report.add(Check(
        id="resolution_px_per_mm",
        label=f"Amostragem sobre {'o objeto' if is_reference else 'o marcador'}",
        passed=marker.src_px_per_mm >= q.min_src_px_per_mm,
        score=_ramp(marker.src_px_per_mm, q.min_src_px_per_mm, q.good_src_px_per_mm),
        value=marker.src_px_per_mm, threshold=q.min_src_px_per_mm,
        severity="blocker", group="resolution",
        hint=f"Resolução insuficiente sobre {'o objeto' if is_reference else 'o marcador'}"
             " de referência — aproxime a câmera.",
    ))
    report.add(Check(
        id="megapixels", label="Resolução da imagem",
        passed=decoded.megapixels >= q.min_megapixels,
        score=_ramp(decoded.megapixels, q.min_megapixels, q.good_megapixels),
        value=decoded.megapixels, threshold=q.min_megapixels,
        severity="warning", group="resolution",
        hint="Imagem de baixa resolução.",
    ))

    # ------------------------------------------------------------------- foco
    roi = metrics.marker_roi_mask(bgr.shape, marker.corners_px, expand=2.4)
    focus_global = metrics.focus_score(gray)
    focus_marker = metrics.focus_score(gray, roi)
    # A nitidez sobre o marcador pesa mais: é dela que depende a precisão subpixel
    # dos cantos, e portanto toda a escala física.
    focus = (0.45 * focus_global + 0.55 * focus_marker) if np.any(roi > 0) \
        else focus_global
    report.add(Check(
        id="focus", label="Nitidez",
        passed=focus >= q.min_focus_score,
        score=_ramp(focus, q.min_focus_score, q.good_focus_score),
        value=focus, threshold=q.min_focus_score,
        severity="blocker", group="focus",
        hint="Foto desfocada. Toque para focar e repita.",
    ))

    aniso = metrics.motion_anisotropy(gray)
    report.add(Check(
        id="motion_blur", label="Ausência de tremor",
        passed=aniso <= q.max_motion_anisotropy,
        score=_ramp(aniso, q.max_motion_anisotropy, q.good_motion_anisotropy),
        value=aniso, threshold=q.max_motion_anisotropy,
        severity="warning", group="motion",
        hint="Possível tremor de câmera. Apoie o celular ou use suporte.",
    ))

    # -------------------------------------------------------------- exposição
    exp = metrics.exposure_stats(bgr)
    # O que reprova é a PERDA de informação (recorte de altas/baixas), não o brilho
    # médio: uma plataforma branca bem iluminada é legitimamente clara.
    exposure_ok = (exp.clipped_high_frac <= q.max_clipped_high_frac
                   and exp.clipped_low_frac <= q.max_clipped_low_frac)
    luma_dev = abs(exp.mean_luma - q.target_mean_luma)
    clip_term = min(_ramp(exp.clipped_high_frac, q.max_clipped_high_frac, 0.0),
                    _ramp(exp.clipped_low_frac, q.max_clipped_low_frac, 0.0))
    luma_term = _ramp(luma_dev, q.max_mean_luma_deviation, 0.0)
    report.add(Check(
        id="exposure", label="Exposição",
        passed=exposure_ok, score=0.72 * clip_term + 0.28 * luma_term,
        value=exp.mean_luma, severity="warning", group="exposure",
        hint="Há áreas estouradas ou totalmente escuras na foto.",
    ))

    # ---------------------------------------------------------------- reflexos
    # A zona de silêncio branca do marcador é branca por projeto: excluí-la evita
    # contabilizá-la como reflexo especular.
    glare_roi = cv2.bitwise_not(
        metrics.marker_roi_mask(bgr.shape, marker.corners_px, expand=2.2))
    _, glare_frac = metrics.glare_mask_and_fraction(bgr, glare_roi,
                                                    marker.src_px_per_mm)
    report.add(Check(
        id="glare", label="Reflexos controlados",
        passed=glare_frac <= q.max_glare_frac,
        score=_ramp(glare_frac, q.max_glare_frac, q.good_glare_frac),
        value=glare_frac, threshold=q.max_glare_frac,
        severity="warning", group="glare",
        hint="Reflexo intenso no vidro. Mude o ângulo ou a iluminação.",
    ))

    # ------------------------------------------------------------- perspectiva
    report.add(Check(
        id="perspective", label="Perspectiva corrigível",
        passed=marker.tilt_deg <= q.max_tilt_deg,
        score=_ramp(marker.tilt_deg, q.max_tilt_deg, q.good_tilt_deg),
        value=marker.tilt_deg, threshold=q.max_tilt_deg,
        severity="blocker", group="perspective",
        hint="Ângulo muito oblíquo. Fotografe mais perpendicular à plataforma.",
    ))
    return report


def evaluate_geometry(report: QualityReport, rect_bgr: np.ndarray,
                      foot_mask: np.ndarray, search_mask: np.ndarray,
                      components: list, px_per_mm: float,
                      round_trip_error_mm: float,
                      extrapolation_mm: float = 0.0,
                      calibration=None) -> QualityReport:
    settings = get_settings()
    q = settings.quality

    n = len(components)
    report.add(Check(
        id="feet_detected", label="Pé(s) detectado(s)",
        passed=n >= 1, score=1.0 if n >= 1 else 0.0, value=float(n),
        severity="blocker" if q.require_at_least_one_foot else "warning",
        group="framing",
        hint="Nenhum pé identificado na plataforma.",
    ))

    if n >= 1:
        cropped = [c for c in components if c.touches_border]
        report.add(Check(
            id="feet_complete", label="Pés completos (calcâneo e pododáctilos)",
            passed=not cropped, score=0.0 if cropped else 1.0,
            severity="blocker" if q.require_feet_inside_frame else "warning",
            group="framing",
            hint="Um pé está cortado no enquadramento. Reposicione e refaça.",
        ))

        margin_mm = _border_margin_mm(components, rect_bgr.shape, px_per_mm)
        report.add(Check(
            id="border_margin", label="Margem até a borda",
            passed=margin_mm >= q.min_border_margin_mm,
            score=_ramp(margin_mm, 0.0, 12.0), value=margin_mm,
            threshold=q.min_border_margin_mm, severity="warning", group="framing",
            hint="Pé muito próximo da borda da área útil.",
        ))

    contrast = metrics.foreground_background_contrast(rect_bgr, foot_mask, search_mask)
    report.add(Check(
        id="foot_bg_contrast", label="Contraste pé/fundo",
        passed=contrast >= q.min_foot_bg_contrast,
        score=_ramp(contrast, q.min_foot_bg_contrast, q.good_foot_bg_contrast),
        value=contrast, threshold=q.min_foot_bg_contrast,
        severity="blocker", group="contrast",
        hint="Pé e fundo com cores muito parecidas. Ajuste a iluminação.",
    ))

    report.add(Check(
        id="scale_round_trip", label="Escala verificada (referência re-medida)",
        passed=round_trip_error_mm <= 0.5,
        score=_ramp(round_trip_error_mm, 0.5, 0.05),
        value=round_trip_error_mm, threshold=0.5,
        severity="blocker", group="marker",
        hint="A verificação de escala falhou; a calibração não é confiável.",
    ))

    # --------------------------------------------------- cobertura da calibração
    # Dentro do casco dos pontos de controle a homografia interpola; fora dele
    # extrapola, e o erro cresce com a distância. Este é o preditor direto do viés
    # medido de até 2 mm com marcador único (ver docs/METROLOGY_CALIBRATION.md).
    warn = settings.max_extrapolation_warn_mm
    block = settings.max_extrapolation_block_mm
    from_reference = getattr(calibration, "target_name", "") == "reference"
    remedy = ("Espalhe mais objetos iguais ao redor da área de apoio: com quatro, "
              "a calibração volta a interpolar em vez de extrapolar."
              if from_reference else
              "Use um alvo com quatro marcadores ao redor da área de apoio "
              "(GET /api/marker.pdf?target=board4).")
    report.add(Check(
        id="calibration_coverage", label="Pés dentro da área calibrada",
        passed=extrapolation_mm <= block,
        score=_ramp(extrapolation_mm, block, 0.0),
        value=extrapolation_mm, threshold=warn,
        severity="blocker", group="marker",
        hint=(f"Os pés estão a {extrapolation_mm:.0f} mm fora da região coberta pela "
              f"referência: a escala está sendo EXTRAPOLADA e a exatidão cai com a "
              f"distância. {remedy}"),
    ))
    if calibration is not None and not calibration.exact:
        report.add(Check(
            id="calibration_residual", label="Resíduo do ajuste da calibração",
            passed=calibration.residual_max_mm <= 1.0,
            score=_ramp(calibration.residual_max_mm, 1.0, 0.05),
            value=calibration.residual_max_mm, threshold=1.0,
            severity="blocker", group="marker",
            hint=("Os objetos detectados não são consistentes com a dimensão "
                  "declarada. Confirme qual referência foi usada e que ela está "
                  "plana, inteira e no mesmo plano dos pés."
                  if from_reference else
                  "Os marcadores não são consistentes com o alvo declarado. Confira "
                  "as posições físicas no arquivo do alvo, com paquímetro."),
        ))
    return report


def _border_margin_mm(components: list, shape, px_per_mm: float) -> float:
    h, w = shape[:2]
    best = float("inf")
    for c in components:
        pts = c.contour_px
        d = min(float(np.min(pts[:, 0])), float(np.min(pts[:, 1])),
                float(w - 1 - np.max(pts[:, 0])), float(h - 1 - np.max(pts[:, 1])))
        best = min(best, d)
    if not np.isfinite(best):
        return 0.0
    return float(best / px_per_mm)

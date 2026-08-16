"""Contratos da API (Pydantic v2).

Espelhados 1:1 em ``frontend/src/types/pedigrafia.ts``. Um teste de paridade garante
que os nomes de campo não divirjam.

REGRA: todo campo dimensional carrega a unidade no nome (``Mm``, ``Mm2``, ``Px``,
``Deg``). Não existe campo dimensional sem unidade explícita.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Laterality = Literal["left", "right"]
ViewPoint = Literal["below", "above"]
ArchBasis = Literal["silhouette", "contact"]


class _Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class PointMm(_Base):
    x: float
    y: float


class PointPx(_Base):
    x: float
    y: float


# --------------------------------------------------------------------------- marcador


class MarkerInfo(_Base):
    detected: bool
    dictionary: str = ""
    markerId: int = -1
    sizeMm: float = 50.0
    cornersPx: list[PointPx] = Field(default_factory=list)
    """Cantos na imagem normalizada, ordem TL, TR, BR, BL (frame do marcador)."""
    confidence: float = 0.0
    srcPxPerMm: float = 0.0
    """Amostragem original no plano, medida na aresta do marcador."""
    reprojectionErrorPx: float = 0.0
    skew: float = 0.0
    """Desvio relativo entre os lados do quadrilátero observado (0 = quadrado perfeito)."""
    tiltDeg: float = 0.0
    """Inclinação estimada do plano em relação ao eixo óptico."""
    roundTripSideMm: list[float] = Field(default_factory=list)
    """Lados do marcador re-medidos na imagem retificada. Devem valer ~50,00 mm."""
    roundTripErrorMm: float = 0.0


class ReferenceObjectInfo(_Base):
    """Objeto de dimensão normalizada usado no lugar do marcador impresso."""

    key: str
    label: str
    widthMm: float
    heightMm: float
    toleranceMm: float
    standard: str
    note: str = ""
    scaleToleranceRel: float = 0.0
    """Incerteza relativa de escala imposta pela tolerância do próprio objeto."""


class CalibrationInfo(_Base):
    """Como a escala física foi estabelecida — e o quanto ela está extrapolando."""

    targetName: str
    markerCount: int
    usedMarkerIds: list[int] = Field(default_factory=list)
    residualRmsMm: float = 0.0
    residualMaxMm: float = 0.0
    exact: bool = True
    """``True`` com 4 pontos (um marcador): o ajuste é exato e o resíduo não informa nada."""
    coverageSpanMm: list[float] = Field(default_factory=list)
    extrapolationMm: float = 0.0
    """Distância dos pés até o casco dos pontos de controle. 0 = interpolado."""
    interpolated: bool = False
    source: str = "aruco"
    """``aruco`` (marcador impresso) ou ``reference`` (objeto de dimensão normalizada)."""
    reference: Optional[ReferenceObjectInfo] = None
    scaleToleranceRel: float = 0.0
    """Incerteza de escala herdada do padrão físico, que nenhum algoritmo remove."""
    scaleToleranceMm: float = 0.0
    """A mesma incerteza aplicada ao maior pé medido, em mm."""
    controlPointsMm: list[PointMm] = Field(default_factory=list)
    """Cantos usados como pontos de controle, no plano em mm — permitem à interface
    mostrar exatamente de onde veio a escala."""
    warnings: list[str] = Field(default_factory=list)


class RectificationInfo(_Base):
    pxPerMm: float
    originMm: PointMm
    widthPx: int
    heightPx: int
    homographyImageToMm: list[float]
    """Matriz 3×3 achatada em row-major."""


# --------------------------------------------------------------------- quality gate


class QualityCheck(_Base):
    id: str
    label: str
    passed: bool
    score: float
    value: Optional[float] = None
    threshold: Optional[float] = None
    hint: str = ""
    severity: Literal["blocker", "warning", "info"] = "warning"


class CaptureQuality(_Base):
    score: float
    passed: bool
    checks: list[QualityCheck]
    summary: str
    blockers: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------------- landmarks


class LandmarkPoint(_Base):
    id: str
    label: str
    positionMm: PointMm
    confidence: float
    estimated: bool = True
    method: str = ""
    """Como o ponto foi obtido. 'contour_extreme', 'peak', 'interpolated_model'…"""


class FootFrame(_Base):
    originMm: PointMm
    axisUMm: PointMm
    """Vetor unitário longitudinal calcâneo → dedos."""
    axisVMm: PointMm
    medialSign: int
    """+1 se a borda medial está no sentido +v; -1 caso contrário."""
    orientationDeg: float


class LocalCoord(_Base):
    id: str
    uMm: float
    vMm: float
    tPct: float


# ---------------------------------------------------------------------------- medidas


class FootMeasurements(_Base):
    lengthMm: float
    forefootWidthMm: float
    midfootWidthMm: float
    heelWidthMm: float
    heelToMetatarsalLineMm: float
    archIndex: float

    # armazenamento interno / auxiliar
    plantarAreaMm2: float
    bboxWidthMm: float
    bboxLengthMm: float
    axisLengthMm: float
    orientationDeg: float
    forefootWidthAtT: float = 0.0
    midfootWidthAtT: float = 0.0
    heelWidthAtT: float = 0.0
    archIndexToeCutT: float = 0.0
    archIndexBasis: ArchBasis = "silhouette"
    archAreasMm2: list[float] = Field(default_factory=list)
    heelToM1Mm: float = 0.0
    heelToM5Mm: float = 0.0
    metatarsalLineLengthMm: float = 0.0
    axisParallelLengthMm: float = 0.0


class SupportZone(_Base):
    id: str
    label: str
    kind: Literal["heel", "midfoot", "forefoot", "first_ray", "custom"]
    polygonMm: list[PointMm]
    confidence: float
    color: str = "#F97316"


class CallosityHint(_Base):
    id: str
    polygonMm: list[PointMm]
    centroidMm: PointMm
    areaMm2: float
    confidence: float
    note: str = "Sugestão visual — não constitui diagnóstico."
    accepted: bool = False


class FootConfidence(_Base):
    segmentation: float
    laterality: float
    marker: float
    metatarsal: float
    supportArea: float
    toes: float
    archIndex: float


class ArchCurve(_Base):
    id: Literal["medial", "lateral"]
    pointsMm: list[PointMm]
    controlPointsMm: list[PointMm]
    confidence: float


class AxisSpec(_Base):
    aMm: PointMm
    """Extremidade posterior (calcâneo)."""
    bMm: PointMm
    """Extremidade anterior (dedos)."""
    angleDeg: float


class FootAnalysis(_Base):
    id: str
    laterality: Laterality
    lateralityConfidence: float
    lateralityMethod: str = ""

    contourMm: list[PointMm]
    """Contorno editável (nós simplificados). Fonte de verdade da revisão."""
    contourHighResMm: list[PointMm] = Field(default_factory=list)
    """Contorno detalhado preservado internamente (não editável)."""

    frame: FootFrame
    axis: AxisSpec
    landmarks: list[LandmarkPoint]
    landmarksFootFrame: list[LocalCoord]
    metatarsalLineMm: list[PointMm]
    medialArch: ArchCurve
    lateralArch: ArchCurve
    supportZones: list[SupportZone]
    callosityHints: list[CallosityHint] = Field(default_factory=list)
    measurements: FootMeasurements
    confidence: FootConfidence
    warnings: list[str] = Field(default_factory=list)
    toeCutT: float = 0.78


class ShoeSizeCheck(_Base):
    """Puramente informativo. NUNCA influencia dimensão alguma."""

    provided: Optional[float] = None
    system: str = "BR"
    expectedRangeMm: Optional[list[float]] = None
    measuredMm: Optional[float] = None
    compatible: Optional[bool] = None
    message: str = ""


class AnalyzeResponse(_Base):
    sessionId: str
    createdAt: float
    expiresAt: float
    view: ViewPoint
    captureQuality: CaptureQuality
    marker: MarkerInfo
    calibration: Optional[CalibrationInfo] = None
    rectification: Optional[RectificationInfo] = None
    rectifiedImageUrl: Optional[str] = None
    feet: list[FootAnalysis] = Field(default_factory=list)
    footCount: int = 0
    shoeSizeCheck: Optional[ShoeSizeCheck] = None
    warnings: list[str] = Field(default_factory=list)
    pipelineVersion: str = "1.0.0"
    timingsMs: dict[str, float] = Field(default_factory=dict)


# ------------------------------------------------------- entrada revisada (pós-editor)


class ReviewedFoot(_Base):
    """Geometria em mm devolvida pelo editor. Continua em milímetros — sempre."""

    id: str
    laterality: Laterality
    contourMm: list[PointMm]
    axis: AxisSpec
    landmarks: list[LandmarkPoint]
    medialArch: ArchCurve
    lateralArch: ArchCurve
    supportZones: list[SupportZone] = Field(default_factory=list)
    callosityHints: list[CallosityHint] = Field(default_factory=list)
    toeCutT: float = 0.78
    notes: str = ""


class MeasureRequest(_Base):
    foot: ReviewedFoot
    view: ViewPoint = "below"


class MeasureResponse(_Base):
    measurements: FootMeasurements
    frame: FootFrame
    landmarksFootFrame: list[LocalCoord]
    metatarsalLineMm: list[PointMm]
    warnings: list[str] = Field(default_factory=list)


class ApproveRequest(_Base):
    sessionId: str
    feet: list[ReviewedFoot]
    approvedBy: str = ""


class ApproveResponse(_Base):
    reviewToken: str
    approvedAt: float
    feet: list[MeasureResponse]


class ExportPdfRequest(_Base):
    sessionId: str
    reviewToken: str
    foot: ReviewedFoot
    view: ViewPoint = "below"
    includeSupportZones: bool = True
    includeArches: bool = True
    includeMetatarsals: bool = True
    includeAxis: bool = True
    includeMeasurements: bool = True
    patientLabel: str = ""


class RenderAnnotatedRequest(_Base):
    sessionId: str
    feet: list[ReviewedFoot]
    view: ViewPoint = "below"
    scalePxPerMm: float = 8.0
    includeCallosities: bool = True


class PdfVerification(_Base):
    """Resultado da releitura programática de um PDF gerado (QA)."""

    mediaBoxPt: list[float]
    mediaBoxMm: list[float]
    isA4: bool
    contourLengthMm: float
    contourWidthMm: float
    contourBboxMm: list[float]
    declaredLengthMm: float
    lengthErrorMm: float
    pathPointCount: int
    ok: bool
    message: str = ""

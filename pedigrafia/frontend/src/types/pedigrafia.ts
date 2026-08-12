/** Contratos da API — espelho de `backend/app/schemas.py`. Tudo em milímetros. */

import type { PointMm } from '../geom/polygon'

export type { PointMm }
export type Laterality = 'left' | 'right'
export type ViewPoint = 'below' | 'above'

export interface PointPx {
  x: number
  y: number
}

export interface MarkerInfo {
  detected: boolean
  dictionary: string
  markerId: number
  sizeMm: number
  cornersPx: PointPx[]
  confidence: number
  srcPxPerMm: number
  reprojectionErrorPx: number
  skew: number
  tiltDeg: number
  roundTripSideMm: number[]
  roundTripErrorMm: number
}

export interface RectificationInfo {
  pxPerMm: number
  originMm: PointMm
  widthPx: number
  heightPx: number
  homographyImageToMm: number[]
}

export interface QualityCheck {
  id: string
  label: string
  passed: boolean
  score: number
  value: number | null
  threshold: number | null
  hint: string
  severity: 'blocker' | 'warning' | 'info'
}

export interface CaptureQuality {
  score: number
  passed: boolean
  checks: QualityCheck[]
  summary: string
  blockers: string[]
}

export interface LandmarkPoint {
  id: string
  label: string
  positionMm: PointMm
  confidence: number
  estimated: boolean
  method: string
}

export interface FootFrameDto {
  originMm: PointMm
  axisUMm: PointMm
  axisVMm: PointMm
  medialSign: number
  orientationDeg: number
}

export interface LocalCoord {
  id: string
  uMm: number
  vMm: number
  tPct: number
}

export interface FootMeasurements {
  lengthMm: number
  forefootWidthMm: number
  midfootWidthMm: number
  heelWidthMm: number
  heelToMetatarsalLineMm: number
  archIndex: number
  plantarAreaMm2: number
  bboxWidthMm: number
  bboxLengthMm: number
  axisLengthMm: number
  orientationDeg: number
  forefootWidthAtT: number
  midfootWidthAtT: number
  heelWidthAtT: number
  archIndexToeCutT: number
  archIndexBasis: 'silhouette' | 'contact'
  archAreasMm2: number[]
  heelToM1Mm: number
  heelToM5Mm: number
  metatarsalLineLengthMm: number
  axisParallelLengthMm: number
}

export interface SupportZone {
  id: string
  label: string
  kind: 'heel' | 'midfoot' | 'forefoot' | 'first_ray' | 'custom'
  polygonMm: PointMm[]
  confidence: number
  color: string
}

export interface CallosityHint {
  id: string
  polygonMm: PointMm[]
  centroidMm: PointMm
  areaMm2: number
  confidence: number
  note: string
  accepted: boolean
}

export interface FootConfidence {
  segmentation: number
  laterality: number
  marker: number
  metatarsal: number
  supportArea: number
  toes: number
  archIndex: number
}

export interface ArchCurve {
  id: 'medial' | 'lateral'
  pointsMm: PointMm[]
  controlPointsMm: PointMm[]
  confidence: number
}

export interface AxisSpec {
  aMm: PointMm
  bMm: PointMm
  angleDeg: number
}

export interface FootAnalysis {
  id: string
  laterality: Laterality
  lateralityConfidence: number
  lateralityMethod: string
  contourMm: PointMm[]
  contourHighResMm: PointMm[]
  frame: FootFrameDto
  axis: AxisSpec
  landmarks: LandmarkPoint[]
  landmarksFootFrame: LocalCoord[]
  metatarsalLineMm: PointMm[]
  medialArch: ArchCurve
  lateralArch: ArchCurve
  supportZones: SupportZone[]
  callosityHints: CallosityHint[]
  measurements: FootMeasurements
  confidence: FootConfidence
  warnings: string[]
  toeCutT: number
}

export interface ShoeSizeCheck {
  provided: number | null
  system: string
  expectedRangeMm: number[] | null
  measuredMm: number | null
  compatible: boolean | null
  message: string
}

export interface AnalyzeResponse {
  sessionId: string
  createdAt: number
  expiresAt: number
  view: ViewPoint
  captureQuality: CaptureQuality
  marker: MarkerInfo
  rectification: RectificationInfo | null
  rectifiedImageUrl: string | null
  feet: FootAnalysis[]
  footCount: number
  shoeSizeCheck: ShoeSizeCheck | null
  warnings: string[]
  pipelineVersion: string
  timingsMs: Record<string, number>
}

export interface CaptureRejected {
  code: 'capture_rejected'
  captureQuality: CaptureQuality
  marker: MarkerInfo
  message: string
}

/** Geometria revisada devolvida ao servidor — continua em milímetros. */
export interface ReviewedFoot {
  id: string
  laterality: Laterality
  contourMm: PointMm[]
  axis: AxisSpec
  landmarks: LandmarkPoint[]
  medialArch: ArchCurve
  lateralArch: ArchCurve
  supportZones: SupportZone[]
  callosityHints: CallosityHint[]
  toeCutT: number
  notes: string
}

export interface MeasureResponse {
  measurements: FootMeasurements
  frame: FootFrameDto
  landmarksFootFrame: LocalCoord[]
  metatarsalLineMm: PointMm[]
  warnings: string[]
}

export interface ApproveResponse {
  reviewToken: string
  approvedAt: number
  feet: MeasureResponse[]
}

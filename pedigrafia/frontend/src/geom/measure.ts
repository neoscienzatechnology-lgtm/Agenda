/**
 * Cálculo de medidas no cliente — espelho exato de
 * `backend/app/measurements/compute.py` e `arch_index.py`.
 *
 * Por que existe: durante o arrasto de um nó o profissional precisa ver o
 * comprimento mudar em tempo real; um ida-e-volta HTTP por quadro seria inviável.
 *
 * Por que NÃO substitui o servidor: o valor que entra no PDF é sempre recomputado
 * em `/api/export-pdf`. Este módulo é feedback; o servidor é a verdade. O teste de
 * paridade garante que os dois concordam em 1e-6 mm.
 */

import {
  area,
  centroid,
  clipBandU,
  maxCaliper,
  pointLineDistance,
  project,
  resampleClosed,
  widthAtU,
  type PointMm,
  type Vec2,
} from './polygon'

export const FOREFOOT_BAND: [number, number] = [0.55, 0.88]
export const MIDFOOT_BAND: [number, number] = [0.38, 0.62]
export const HEEL_BAND: [number, number] = [0.02, 0.24]
const WIDTH_SAMPLES = 121
const DENSE_STEP_MM = 0.5

export interface FootFrame {
  origin: PointMm
  u: Vec2
  v: Vec2
  lengthMm: number
  medialSign: number
  orientationDeg: number
}

export interface Measurements {
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

function rot90(u: Vec2): Vec2 {
  return [-u[1], u[0]]
}

/** Espelha `_frame_from_direction` do backend, incluindo o recentramento em v. */
export function frameFromDirection(contour: PointMm[], dir: Vec2): FootFrame {
  const norm = Math.hypot(dir[0], dir[1])
  const u: Vec2 = norm > 1e-12 ? [dir[0] / norm, dir[1] / norm] : [0, 1]
  const v = rot90(u)

  const projU = contour.map((p) => p.x * u[0] + p.y * u[1])
  const uMin = Math.min(...projU)
  const uMax = Math.max(...projU)
  const lengthMm = uMax - uMin

  let heelIdx = 0
  for (let i = 1; i < projU.length; i++) if (projU[i] < projU[heelIdx]) heelIdx = i
  const heel = contour[heelIdx]
  const heelProj = heel.x * u[0] + heel.y * u[1]
  let origin: PointMm = {
    x: heel.x - (heelProj - uMin) * u[0],
    y: heel.y - (heelProj - uMin) * u[1],
  }

  const uvTmp = project(contour, origin, u, v)
  const band = clipBandU(uvTmp, 0, Math.max(1e-3, 0.15 * lengthMm))
  let vCenter: number
  if (band.length >= 3) {
    vCenter = centroid(band).y
  } else {
    const sorted = uvTmp.map((p) => p.y).sort((a, b) => a - b)
    const mid = sorted.length >> 1
    vCenter = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
  }
  origin = { x: origin.x + vCenter * v[0], y: origin.y + vCenter * v[1] }

  return {
    origin,
    u,
    v,
    lengthMm,
    medialSign: 1,
    orientationDeg: (Math.atan2(u[1], u[0]) * 180) / Math.PI,
  }
}

export function frameFromAxisPoints(
  contour: PointMm[],
  a: PointMm,
  b: PointMm,
  medialSign = 1,
): FootFrame {
  const dir: Vec2 = [b.x - a.x, b.y - a.y]
  const frame = frameFromDirection(contour, Math.hypot(dir[0], dir[1]) < 1e-9 ? [0, 1] : dir)
  frame.medialSign = medialSign
  return frame
}

function extremeWidth(
  uv: PointMm[],
  frame: FootFrame,
  band: [number, number],
  mode: 'max' | 'min',
): { width: number; t: number } {
  let bestWidth = 0
  let bestT = 0
  let found = false
  for (let i = 0; i < WIDTH_SAMPLES; i++) {
    const t = band[0] + ((band[1] - band[0]) * i) / (WIDTH_SAMPLES - 1)
    const w = widthAtU(uv, t * frame.lengthMm)
    if (!Number.isFinite(w) || w <= 0) continue
    if (!found || (mode === 'max' ? w > bestWidth : w < bestWidth)) {
      bestWidth = w
      bestT = t
      found = true
    }
  }
  return found ? { width: bestWidth, t: bestT } : { width: 0, t: 0 }
}

export function computeArchIndex(
  dense: PointMm[],
  frame: FootFrame,
  toeCutT: number,
): { value: number; areas: number[]; toeCutT: number } {
  const uv = project(dense, frame.origin, frame.u, frame.v)
  const cut = Math.min(0.97, Math.max(0.55, toeCutT))
  const uCut = cut * frame.lengthMm
  const body = clipBandU(uv, 0, uCut)
  if (body.length < 3) return { value: 0, areas: [], toeCutT: cut }

  const third = uCut / 3
  const areas: number[] = []
  for (let k = 0; k < 3; k++) {
    const band = clipBandU(uv, k * third, (k + 1) * third)
    areas.push(band.length >= 3 ? area(band) : 0)
  }
  const total = areas.reduce((s, v) => s + v, 0)
  return { value: total > 1e-9 ? areas[1] / total : 0, areas, toeCutT: cut }
}

export function computeMeasurements(
  contour: PointMm[],
  frame: FootFrame,
  m1: PointMm | null,
  m5: PointMm | null,
  toeCutT: number,
): Measurements {
  const dense = resampleClosed(contour, DENSE_STEP_MM)
  const uv = project(dense, frame.origin, frame.u, frame.v)

  const us = uv.map((p) => p.x)
  const vs = uv.map((p) => p.y)
  const axisParallelLengthMm = Math.max(...us) - Math.min(...us)
  const bboxWidthMm = Math.max(...vs) - Math.min(...vs)
  // Comprimento oficial: diâmetro do fecho convexo — estacionário e independente
  // do eixo (ver a docstring de measurements/compute.py).
  const lengthMm = maxCaliper(dense)

  const fore = extremeWidth(uv, frame, FOREFOOT_BAND, 'max')
  const mid = extremeWidth(uv, frame, MIDFOOT_BAND, 'min')
  const heel = extremeWidth(uv, frame, HEEL_BAND, 'max')

  let heelIdx = 0
  for (let i = 1; i < uv.length; i++) if (uv[i].x < uv[heelIdx].x) heelIdx = i
  const heelPoint = dense[heelIdx]

  let heelToLine = 0
  let heelToM1 = 0
  let heelToM5 = 0
  let mtLen = 0
  if (m1 && m5) {
    mtLen = Math.hypot(m5.x - m1.x, m5.y - m1.y)
    if (mtLen > 1e-6) heelToLine = pointLineDistance(heelPoint, m1, m5)
    heelToM1 = Math.hypot(m1.x - heelPoint.x, m1.y - heelPoint.y)
    heelToM5 = Math.hypot(m5.x - heelPoint.x, m5.y - heelPoint.y)
  }

  const arch = computeArchIndex(dense, frame, toeCutT)

  return {
    lengthMm,
    forefootWidthMm: fore.width,
    midfootWidthMm: mid.width,
    heelWidthMm: heel.width,
    heelToMetatarsalLineMm: heelToLine,
    archIndex: arch.value,
    plantarAreaMm2: area(dense),
    bboxWidthMm,
    bboxLengthMm: axisParallelLengthMm,
    axisLengthMm: axisParallelLengthMm,
    orientationDeg: frame.orientationDeg,
    forefootWidthAtT: fore.t,
    midfootWidthAtT: mid.t,
    heelWidthAtT: heel.t,
    archIndexToeCutT: arch.toeCutT,
    archIndexBasis: 'silhouette',
    archAreasMm2: arch.areas,
    heelToM1Mm: heelToM1,
    heelToM5Mm: heelToM5,
    metatarsalLineLengthMm: mtLen,
    axisParallelLengthMm,
  }
}

/** Formata com uma casa decimal e vírgula decimal (pt-BR). */
export function mm(value: number, decimals = 1): string {
  return `${value.toFixed(decimals).replace('.', ',')} mm`
}

export function num(value: number, decimals = 3): string {
  return value.toFixed(decimals).replace('.', ',')
}

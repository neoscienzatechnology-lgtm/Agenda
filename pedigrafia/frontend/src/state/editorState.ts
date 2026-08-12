/**
 * Estado do editor de revisão, com histórico de desfazer/refazer.
 *
 * O modelo é sempre geometria em MILÍMETROS. O zoom/pan vive em `Viewport` e nunca
 * toca no modelo — é impossível, por construção, que uma interação da UI altere uma
 * dimensão física.
 */

import type {
  ArchCurve,
  AxisSpec,
  CallosityHint,
  FootAnalysis,
  LandmarkPoint,
  Laterality,
  PointMm,
  ReviewedFoot,
  SupportZone,
} from '../types/pedigrafia'

export type ToolId = 'contour' | 'axis' | 'metatarsals' | 'arches' | 'zones' | 'crop'

export interface EditableFoot {
  id: string
  laterality: Laterality
  lateralityConfidence: number
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

export interface EditorState {
  feet: EditableFoot[]
  activeFootIndex: number
  tool: ToolId
  selection: Selection | null
}

export type Selection =
  | { kind: 'contourNode'; index: number }
  | { kind: 'landmark'; id: string }
  | { kind: 'axisHandle'; end: 'a' | 'b' }
  | { kind: 'archControl'; arch: 'medial' | 'lateral'; index: number }
  | { kind: 'zoneVertex'; zoneId: string; index: number }
  | { kind: 'zone'; zoneId: string }
  | { kind: 'toeCut' }

export function fromAnalysis(foot: FootAnalysis): EditableFoot {
  return {
    id: foot.id,
    laterality: foot.laterality,
    lateralityConfidence: foot.lateralityConfidence,
    contourMm: foot.contourMm.map((p) => ({ ...p })),
    axis: {
      aMm: { ...foot.axis.aMm },
      bMm: { ...foot.axis.bMm },
      angleDeg: foot.axis.angleDeg,
    },
    landmarks: foot.landmarks.map((l) => ({ ...l, positionMm: { ...l.positionMm } })),
    medialArch: cloneArch(foot.medialArch),
    lateralArch: cloneArch(foot.lateralArch),
    supportZones: foot.supportZones.map((z) => ({
      ...z,
      polygonMm: z.polygonMm.map((p) => ({ ...p })),
    })),
    callosityHints: foot.callosityHints.map((c) => ({
      ...c,
      polygonMm: c.polygonMm.map((p) => ({ ...p })),
      centroidMm: { ...c.centroidMm },
    })),
    toeCutT: foot.toeCutT,
    notes: '',
  }
}

function cloneArch(arch: ArchCurve): ArchCurve {
  return {
    id: arch.id,
    pointsMm: arch.pointsMm.map((p) => ({ ...p })),
    controlPointsMm: arch.controlPointsMm.map((p) => ({ ...p })),
    confidence: arch.confidence,
  }
}

export function toReviewed(foot: EditableFoot): ReviewedFoot {
  return {
    id: foot.id,
    laterality: foot.laterality,
    contourMm: foot.contourMm,
    axis: foot.axis,
    landmarks: foot.landmarks,
    medialArch: foot.medialArch,
    lateralArch: foot.lateralArch,
    supportZones: foot.supportZones,
    callosityHints: foot.callosityHints,
    toeCutT: foot.toeCutT,
    notes: foot.notes,
  }
}

export function landmarkOf(foot: EditableFoot, id: string): PointMm | null {
  const lm = foot.landmarks.find((l) => l.id === id)
  return lm ? lm.positionMm : null
}

// ------------------------------------------------------------------- histórico

export interface History {
  past: EditorState[]
  present: EditorState
  future: EditorState[]
  baseline: EditorState
}

const MAX_HISTORY = 80

export function initHistory(state: EditorState): History {
  return { past: [], present: state, future: [], baseline: clone(state) }
}

export function clone(state: EditorState): EditorState {
  return {
    feet: state.feet.map((f) => structuredClone(f)),
    activeFootIndex: state.activeFootIndex,
    tool: state.tool,
    selection: state.selection ? { ...state.selection } : null,
  }
}

/** Commit: guarda o estado anterior no histórico (uma entrada por gesto). */
export function commit(history: History, next: EditorState): History {
  const past = [...history.past, history.present]
  if (past.length > MAX_HISTORY) past.shift()
  return { past, present: next, future: [], baseline: history.baseline }
}

/** Alteração transitória (arrasto em andamento): não cria entrada no histórico. */
export function live(history: History, next: EditorState): History {
  return { ...history, present: next }
}

export function undo(history: History): History {
  if (history.past.length === 0) return history
  const previous = history.past[history.past.length - 1]
  return {
    past: history.past.slice(0, -1),
    present: previous,
    future: [history.present, ...history.future],
    baseline: history.baseline,
  }
}

export function redo(history: History): History {
  if (history.future.length === 0) return history
  const next = history.future[0]
  return {
    past: [...history.past, history.present],
    present: next,
    future: history.future.slice(1),
    baseline: history.baseline,
  }
}

export function restoreAutomatic(history: History): History {
  return commit(history, clone(history.baseline))
}

export const canUndo = (h: History) => h.past.length > 0
export const canRedo = (h: History) => h.future.length > 0

// -------------------------------------------------------------------- edições

export function updateFoot(
  state: EditorState,
  index: number,
  updater: (foot: EditableFoot) => EditableFoot,
): EditorState {
  const feet = state.feet.map((f, i) => (i === index ? updater(f) : f))
  return { ...state, feet }
}

export function moveContourNode(foot: EditableFoot, index: number, p: PointMm): EditableFoot {
  const contourMm = foot.contourMm.map((q, i) => (i === index ? { ...p } : q))
  return { ...foot, contourMm }
}

export function insertContourNode(foot: EditableFoot, edgeIndex: number, p: PointMm): EditableFoot {
  const contourMm = [...foot.contourMm]
  contourMm.splice(edgeIndex + 1, 0, { ...p })
  return { ...foot, contourMm }
}

export function removeContourNode(foot: EditableFoot, index: number): EditableFoot {
  if (foot.contourMm.length <= 12) return foot
  const contourMm = foot.contourMm.filter((_, i) => i !== index)
  return { ...foot, contourMm }
}

export function moveLandmark(foot: EditableFoot, id: string, p: PointMm): EditableFoot {
  const landmarks = foot.landmarks.map((l) =>
    l.id === id
      ? { ...l, positionMm: { ...p }, method: `${stripManual(l.method)}+manual`, confidence: 1 }
      : l,
  )
  return { ...foot, landmarks }
}

function stripManual(method: string): string {
  return method.endsWith('+manual') ? method.slice(0, -'+manual'.length) : method
}

export function moveAxisHandle(foot: EditableFoot, end: 'a' | 'b', p: PointMm): EditableFoot {
  const axis: AxisSpec = { ...foot.axis, [end === 'a' ? 'aMm' : 'bMm']: { ...p } }
  const dx = axis.bMm.x - axis.aMm.x
  const dy = axis.bMm.y - axis.aMm.y
  axis.angleDeg = (Math.atan2(dy, dx) * 180) / Math.PI
  return { ...foot, axis }
}

export function translateAxis(foot: EditableFoot, dx: number, dy: number): EditableFoot {
  return {
    ...foot,
    axis: {
      aMm: { x: foot.axis.aMm.x + dx, y: foot.axis.aMm.y + dy },
      bMm: { x: foot.axis.bMm.x + dx, y: foot.axis.bMm.y + dy },
      angleDeg: foot.axis.angleDeg,
    },
  }
}

export function moveArchControl(
  foot: EditableFoot,
  arch: 'medial' | 'lateral',
  index: number,
  p: PointMm,
): EditableFoot {
  const key = arch === 'medial' ? 'medialArch' : 'lateralArch'
  const curve = foot[key]
  const controlPointsMm = curve.controlPointsMm.map((q, i) => (i === index ? { ...p } : q))
  return {
    ...foot,
    [key]: { ...curve, controlPointsMm, pointsMm: catmullRom(controlPointsMm, 12) },
  }
}

export function moveZone(foot: EditableFoot, zoneId: string, dx: number, dy: number): EditableFoot {
  const supportZones = foot.supportZones.map((z) =>
    z.id === zoneId
      ? { ...z, polygonMm: z.polygonMm.map((p) => ({ x: p.x + dx, y: p.y + dy })) }
      : z,
  )
  return { ...foot, supportZones }
}

export function scaleZone(foot: EditableFoot, zoneId: string, factor: number): EditableFoot {
  const supportZones = foot.supportZones.map((z) => {
    if (z.id !== zoneId) return z
    const cx = z.polygonMm.reduce((s, p) => s + p.x, 0) / z.polygonMm.length
    const cy = z.polygonMm.reduce((s, p) => s + p.y, 0) / z.polygonMm.length
    return {
      ...z,
      polygonMm: z.polygonMm.map((p) => ({
        x: cx + (p.x - cx) * factor,
        y: cy + (p.y - cy) * factor,
      })),
    }
  })
  return { ...foot, supportZones }
}

export function moveZoneVertex(
  foot: EditableFoot,
  zoneId: string,
  index: number,
  p: PointMm,
): EditableFoot {
  const supportZones = foot.supportZones.map((z) =>
    z.id === zoneId
      ? { ...z, polygonMm: z.polygonMm.map((q, i) => (i === index ? { ...p } : q)) }
      : z,
  )
  return { ...foot, supportZones }
}

export function addZone(foot: EditableFoot, center: PointMm, radiusMm = 18): EditableFoot {
  const polygonMm: PointMm[] = []
  for (let i = 0; i < 10; i++) {
    const a = (2 * Math.PI * i) / 10
    polygonMm.push({ x: center.x + radiusMm * Math.cos(a), y: center.y + radiusMm * Math.sin(a) })
  }
  const zone: SupportZone = {
    id: `zone-custom-${Date.now().toString(36)}`,
    label: 'Zona personalizada',
    kind: 'custom',
    polygonMm,
    confidence: 1,
    color: '#F97316',
  }
  return { ...foot, supportZones: [...foot.supportZones, zone] }
}

export function removeZone(foot: EditableFoot, zoneId: string): EditableFoot {
  return { ...foot, supportZones: foot.supportZones.filter((z) => z.id !== zoneId) }
}

export function toggleCallosity(foot: EditableFoot, id: string): EditableFoot {
  return {
    ...foot,
    callosityHints: foot.callosityHints.map((c) =>
      c.id === id ? { ...c, accepted: !c.accepted } : c,
    ),
  }
}

export function removeCallosity(foot: EditableFoot, id: string): EditableFoot {
  return { ...foot, callosityHints: foot.callosityHints.filter((c) => c.id !== id) }
}

export function setLaterality(foot: EditableFoot, laterality: Laterality): EditableFoot {
  return { ...foot, laterality, lateralityConfidence: 1 }
}

export function setToeCut(foot: EditableFoot, t: number): EditableFoot {
  return { ...foot, toeCutT: Math.min(0.97, Math.max(0.55, t)) }
}

/** Spline Catmull-Rom aberta — usada para redesenhar os arcos a partir dos controles. */
export function catmullRom(control: PointMm[], samplesPerSegment: number): PointMm[] {
  if (control.length < 2) return control.map((p) => ({ ...p }))
  const out: PointMm[] = []
  const pts = [control[0], ...control, control[control.length - 1]]
  for (let i = 0; i < pts.length - 3; i++) {
    const [p0, p1, p2, p3] = [pts[i], pts[i + 1], pts[i + 2], pts[i + 3]]
    for (let s = 0; s < samplesPerSegment; s++) {
      const t = s / samplesPerSegment
      const t2 = t * t
      const t3 = t2 * t
      out.push({
        x:
          0.5 *
          (2 * p1.x +
            (-p0.x + p2.x) * t +
            (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2 +
            (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t3),
        y:
          0.5 *
          (2 * p1.y +
            (-p0.y + p2.y) * t +
            (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2 +
            (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3),
      })
    }
  }
  out.push({ ...control[control.length - 1] })
  return out
}

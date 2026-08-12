/** Desenho do canvas do editor. Só leitura do modelo — nunca o modifica. */

import { crossingsAtU, project, resampleClosed, type PointMm } from '../geom/polygon'
import { frameFromAxisPoints } from '../geom/measure'
import type { EditableFoot, Selection, ToolId } from '../state/editorState'
import { toScreen, type Viewport } from './viewport'

export const COLORS = {
  contour: '#0F2D5C',
  contourNode: '#1D4ED8',
  contourNodeActive: '#F97316',
  axis: '#64748B',
  metatarsal: '#1D4ED8',
  toe: '#334155',
  medialArch: '#16A34A',
  lateralArch: '#0EA5E9',
  support: '#F97316',
  callosity: '#DC2626',
  lowConfidence: '#DC2626',
  toeCut: '#8B5CF6',
  inactive: '#94A3B8',
}

export interface RenderOptions {
  viewport: Viewport
  tool: ToolId
  selection: Selection | null
  active: boolean
  showBackground: boolean
  dpr: number
}

export function drawFoot(
  ctx: CanvasRenderingContext2D,
  foot: EditableFoot,
  opts: RenderOptions,
): void {
  const { viewport: vp, active } = opts
  const alpha = active ? 1 : 0.4
  ctx.save()
  ctx.globalAlpha = alpha

  drawSupportZones(ctx, foot, opts)
  drawCallosities(ctx, foot, opts)
  drawArches(ctx, foot, opts)
  drawToeCut(ctx, foot, opts)
  drawAxis(ctx, foot, opts)
  drawMetatarsalLine(ctx, foot, opts)
  drawContour(ctx, foot, opts)
  drawLandmarks(ctx, foot, opts)
  drawSideLabel(ctx, foot, vp)

  ctx.restore()
}

function polyPath(ctx: CanvasRenderingContext2D, pts: PointMm[], vp: Viewport, close: boolean) {
  if (pts.length === 0) return
  ctx.beginPath()
  const first = toScreen(vp, pts[0])
  ctx.moveTo(first.x, first.y)
  for (let i = 1; i < pts.length; i++) {
    const s = toScreen(vp, pts[i])
    ctx.lineTo(s.x, s.y)
  }
  if (close) ctx.closePath()
}

function drawContour(ctx: CanvasRenderingContext2D, foot: EditableFoot, o: RenderOptions) {
  const { viewport: vp } = o
  polyPath(ctx, foot.contourMm, vp, true)
  ctx.strokeStyle = COLORS.contour
  ctx.lineWidth = o.active ? 2.2 : 1.4
  ctx.lineJoin = 'round'
  ctx.stroke()

  if (!o.active || o.tool !== 'contour') return
  const r = 4.5
  for (let i = 0; i < foot.contourMm.length; i++) {
    const s = toScreen(vp, foot.contourMm[i])
    const selected = o.selection?.kind === 'contourNode' && o.selection.index === i
    ctx.beginPath()
    ctx.arc(s.x, s.y, selected ? r + 2 : r, 0, Math.PI * 2)
    ctx.fillStyle = selected ? COLORS.contourNodeActive : '#FFFFFF'
    ctx.fill()
    ctx.strokeStyle = selected ? COLORS.contourNodeActive : COLORS.contourNode
    ctx.lineWidth = 1.6
    ctx.stroke()
  }
}

function drawAxis(ctx: CanvasRenderingContext2D, foot: EditableFoot, o: RenderOptions) {
  const { viewport: vp } = o
  const a = toScreen(vp, foot.axis.aMm)
  const b = toScreen(vp, foot.axis.bMm)
  ctx.save()
  ctx.setLineDash([9, 6])
  ctx.strokeStyle = o.tool === 'axis' && o.active ? COLORS.contourNodeActive : COLORS.axis
  ctx.lineWidth = o.tool === 'axis' && o.active ? 2 : 1.3
  ctx.beginPath()
  ctx.moveTo(a.x, a.y)
  ctx.lineTo(b.x, b.y)
  ctx.stroke()
  ctx.restore()

  if (o.tool !== 'axis' || !o.active) return
  for (const [end, s] of [
    ['a', a],
    ['b', b],
  ] as const) {
    const selected = o.selection?.kind === 'axisHandle' && o.selection.end === end
    ctx.beginPath()
    ctx.arc(s.x, s.y, selected ? 9 : 7, 0, Math.PI * 2)
    ctx.fillStyle = selected ? COLORS.contourNodeActive : '#FFFFFF'
    ctx.fill()
    ctx.strokeStyle = COLORS.axis
    ctx.lineWidth = 2
    ctx.stroke()
  }
}

function drawMetatarsalLine(ctx: CanvasRenderingContext2D, foot: EditableFoot, o: RenderOptions) {
  const m1 = foot.landmarks.find((l) => l.id === 'M1')
  const m5 = foot.landmarks.find((l) => l.id === 'M5')
  if (!m1 || !m5) return
  const a = toScreen(o.viewport, m1.positionMm)
  const b = toScreen(o.viewport, m5.positionMm)
  ctx.beginPath()
  ctx.moveTo(a.x, a.y)
  ctx.lineTo(b.x, b.y)
  ctx.strokeStyle = COLORS.metatarsal
  ctx.lineWidth = 1.6
  ctx.stroke()
}

function drawLandmarks(ctx: CanvasRenderingContext2D, foot: EditableFoot, o: RenderOptions) {
  const showMt = o.tool === 'metatarsals' && o.active
  for (const lm of foot.landmarks) {
    const isMt = lm.id.startsWith('M')
    const isToe = lm.id.startsWith('T')
    if (!isMt && !isToe && lm.id !== 'H') continue
    if (!o.active && !isMt) continue

    const s = toScreen(o.viewport, lm.positionMm)
    const selected = o.selection?.kind === 'landmark' && o.selection.id === lm.id
    const radius = isMt ? (showMt ? 8 : 5) : 4.5
    const uncertain = lm.confidence < 0.55

    if (uncertain && o.active) {
      ctx.beginPath()
      ctx.arc(s.x, s.y, radius + 5, 0, Math.PI * 2)
      ctx.strokeStyle = COLORS.lowConfidence
      ctx.setLineDash([3, 3])
      ctx.lineWidth = 1.4
      ctx.stroke()
      ctx.setLineDash([])
    }

    ctx.beginPath()
    ctx.arc(s.x, s.y, selected ? radius + 2 : radius, 0, Math.PI * 2)
    ctx.fillStyle = selected ? COLORS.contourNodeActive : isMt ? COLORS.metatarsal : '#FFFFFF'
    ctx.fill()
    ctx.strokeStyle = isMt ? '#FFFFFF' : COLORS.toe
    ctx.lineWidth = 1.6
    ctx.stroke()

    if (o.active && (isMt || isToe)) {
      ctx.font = '600 11px ui-sans-serif, system-ui, sans-serif'
      ctx.fillStyle = isMt ? COLORS.metatarsal : COLORS.toe
      ctx.fillText(lm.id, s.x + radius + 3, s.y - radius - 2)
    }
  }
}

function drawArches(ctx: CanvasRenderingContext2D, foot: EditableFoot, o: RenderOptions) {
  const editing = o.tool === 'arches' && o.active
  const archLayers: { arch: typeof foot.medialArch; color: string; dash: number[] }[] = [
    { arch: foot.medialArch, color: COLORS.medialArch, dash: [8, 5] },
    { arch: foot.lateralArch, color: COLORS.lateralArch, dash: [3, 4] },
  ]
  for (const { arch, color, dash } of archLayers) {
    if (arch.pointsMm.length < 2) continue
    ctx.save()
    ctx.setLineDash(dash)
    ctx.strokeStyle = color
    ctx.lineWidth = editing ? 2.2 : 1.5
    polyPath(ctx, arch.pointsMm, o.viewport, false)
    ctx.stroke()
    ctx.restore()

    if (!editing) continue
    arch.controlPointsMm.forEach((p, i) => {
      const s = toScreen(o.viewport, p)
      const selected =
        o.selection?.kind === 'archControl' &&
        o.selection.arch === arch.id &&
        o.selection.index === i
      ctx.beginPath()
      ctx.rect(s.x - 5, s.y - 5, 10, 10)
      ctx.fillStyle = selected ? COLORS.contourNodeActive : '#FFFFFF'
      ctx.fill()
      ctx.strokeStyle = color
      ctx.lineWidth = 1.8
      ctx.stroke()
    })
  }
}

function drawSupportZones(ctx: CanvasRenderingContext2D, foot: EditableFoot, o: RenderOptions) {
  const editing = o.tool === 'zones' && o.active
  for (const zone of foot.supportZones) {
    if (zone.polygonMm.length < 3) continue
    polyPath(ctx, zone.polygonMm, o.viewport, true)
    ctx.fillStyle = hexToRgba(zone.color, editing ? 0.22 : 0.14)
    ctx.fill()
    ctx.strokeStyle = hexToRgba(zone.color, editing ? 0.9 : 0.4)
    ctx.lineWidth = editing ? 1.8 : 1
    ctx.stroke()

    if (!editing) continue
    const selected = o.selection?.kind === 'zone' && o.selection.zoneId === zone.id
    zone.polygonMm.forEach((p, i) => {
      const s = toScreen(o.viewport, p)
      const vSel =
        o.selection?.kind === 'zoneVertex' &&
        o.selection.zoneId === zone.id &&
        o.selection.index === i
      ctx.beginPath()
      ctx.arc(s.x, s.y, vSel ? 6 : 4, 0, Math.PI * 2)
      ctx.fillStyle = vSel || selected ? COLORS.contourNodeActive : '#FFFFFF'
      ctx.fill()
      ctx.strokeStyle = zone.color
      ctx.lineWidth = 1.4
      ctx.stroke()
    })
  }
}

function drawCallosities(ctx: CanvasRenderingContext2D, foot: EditableFoot, o: RenderOptions) {
  for (const hint of foot.callosityHints) {
    if (hint.polygonMm.length < 3) continue
    polyPath(ctx, hint.polygonMm, o.viewport, true)
    ctx.strokeStyle = COLORS.callosity
    ctx.setLineDash(hint.accepted ? [] : [4, 3])
    ctx.lineWidth = hint.accepted ? 2 : 1.2
    ctx.stroke()
    ctx.setLineDash([])
  }
}

function drawToeCut(ctx: CanvasRenderingContext2D, foot: EditableFoot, o: RenderOptions) {
  if (foot.contourMm.length < 3) return
  const frame = frameFromAxisPoints(foot.contourMm, foot.axis.aMm, foot.axis.bMm)
  const dense = resampleClosed(foot.contourMm, 1)
  const uv = project(dense, frame.origin, frame.u, frame.v)
  const u = foot.toeCutT * frame.lengthMm
  const vs = crossingsAtU(uv, u)
  if (vs.length < 2) return
  const a = { x: u, y: Math.min(...vs) }
  const b = { x: u, y: Math.max(...vs) }
  const toPlane = (p: PointMm): PointMm => ({
    x: frame.origin.x + p.x * frame.u[0] + p.y * frame.v[0],
    y: frame.origin.y + p.x * frame.u[1] + p.y * frame.v[1],
  })
  const sa = toScreen(o.viewport, toPlane(a))
  const sb = toScreen(o.viewport, toPlane(b))
  ctx.save()
  ctx.setLineDash([5, 4])
  ctx.strokeStyle = COLORS.toeCut
  ctx.lineWidth = o.tool === 'crop' && o.active ? 2.4 : 1.2
  ctx.beginPath()
  ctx.moveTo(sa.x, sa.y)
  ctx.lineTo(sb.x, sb.y)
  ctx.stroke()
  ctx.restore()
}

function drawSideLabel(ctx: CanvasRenderingContext2D, foot: EditableFoot, vp: Viewport) {
  if (foot.contourMm.length < 3) return
  let minY = Infinity
  let minX = Infinity
  for (const p of foot.contourMm) {
    if (p.y < minY) minY = p.y
    if (p.x < minX) minX = p.x
  }
  const s = toScreen(vp, { x: minX, y: minY - 6 })
  ctx.font = '700 15px ui-sans-serif, system-ui, sans-serif'
  ctx.fillStyle = COLORS.contour
  ctx.fillText(foot.laterality === 'right' ? 'DIREITO' : 'ESQUERDO', s.x, s.y)
}

export function hexToRgba(hex: string, alpha: number): string {
  const h = hex.replace('#', '')
  const r = parseInt(h.slice(0, 2), 16)
  const g = parseInt(h.slice(2, 4), 16)
  const b = parseInt(h.slice(4, 6), 16)
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

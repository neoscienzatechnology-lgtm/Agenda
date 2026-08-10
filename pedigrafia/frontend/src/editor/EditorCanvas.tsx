/**
 * Canvas do editor: zoom/pan, pinch no mobile, seleção e arrasto de todos os
 * elementos vetoriais. O modelo permanece em milímetros o tempo todo.
 */

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'

import { boundingBox, nearestEdge, type PointMm } from '../geom/polygon'
import {
  addZone,
  insertContourNode,
  moveArchControl,
  moveAxisHandle,
  moveContourNode,
  moveLandmark,
  moveZone,
  moveZoneVertex,
  removeContourNode,
  translateAxis,
  type EditableFoot,
  type EditorState,
  type Selection,
} from '../state/editorState'
import { drawFoot } from './render'
import { createViewport, fitTo, pan, toMm, toScreen, zoomAt, type Viewport } from './viewport'

const HIT_RADIUS_PX = 16

export interface EditorCanvasProps {
  state: EditorState
  backgroundUrl: string | null
  backgroundOriginMm: PointMm | null
  backgroundPxPerMm: number | null
  showBackground: boolean
  onLiveChange: (next: EditorState) => void
  onCommit: (next: EditorState) => void
  onSelect: (selection: Selection | null) => void
  onActivateFoot: (index: number) => void
}

type DragState =
  | { kind: 'none' }
  | { kind: 'pan'; lastX: number; lastY: number }
  | { kind: 'element'; selection: Selection; grabOffset: PointMm; moved: boolean }
  | { kind: 'axisBody'; lastMm: PointMm; moved: boolean }

export function EditorCanvas(props: EditorCanvasProps) {
  const {
    state,
    backgroundUrl,
    backgroundOriginMm,
    backgroundPxPerMm,
    showBackground,
    onLiveChange,
    onCommit,
    onSelect,
    onActivateFoot,
  } = props

  const wrapRef = useRef<HTMLDivElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [viewport, setViewport] = useState<Viewport>(createViewport)
  const [size, setSize] = useState({ w: 640, h: 480 })
  const bgRef = useRef<HTMLImageElement | null>(null)
  const [bgReady, setBgReady] = useState(false)
  const drag = useRef<DragState>({ kind: 'none' })
  const pointers = useRef<Map<number, { x: number; y: number }>>(new Map())
  const pinch = useRef<{ distance: number; centerX: number; centerY: number } | null>(null)
  const fitted = useRef(false)

  // ------------------------------------------------------------- dimensionamento
  useLayoutEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect()
      setSize({ w: Math.max(240, r.width), h: Math.max(240, r.height) })
    })
    ro.observe(el)
    const r = el.getBoundingClientRect()
    setSize({ w: Math.max(240, r.width), h: Math.max(240, r.height) })
    return () => ro.disconnect()
  }, [])

  useEffect(() => {
    if (!backgroundUrl) {
      bgRef.current = null
      setBgReady(false)
      return
    }
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      bgRef.current = img
      setBgReady(true)
    }
    img.onerror = () => setBgReady(false)
    img.src = backgroundUrl
  }, [backgroundUrl])

  const allPoints = state.feet.flatMap((f) => f.contourMm)

  useEffect(() => {
    if (fitted.current || allPoints.length === 0 || size.w < 100) return
    setViewport(fitTo(boundingBox(allPoints), size.w, size.h))
    fitted.current = true
  }, [allPoints, size])

  const fitAll = useCallback(() => {
    const pts = state.feet.flatMap((f) => f.contourMm)
    if (pts.length) setViewport(fitTo(boundingBox(pts), size.w, size.h))
  }, [state.feet, size])

  const fitActive = useCallback(() => {
    const foot = state.feet[state.activeFootIndex]
    if (foot) setViewport(fitTo(boundingBox(foot.contourMm), size.w, size.h))
  }, [state.feet, state.activeFootIndex, size])

  // ------------------------------------------------------------------- desenho
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const dpr = Math.min(window.devicePixelRatio || 1, 2.5)
    canvas.width = Math.round(size.w * dpr)
    canvas.height = Math.round(size.h * dpr)
    canvas.style.width = `${size.w}px`
    canvas.style.height = `${size.h}px`
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, size.w, size.h)
    ctx.fillStyle = '#FFFFFF'
    ctx.fillRect(0, 0, size.w, size.h)

    if (showBackground && bgReady && bgRef.current && backgroundOriginMm && backgroundPxPerMm) {
      const img = bgRef.current
      const topLeft = toScreen(viewport, backgroundOriginMm)
      const w = (img.width / backgroundPxPerMm) * viewport.scale
      const h = (img.height / backgroundPxPerMm) * viewport.scale
      ctx.save()
      ctx.globalAlpha = 0.9
      ctx.imageSmoothingQuality = 'high'
      ctx.drawImage(img, topLeft.x, topLeft.y, w, h)
      ctx.restore()
    }

    state.feet.forEach((foot, index) => {
      drawFoot(ctx, foot, {
        viewport,
        tool: state.tool,
        selection: index === state.activeFootIndex ? state.selection : null,
        active: index === state.activeFootIndex,
        showBackground,
        dpr,
      })
    })
  }, [state, viewport, size, bgReady, showBackground, backgroundOriginMm, backgroundPxPerMm])

  // --------------------------------------------------------------- hit testing
  const hitTest = useCallback(
    (foot: EditableFoot, mm: PointMm, radiusMm: number): Selection | null => {
      const near = (p: PointMm) => Math.hypot(p.x - mm.x, p.y - mm.y) <= radiusMm

      if (state.tool === 'metatarsals') {
        for (const lm of foot.landmarks) {
          if ((lm.id.startsWith('M') || lm.id.startsWith('T')) && near(lm.positionMm)) {
            return { kind: 'landmark', id: lm.id }
          }
        }
      }
      if (state.tool === 'axis') {
        if (near(foot.axis.aMm)) return { kind: 'axisHandle', end: 'a' }
        if (near(foot.axis.bMm)) return { kind: 'axisHandle', end: 'b' }
      }
      if (state.tool === 'arches') {
        for (const arch of [foot.medialArch, foot.lateralArch]) {
          for (let i = 0; i < arch.controlPointsMm.length; i++) {
            if (near(arch.controlPointsMm[i])) {
              return { kind: 'archControl', arch: arch.id, index: i }
            }
          }
        }
      }
      if (state.tool === 'zones') {
        for (const zone of foot.supportZones) {
          for (let i = 0; i < zone.polygonMm.length; i++) {
            if (near(zone.polygonMm[i])) {
              return { kind: 'zoneVertex', zoneId: zone.id, index: i }
            }
          }
        }
        for (const zone of foot.supportZones) {
          if (pointInPolygon(mm, zone.polygonMm)) return { kind: 'zone', zoneId: zone.id }
        }
      }
      if (state.tool === 'contour') {
        for (let i = 0; i < foot.contourMm.length; i++) {
          if (near(foot.contourMm[i])) return { kind: 'contourNode', index: i }
        }
      }
      return null
    },
    [state.tool],
  )

  const footAt = useCallback(
    (mm: PointMm): number => {
      for (let i = 0; i < state.feet.length; i++) {
        if (pointInPolygon(mm, state.feet[i].contourMm)) return i
      }
      return -1
    },
    [state.feet],
  )

  // ------------------------------------------------------------------- eventos
  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return
    canvas.setPointerCapture(e.pointerId)
    const rect = canvas.getBoundingClientRect()
    const sx = e.clientX - rect.left
    const sy = e.clientY - rect.top
    pointers.current.set(e.pointerId, { x: sx, y: sy })

    if (pointers.current.size === 2) {
      const [a, b] = [...pointers.current.values()]
      pinch.current = {
        distance: Math.hypot(b.x - a.x, b.y - a.y),
        centerX: (a.x + b.x) / 2,
        centerY: (a.y + b.y) / 2,
      }
      drag.current = { kind: 'none' }
      return
    }

    const mm = toMm(viewport, sx, sy)
    const radiusMm = HIT_RADIUS_PX / viewport.scale
    const activeFoot = state.feet[state.activeFootIndex]

    if (e.shiftKey || e.button === 1 || state.tool === 'crop') {
      drag.current = { kind: 'pan', lastX: sx, lastY: sy }
      return
    }

    if (activeFoot) {
      const hit = hitTest(activeFoot, mm, radiusMm)
      if (hit) {
        onSelect(hit)
        const anchor = selectionAnchor(activeFoot, hit)
        drag.current = {
          kind: 'element',
          selection: hit,
          grabOffset: anchor ? { x: mm.x - anchor.x, y: mm.y - anchor.y } : { x: 0, y: 0 },
          moved: false,
        }
        return
      }
      if (state.tool === 'axis' && distanceToSegment(mm, activeFoot.axis.aMm, activeFoot.axis.bMm) <= radiusMm) {
        drag.current = { kind: 'axisBody', lastMm: mm, moved: false }
        return
      }
    }

    const idx = footAt(mm)
    if (idx >= 0 && idx !== state.activeFootIndex) {
      onActivateFoot(idx)
      drag.current = { kind: 'none' }
      return
    }
    onSelect(null)
    drag.current = { kind: 'pan', lastX: sx, lastY: sy }
  }

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const sx = e.clientX - rect.left
    const sy = e.clientY - rect.top
    if (pointers.current.has(e.pointerId)) pointers.current.set(e.pointerId, { x: sx, y: sy })

    if (pointers.current.size === 2 && pinch.current) {
      const [a, b] = [...pointers.current.values()]
      const dist = Math.hypot(b.x - a.x, b.y - a.y)
      const cx = (a.x + b.x) / 2
      const cy = (a.y + b.y) / 2
      const factor = dist / (pinch.current.distance || dist)
      setViewport((vp) =>
        pan(zoomAt(vp, factor, cx, cy), cx - pinch.current!.centerX, cy - pinch.current!.centerY),
      )
      pinch.current = { distance: dist, centerX: cx, centerY: cy }
      return
    }

    const d = drag.current
    if (d.kind === 'none') return

    if (d.kind === 'pan') {
      setViewport((vp) => pan(vp, sx - d.lastX, sy - d.lastY))
      drag.current = { kind: 'pan', lastX: sx, lastY: sy }
      return
    }

    const mm = toMm(viewport, sx, sy)

    if (d.kind === 'axisBody') {
      const next = applyToActive(state, (foot) =>
        translateAxis(foot, mm.x - d.lastMm.x, mm.y - d.lastMm.y),
      )
      drag.current = { kind: 'axisBody', lastMm: mm, moved: true }
      onLiveChange(next)
      return
    }

    const target = { x: mm.x - d.grabOffset.x, y: mm.y - d.grabOffset.y }
    const next = applyToActive(state, (foot) => applySelectionMove(foot, d.selection, target))
    drag.current = { ...d, moved: true }
    onLiveChange(next)
  }

  const onPointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    pointers.current.delete(e.pointerId)
    if (pointers.current.size < 2) pinch.current = null
    const d = drag.current
    drag.current = { kind: 'none' }
    if ((d.kind === 'element' || d.kind === 'axisBody') && d.moved) onCommit(state)
  }

  const onDoubleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    const foot = state.feet[state.activeFootIndex]
    if (!canvas || !foot) return
    const rect = canvas.getBoundingClientRect()
    const mm = toMm(viewport, e.clientX - rect.left, e.clientY - rect.top)
    const radiusMm = HIT_RADIUS_PX / viewport.scale

    if (state.tool === 'contour') {
      const hit = hitTest(foot, mm, radiusMm)
      if (hit?.kind === 'contourNode') {
        onCommit(applyToActive(state, (f) => removeContourNode(f, hit.index)))
        onSelect(null)
        return
      }
      const edge = nearestEdge(foot.contourMm, mm)
      if (edge.index >= 0 && edge.distance <= radiusMm * 1.6) {
        onCommit(applyToActive(state, (f) => insertContourNode(f, edge.index, edge.point)))
        onSelect({ kind: 'contourNode', index: edge.index + 1 })
      }
      return
    }

    if (state.tool === 'zones') {
      onCommit(applyToActive(state, (f) => addZone(f, mm)))
    }
  }

  const onWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const factor = Math.exp(-e.deltaY * 0.0016)
    setViewport((vp) => zoomAt(vp, factor, e.clientX - rect.left, e.clientY - rect.top))
  }

  return (
    <div ref={wrapRef} className="editor-canvas-wrap">
      <canvas
        ref={canvasRef}
        className="editor-canvas"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onDoubleClick={onDoubleClick}
        onWheel={onWheel}
      />
      <div className="editor-zoom-controls">
        <button type="button" aria-label="Aproximar" onClick={() => setViewport((v) => zoomAt(v, 1.25, size.w / 2, size.h / 2))}>
          +
        </button>
        <button type="button" aria-label="Afastar" onClick={() => setViewport((v) => zoomAt(v, 0.8, size.w / 2, size.h / 2))}>
          −
        </button>
        <button type="button" aria-label="Enquadrar pé ativo" onClick={fitActive}>
          ⤢
        </button>
        <button type="button" aria-label="Enquadrar tudo" onClick={fitAll}>
          ⛶
        </button>
      </div>
    </div>
  )
}

// -------------------------------------------------------------------- helpers

function applyToActive(
  state: EditorState,
  updater: (foot: EditableFoot) => EditableFoot,
): EditorState {
  return {
    ...state,
    feet: state.feet.map((f, i) => (i === state.activeFootIndex ? updater(f) : f)),
  }
}

function selectionAnchor(foot: EditableFoot, sel: Selection): PointMm | null {
  switch (sel.kind) {
    case 'contourNode':
      return foot.contourMm[sel.index] ?? null
    case 'landmark':
      return foot.landmarks.find((l) => l.id === sel.id)?.positionMm ?? null
    case 'axisHandle':
      return sel.end === 'a' ? foot.axis.aMm : foot.axis.bMm
    case 'archControl': {
      const arch = sel.arch === 'medial' ? foot.medialArch : foot.lateralArch
      return arch.controlPointsMm[sel.index] ?? null
    }
    case 'zoneVertex': {
      const zone = foot.supportZones.find((z) => z.id === sel.zoneId)
      return zone?.polygonMm[sel.index] ?? null
    }
    case 'zone': {
      const zone = foot.supportZones.find((z) => z.id === sel.zoneId)
      if (!zone) return null
      const cx = zone.polygonMm.reduce((s, p) => s + p.x, 0) / zone.polygonMm.length
      const cy = zone.polygonMm.reduce((s, p) => s + p.y, 0) / zone.polygonMm.length
      return { x: cx, y: cy }
    }
    default:
      return null
  }
}

function applySelectionMove(foot: EditableFoot, sel: Selection, target: PointMm): EditableFoot {
  switch (sel.kind) {
    case 'contourNode':
      return moveContourNode(foot, sel.index, target)
    case 'landmark':
      return moveLandmark(foot, sel.id, target)
    case 'axisHandle':
      return moveAxisHandle(foot, sel.end, target)
    case 'archControl':
      return moveArchControl(foot, sel.arch, sel.index, target)
    case 'zoneVertex':
      return moveZoneVertex(foot, sel.zoneId, sel.index, target)
    case 'zone': {
      const zone = foot.supportZones.find((z) => z.id === sel.zoneId)
      if (!zone) return foot
      const cx = zone.polygonMm.reduce((s, p) => s + p.x, 0) / zone.polygonMm.length
      const cy = zone.polygonMm.reduce((s, p) => s + p.y, 0) / zone.polygonMm.length
      return moveZone(foot, sel.zoneId, target.x - cx, target.y - cy)
    }
    default:
      return foot
  }
}

export function pointInPolygon(p: PointMm, poly: PointMm[]): boolean {
  let inside = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const a = poly[i]
    const b = poly[j]
    if (a.y > p.y !== b.y > p.y && p.x < ((b.x - a.x) * (p.y - a.y)) / (b.y - a.y) + a.x) {
      inside = !inside
    }
  }
  return inside
}

function distanceToSegment(p: PointMm, a: PointMm, b: PointMm): number {
  const abx = b.x - a.x
  const aby = b.y - a.y
  const len2 = abx * abx + aby * aby
  const t = len2 < 1e-12 ? 0 : Math.max(0, Math.min(1, ((p.x - a.x) * abx + (p.y - a.y) * aby) / len2))
  return Math.hypot(p.x - (a.x + t * abx), p.y - (a.y + t * aby))
}

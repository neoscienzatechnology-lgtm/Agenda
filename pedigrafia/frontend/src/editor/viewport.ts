/**
 * Transformação de *view* (mm ↔ pixels de tela).
 *
 * Esta é a ÚNICA parte do frontend autorizada a multiplicar coordenadas por um
 * fator de escala, e ela nunca escreve no modelo: converte mm→px para desenhar e
 * px→mm para interpretar o ponteiro. Zoom não altera nenhuma medida.
 */

import type { PointMm } from '../geom/polygon'

export interface Viewport {
  scale: number // px de tela por mm
  offsetX: number // px
  offsetY: number
}

export const MIN_SCALE = 0.25
export const MAX_SCALE = 26

export function createViewport(): Viewport {
  return { scale: 1, offsetX: 0, offsetY: 0 }
}

export function toScreen(vp: Viewport, p: PointMm): { x: number; y: number } {
  return { x: p.x * vp.scale + vp.offsetX, y: p.y * vp.scale + vp.offsetY }
}

export function toMm(vp: Viewport, x: number, y: number): PointMm {
  return { x: (x - vp.offsetX) / vp.scale, y: (y - vp.offsetY) / vp.scale }
}

export function pxToMm(vp: Viewport, px: number): number {
  return px / vp.scale
}

export function fitTo(
  bounds: { min: PointMm; max: PointMm },
  widthPx: number,
  heightPx: number,
  paddingPx = 28,
): Viewport {
  const w = Math.max(bounds.max.x - bounds.min.x, 1e-3)
  const h = Math.max(bounds.max.y - bounds.min.y, 1e-3)
  const scale = Math.min(
    (widthPx - 2 * paddingPx) / w,
    (heightPx - 2 * paddingPx) / h,
  )
  const clamped = Math.max(MIN_SCALE, Math.min(MAX_SCALE, scale))
  return {
    scale: clamped,
    offsetX: widthPx / 2 - ((bounds.min.x + bounds.max.x) / 2) * clamped,
    offsetY: heightPx / 2 - ((bounds.min.y + bounds.max.y) / 2) * clamped,
  }
}

/** Zoom mantendo o ponto sob o cursor/dedo fixo na tela. */
export function zoomAt(vp: Viewport, factor: number, screenX: number, screenY: number): Viewport {
  const next = Math.max(MIN_SCALE, Math.min(MAX_SCALE, vp.scale * factor))
  const applied = next / vp.scale
  return {
    scale: next,
    offsetX: screenX - (screenX - vp.offsetX) * applied,
    offsetY: screenY - (screenY - vp.offsetY) * applied,
  }
}

export function pan(vp: Viewport, dx: number, dy: number): Viewport {
  return { ...vp, offsetX: vp.offsetX + dx, offsetY: vp.offsetY + dy }
}

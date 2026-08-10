/**
 * Primitivas de polígono em MILÍMETROS.
 *
 * Espelho exato de `backend/app/geometry/polygon.py`. Qualquer divergência entre os
 * dois quebra o teste de paridade (`tests/fixtures/measurement_parity.json`), que é
 * executado tanto pelo pytest quanto pelo vitest.
 *
 * REGRA: este arquivo nunca conhece pixels. Zoom e pan são matrizes de *view*
 * aplicadas só no momento de desenhar.
 */

export interface PointMm {
  x: number
  y: number
}

export type Vec2 = [number, number]

export function signedArea(pts: PointMm[]): number {
  if (pts.length < 3) return 0
  let acc = 0
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i]
    const b = pts[(i + 1) % pts.length]
    acc += a.x * b.y - b.x * a.y
  }
  return 0.5 * acc
}

export function area(pts: PointMm[]): number {
  return Math.abs(signedArea(pts))
}

export function perimeter(pts: PointMm[], closed = true): number {
  if (pts.length < 2) return 0
  let acc = 0
  const n = closed ? pts.length : pts.length - 1
  for (let i = 0; i < n; i++) {
    const a = pts[i]
    const b = pts[(i + 1) % pts.length]
    acc += Math.hypot(b.x - a.x, b.y - a.y)
  }
  return acc
}

export function centroid(pts: PointMm[]): PointMm {
  const a = signedArea(pts)
  if (Math.abs(a) < 1e-12) {
    const sx = pts.reduce((s, p) => s + p.x, 0)
    const sy = pts.reduce((s, p) => s + p.y, 0)
    return { x: sx / pts.length, y: sy / pts.length }
  }
  let cx = 0
  let cy = 0
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i]
    const q = pts[(i + 1) % pts.length]
    const cross = p.x * q.y - q.x * p.y
    cx += (p.x + q.x) * cross
    cy += (p.y + q.y) * cross
  }
  return { x: cx / (6 * a), y: cy / (6 * a) }
}

/** Reamostragem de arco uniforme — mesma ordem de operações do numpy. */
export function resampleClosed(pts: PointMm[], stepMm: number): PointMm[] {
  if (pts.length < 3 || stepMm <= 0) return pts
  const closed = [...pts, pts[0]]
  const seg: number[] = []
  for (let i = 0; i < closed.length - 1; i++) {
    seg.push(Math.hypot(closed[i + 1].x - closed[i].x, closed[i + 1].y - closed[i].y))
  }
  const total = seg.reduce((s, v) => s + v, 0)
  if (total < stepMm * 3) return pts

  const n = Math.max(8, Math.round(total / stepMm))
  const cum: number[] = [0]
  for (const s of seg) cum.push(cum[cum.length - 1] + s)

  const out: PointMm[] = []
  // Mesma aritmética de `np.linspace(0, total, n, endpoint=False)`: passo calculado
  // uma vez e multiplicado pelo índice. `(total * k) / n` divergiria no último ulp e
  // essa diferença se propaga até as áreas do índice de arco.
  const step = total / n
  for (let k = 0; k < n; k++) {
    const target = step * k
    // searchsorted(cum, target, side='right') - 1
    let idx = upperBound(cum, target) - 1
    if (idx < 0) idx = 0
    if (idx > seg.length - 1) idx = seg.length - 1
    const segLen = seg[idx] > 1e-12 ? seg[idx] : 1
    const frac = (target - cum[idx]) / segLen
    const a = closed[idx]
    const b = closed[idx + 1]
    out.push({ x: a.x + frac * (b.x - a.x), y: a.y + frac * (b.y - a.y) })
  }
  return out
}

function upperBound(arr: number[], value: number): number {
  let lo = 0
  let hi = arr.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (arr[mid] <= value) lo = mid + 1
    else hi = mid
  }
  return lo
}

export function project(pts: PointMm[], origin: PointMm, u: Vec2, v: Vec2): PointMm[] {
  return pts.map((p) => {
    const dx = p.x - origin.x
    const dy = p.y - origin.y
    return { x: dx * u[0] + dy * u[1], y: dx * v[0] + dy * v[1] }
  })
}

export function unproject(uv: PointMm[], origin: PointMm, u: Vec2, v: Vec2): PointMm[] {
  return uv.map((p) => ({
    x: origin.x + p.x * u[0] + p.y * v[0],
    y: origin.y + p.x * u[1] + p.y * v[1],
  }))
}

/** Valores de v onde o polígono local cruza a reta u = uValue. */
export function crossingsAtU(uv: PointMm[], uValue: number): number[] {
  const collect = (inclusiveUpper: boolean): number[] => {
    const out: number[] = []
    for (let i = 0; i < uv.length; i++) {
      const a = uv[i]
      const b = uv[(i + 1) % uv.length]
      const lo = Math.min(a.x, b.x)
      const hi = Math.max(a.x, b.x)
      const hit = inclusiveUpper ? lo < uValue && uValue <= hi : lo <= uValue && uValue < hi
      if (!hit) continue
      const denom = Math.abs(b.x - a.x) < 1e-12 ? 1 : b.x - a.x
      const t = (uValue - a.x) / denom
      out.push(a.y + t * (b.y - a.y))
    }
    return out
  }
  const first = collect(false)
  return first.length > 0 ? first : collect(true)
}

export function widthAtU(uv: PointMm[], uValue: number): number {
  const vs = crossingsAtU(uv, uValue)
  if (vs.length < 2) return 0
  return Math.max(...vs) - Math.min(...vs)
}

/** Sutherland–Hodgman: mantém dot(p, normal) <= offset. */
export function clipHalfPlane(pts: PointMm[], normal: Vec2, offset: number): PointMm[] {
  if (pts.length < 3) return []
  const out: PointMm[] = []
  for (let i = 0; i < pts.length; i++) {
    const cur = pts[i]
    const nxt = pts[(i + 1) % pts.length]
    const dc = cur.x * normal[0] + cur.y * normal[1] - offset
    const dn = nxt.x * normal[0] + nxt.y * normal[1] - offset
    if (dc <= 0) out.push(cur)
    if (dc > 0 !== dn > 0) {
      const denom = dc - dn
      if (Math.abs(denom) > 1e-12) {
        const t = dc / denom
        out.push({ x: cur.x + t * (nxt.x - cur.x), y: cur.y + t * (nxt.y - cur.y) })
      }
    }
  }
  return out.length >= 3 ? out : []
}

export function clipBandU(uv: PointMm[], uLo: number, uHi: number): PointMm[] {
  const first = clipHalfPlane(uv, [1, 0], uHi)
  if (first.length < 3) return first
  return clipHalfPlane(first, [-1, 0], -uLo)
}

export function pointLineDistance(p: PointMm, a: PointMm, b: PointMm): number {
  const abx = b.x - a.x
  const aby = b.y - a.y
  const n = Math.hypot(abx, aby)
  if (n < 1e-12) return Math.hypot(p.x - a.x, p.y - a.y)
  return Math.abs(abx * (p.y - a.y) - aby * (p.x - a.x)) / n
}

export function distance(a: PointMm, b: PointMm): number {
  return Math.hypot(b.x - a.x, b.y - a.y)
}

/** Índice do ponto mais próximo dentro de um raio (mm). -1 se nenhum. */
export function nearestIndex(pts: PointMm[], target: PointMm, radiusMm: number): number {
  let best = -1
  let bestD = radiusMm
  for (let i = 0; i < pts.length; i++) {
    const d = distance(pts[i], target)
    if (d <= bestD) {
      bestD = d
      best = i
    }
  }
  return best
}

/** Aresta mais próxima de um ponto: usado para inserir um nó no contorno. */
export function nearestEdge(
  pts: PointMm[],
  target: PointMm,
): { index: number; distance: number; point: PointMm } {
  let best = { index: -1, distance: Infinity, point: target }
  for (let i = 0; i < pts.length; i++) {
    const a = pts[i]
    const b = pts[(i + 1) % pts.length]
    const abx = b.x - a.x
    const aby = b.y - a.y
    const len2 = abx * abx + aby * aby
    const t = len2 < 1e-12 ? 0 : Math.max(0, Math.min(1, ((target.x - a.x) * abx + (target.y - a.y) * aby) / len2))
    const px = a.x + t * abx
    const py = a.y + t * aby
    const d = Math.hypot(target.x - px, target.y - py)
    if (d < best.distance) best = { index: i, distance: d, point: { x: px, y: py } }
  }
  return best
}

export function boundingBox(pts: PointMm[]): { min: PointMm; max: PointMm } {
  const xs = pts.map((p) => p.x)
  const ys = pts.map((p) => p.y)
  return {
    min: { x: Math.min(...xs), y: Math.min(...ys) },
    max: { x: Math.max(...xs), y: Math.max(...ys) },
  }
}

/** Monotone chain — vértices do fecho convexo em sentido anti-horário. */
export function convexHull(pts: PointMm[]): PointMm[] {
  if (pts.length < 3) return [...pts]
  const sorted = [...pts].sort((a, b) => (a.x === b.x ? a.y - b.y : a.x - b.x))
  const cross = (o: PointMm, a: PointMm, b: PointMm) =>
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)

  const build = (seq: PointMm[]): PointMm[] => {
    const out: PointMm[] = []
    for (const pt of seq) {
      while (out.length >= 2 && cross(out[out.length - 2], out[out.length - 1], pt) <= 1e-12) {
        out.pop()
      }
      out.push(pt)
    }
    return out
  }
  const lower = build(sorted)
  const upper = build([...sorted].reverse())
  return [...lower.slice(0, -1), ...upper.slice(0, -1)]
}

/**
 * Diâmetro do fecho convexo (rotating calipers) — o COMPRIMENTO oficial do pé.
 * Espelha `polygon.max_caliper` do backend, inclusive na ordem de varredura.
 */
export function maxCaliper(pts: PointMm[]): number {
  const h = convexHull(pts)
  if (h.length < 2) return 0
  let best = 0
  let j = 1
  const n = h.length
  for (let i = 0; i < n; i++) {
    for (;;) {
      const next = (j + 1) % n
      if (distance(h[next], h[i]) > distance(h[j], h[i])) j = next
      else break
    }
    const d = distance(h[j], h[i])
    if (d > best) best = d
  }
  return best
}

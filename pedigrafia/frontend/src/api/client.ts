/** Cliente HTTP. Nenhuma resposta da API é cacheada — medida em cache é medida errada. */

import type {
  AnalyzeResponse,
  ApproveResponse,
  CaptureRejected,
  MeasureResponse,
  ReviewedFoot,
  ViewPoint,
} from '../types/pedigrafia'

const BASE = import.meta.env.VITE_API_BASE ?? ''

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly code = 'error',
    readonly payload?: unknown,
  ) {
    super(message)
  }
}

export class CaptureRejectedError extends ApiError {
  constructor(readonly rejection: CaptureRejected) {
    super(rejection.message, 422, 'capture_rejected', rejection)
  }
}

async function parseError(res: Response): Promise<never> {
  let body: any = null
  try {
    body = await res.json()
  } catch {
    /* corpo não-JSON */
  }
  if (res.status === 422 && body?.code === 'capture_rejected') {
    throw new CaptureRejectedError(body as CaptureRejected)
  }
  const detail = body?.detail ?? body
  throw new ApiError(
    detail?.message ?? `Falha na requisição (${res.status}).`,
    res.status,
    detail?.code ?? 'error',
    body,
  )
}

/** De onde vem a escala física. `aruco` = alvo impresso; o resto são objetos de
 *  dimensão normalizada. Declarar é obrigatório na prática: a razão largura/altura
 *  de um cartão e a de uma folha A4 ficam a ~11 % uma da outra, e trocá-las erraria
 *  a escala em 2,45× sem nenhum outro sintoma. */
export type CalibrationSource = 'auto' | 'aruco' | 'card' | 'a4' | 'a5'

export interface AnalyzeOptions {
  view?: ViewPoint
  shoeSize?: string
  shoeSizeSystem?: string
  detectCallosities?: boolean
  calibration?: CalibrationSource
  signal?: AbortSignal
}

export async function analyze(file: File | Blob, opts: AnalyzeOptions = {}): Promise<AnalyzeResponse> {
  const form = new FormData()
  form.append('image', file, 'captura')
  if (opts.view) form.append('view', opts.view)
  if (opts.shoeSize) form.append('shoeSize', opts.shoeSize)
  if (opts.shoeSizeSystem) form.append('shoeSizeSystem', opts.shoeSizeSystem)
  form.append('detectCallosities', String(opts.detectCallosities ?? true))
  form.append('calibration', opts.calibration ?? 'auto')

  const res = await fetch(`${BASE}/api/analyze`, {
    method: 'POST',
    body: form,
    cache: 'no-store',
    signal: opts.signal,
  })
  if (!res.ok) await parseError(res)
  return res.json()
}

export async function measure(foot: ReviewedFoot, view: ViewPoint): Promise<MeasureResponse> {
  const res = await fetch(`${BASE}/api/measure`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ foot, view }),
    cache: 'no-store',
  })
  if (!res.ok) await parseError(res)
  return res.json()
}

export async function approve(sessionId: string, feet: ReviewedFoot[]): Promise<ApproveResponse> {
  const res = await fetch(`${BASE}/api/review/approve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessionId, feet }),
    cache: 'no-store',
  })
  if (!res.ok) await parseError(res)
  return res.json()
}

export interface ExportOptions {
  includeSupportZones?: boolean
  includeArches?: boolean
  includeMetatarsals?: boolean
  includeAxis?: boolean
  includeMeasurements?: boolean
  patientLabel?: string
}

export async function exportPdf(
  sessionId: string,
  reviewToken: string,
  foot: ReviewedFoot,
  view: ViewPoint,
  opts: ExportOptions = {},
): Promise<{ blob: Blob; lengthMm: number; fitsA4: boolean }> {
  const res = await fetch(`${BASE}/api/export-pdf`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      sessionId,
      reviewToken,
      foot,
      view,
      includeSupportZones: opts.includeSupportZones ?? true,
      includeArches: opts.includeArches ?? true,
      includeMetatarsals: opts.includeMetatarsals ?? true,
      includeAxis: opts.includeAxis ?? true,
      includeMeasurements: opts.includeMeasurements ?? true,
      patientLabel: opts.patientLabel ?? '',
    }),
    cache: 'no-store',
  })
  if (!res.ok) await parseError(res)
  return {
    blob: await res.blob(),
    lengthMm: Number(res.headers.get('X-Pedigrafia-Length-Mm') ?? '0'),
    fitsA4: res.headers.get('X-Pedigrafia-Fits-A4') !== 'false',
  }
}

export async function renderAnnotated(
  sessionId: string,
  feet: ReviewedFoot[],
  view: ViewPoint,
): Promise<Blob> {
  const res = await fetch(`${BASE}/api/render-annotated`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sessionId, feet, view, scalePxPerMm: 8, includeCallosities: true }),
    cache: 'no-store',
  })
  if (!res.ok) await parseError(res)
  return res.blob()
}

export async function deleteSession(sessionId: string): Promise<void> {
  await fetch(`${BASE}/api/session/${sessionId}`, { method: 'DELETE', cache: 'no-store' }).catch(
    () => undefined,
  )
}

export function markerPdfUrl(markerId = 7): string {
  return `${BASE}/api/marker.pdf?markerId=${markerId}`
}

/** Alvo de quatro marcadores — o layout recomendado, que evita extrapolação. */
export function targetPdfUrl(target = 'board4'): string {
  return `${BASE}/api/marker.pdf?target=${encodeURIComponent(target)}`
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 4000)
}

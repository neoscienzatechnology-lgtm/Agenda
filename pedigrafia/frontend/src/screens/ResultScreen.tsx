import { useState } from 'react'

import * as api from '../api/client'
import { computeMeasurements, frameFromAxisPoints, mm, num } from '../geom/measure'
import type { AnalyzeResponse } from '../types/pedigrafia'
import type { ApprovedState } from '../App'

export function ResultScreen({
  analysis,
  approved,
  onBackToReview,
  onNew,
}: {
  analysis: AnalyzeResponse
  approved: ApprovedState
  onBackToReview: () => void
  onNew: () => void
}) {
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [warning, setWarning] = useState<string | null>(null)

  const downloadPdf = async (index: number) => {
    const foot = approved.feet[index]
    setBusy(`pdf-${index}`)
    setError(null)
    try {
      const { blob, lengthMm, fitsA4 } = await api.exportPdf(
        analysis.sessionId,
        approved.reviewToken,
        foot,
        analysis.view,
      )
      const side = foot.laterality === 'right' ? 'direito' : 'esquerdo'
      api.downloadBlob(blob, `pedigrafia-${side}-${lengthMm.toFixed(0)}mm-1x1.pdf`)
      if (!fitsA4) {
        setWarning(
          'O contorno excede a área útil da folha A4. A escala foi mantida em 1:1 — não reduza na impressão.',
        )
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Falha ao gerar o PDF.')
    } finally {
      setBusy(null)
    }
  }

  const downloadImage = async () => {
    setBusy('img')
    setError(null)
    try {
      const blob = await api.renderAnnotated(analysis.sessionId, approved.feet, analysis.view)
      api.downloadBlob(blob, 'pedigrafia-anotada.png')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Falha ao gerar a imagem.')
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="result-screen">
      <header className="result-header">
        <span className="pill success">Pedigrafia concluída</span>
        <h2>Medidas aprovadas</h2>
        <p className="muted">
          Valores obtidos por calibração geométrica do marcador de 50 × 50 mm.
        </p>
      </header>

      <div className="result-grid">
        {approved.feet.map((foot, index) => {
          const frame = frameFromAxisPoints(foot.contourMm, foot.axis.aMm, foot.axis.bMm)
          const m1 = foot.landmarks.find((l) => l.id === 'M1')?.positionMm ?? null
          const m5 = foot.landmarks.find((l) => l.id === 'M5')?.positionMm ?? null
          const measures = computeMeasurements(foot.contourMm, frame, m1, m5, foot.toeCutT)
          return (
            <article key={foot.id} className="result-card">
              <h3>{foot.laterality === 'right' ? 'Pé direito' : 'Pé esquerdo'}</h3>
              <dl className="measure-list">
                <Row label="Comprimento" value={mm(measures.lengthMm)} strong />
                <Row label="Largura do antepé" value={mm(measures.forefootWidthMm)} />
                <Row label="Largura do mediopé" value={mm(measures.midfootWidthMm)} />
                <Row label="Largura do calcâneo" value={mm(measures.heelWidthMm)} />
                <Row label="Calcâneo → metatarsos" value={mm(measures.heelToMetatarsalLineMm)} />
                <Row label="Índice de arco (projeção)" value={num(measures.archIndex)} />
              </dl>
              <button
                className="primary"
                type="button"
                disabled={busy !== null}
                onClick={() => void downloadPdf(index)}
              >
                {busy === `pdf-${index}`
                  ? 'Gerando…'
                  : `Pé ${foot.laterality === 'right' ? 'direito' : 'esquerdo'} — PDF 1:1`}
              </button>
            </article>
          )
        })}
      </div>

      <div className="print-notice">
        <h3>Para preservar a escala 1:1</h3>
        <p>
          Selecione <strong>Tamanho real / 100%</strong>. Não utilize “Ajustar”, “Fit”,
          “Encolher para caber” nem “Ampliar”. Cada folha A4 contém um único pé.
        </p>
        <p className="muted small">
          Confira depois de imprimir: meça o comprimento com régua; ele deve coincidir com
          o valor acima.
        </p>
      </div>

      {warning && <div className="banner warn">{warning}</div>}
      {error && <div className="banner error">{error}</div>}

      <div className="result-actions">
        <button className="secondary" type="button" disabled={busy !== null} onClick={() => void downloadImage()}>
          {busy === 'img' ? 'Gerando…' : 'Baixar imagem anotada'}
        </button>
        <button className="ghost" type="button" onClick={onBackToReview}>
          Voltar à revisão
        </button>
        <button className="ghost" type="button" onClick={onNew}>
          Nova pedigrafia
        </button>
      </div>

      <p className="note small">
        A imagem anotada é para análise visual. O arquivo metrológico de fabricação é o PDF
        A4 em escala 1:1.
      </p>
    </div>
  )
}

function Row({ label, value, strong }: { label: string; value: string; strong?: boolean }) {
  return (
    <div className={strong ? 'measure strong' : 'measure'}>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  )
}

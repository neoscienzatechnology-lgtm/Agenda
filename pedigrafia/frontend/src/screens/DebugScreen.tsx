/**
 * Modo de validação visual — NÃO faz parte do fluxo principal.
 * Existe para ajustar o algoritmo: mostra a cadeia metrológica inteira em números.
 */

import { QualityReport } from '../components/QualityReport'
import type { AnalyzeResponse } from '../types/pedigrafia'

export function DebugScreen({
  analysis,
  onClose,
}: {
  analysis: AnalyzeResponse
  onClose: () => void
}) {
  const m = analysis.marker
  const r = analysis.rectification
  const c = analysis.calibration
  return (
    <div className="debug-screen">
      <div className="debug-head">
        <h2>Validação visual</h2>
        <button className="ghost small" type="button" onClick={onClose}>
          Voltar à revisão
        </button>
      </div>

      <section className="debug-section">
        <h3>Referência e escala</h3>
        <dl className="kv">
          <Item k="Detectado" v={m.detected ? 'sim' : 'não'} />
          <Item k="Origem da escala" v={c?.source === 'reference' ? 'objeto normalizado' : 'marcador impresso'} />
          <Item k="Dicionário / id" v={`${m.dictionary} · ${m.markerId}`} />
          <Item
            k="Dimensão física"
            v={
              c?.reference
                ? `${c.reference.widthMm} × ${c.reference.heightMm} mm (${c.reference.standard}, ±${c.reference.toleranceMm} mm)`
                : `${m.sizeMm.toFixed(2)} mm de aresta`
            }
          />
          <Item
            k="Incerteza herdada do padrão"
            v={c ? `± ${c.scaleToleranceMm.toFixed(2)} mm (${(c.scaleToleranceRel * 100).toFixed(2)} %)` : '—'}
          />
          <Item k="Amostragem na foto" v={`${m.srcPxPerMm.toFixed(2)} px/mm`} />
          <Item k="Distorção do quadrilátero" v={m.skew.toFixed(4)} />
          <Item k="Inclinação estimada" v={`${m.tiltDeg.toFixed(1)}°`} />
          <Item k="Confiança" v={`${Math.round(m.confidence * 100)}%`} />
          <Item
            k="Lados re-medidos na retificada"
            v={m.roundTripSideMm.map((s) => s.toFixed(3)).join(' · ') || '—'}
          />
          <Item
            k="Erro de ida-e-volta"
            v={m.roundTripErrorMm >= 0 ? `${m.roundTripErrorMm.toFixed(4)} mm` : '—'}
          />
        </dl>
      </section>

      {r && (
        <section className="debug-section">
          <h3>Retificação</h3>
          <dl className="kv">
            <Item k="Escala do raster" v={`${r.pxPerMm.toFixed(4)} px/mm`} />
            <Item k="Origem" v={`(${r.originMm.x.toFixed(2)}, ${r.originMm.y.toFixed(2)}) mm`} />
            <Item k="Tamanho" v={`${r.widthPx} × ${r.heightPx} px`} />
          </dl>
          <pre className="matrix">
            {chunk(r.homographyImageToMm, 3)
              .map((row) => row.map((v) => v.toExponential(6).padStart(15)).join(' '))
              .join('\n')}
          </pre>
          {analysis.rectifiedImageUrl && (
            <img className="debug-image" src={analysis.rectifiedImageUrl} alt="Imagem retificada" />
          )}
        </section>
      )}

      <section className="debug-section">
        <h3>Qualidade da captura — {analysis.captureQuality.score.toFixed(1)}/100</h3>
        <QualityReport quality={analysis.captureQuality} />
      </section>

      {analysis.feet.map((foot) => (
        <section className="debug-section" key={foot.id}>
          <h3>
            {foot.id} — {foot.laterality} ({Math.round(foot.lateralityConfidence * 100)}%)
          </h3>
          <dl className="kv">
            <Item k="Método de lateralidade" v={foot.lateralityMethod} />
            <Item k="Nós editáveis / alta resolução" v={`${foot.contourMm.length} / ${foot.contourHighResMm.length}`} />
            <Item k="Orientação do eixo" v={`${foot.frame.orientationDeg.toFixed(2)}°`} />
            <Item k="Sinal medial" v={String(foot.frame.medialSign)} />
            <Item k="Corte dos dedos" v={foot.toeCutT.toFixed(3)} />
            <Item k="Área plantar" v={`${foot.measurements.plantarAreaMm2.toFixed(0)} mm²`} />
            <Item k="Base do índice de arco" v={foot.measurements.archIndexBasis} />
            <Item k="Áreas A/B/C" v={foot.measurements.archAreasMm2.map((a) => a.toFixed(0)).join(' · ')} />
          </dl>
          <h4>Landmarks (frame do pé)</h4>
          <table className="debug-table">
            <thead>
              <tr>
                <th>id</th>
                <th>u (mm)</th>
                <th>v (mm)</th>
                <th>t (%)</th>
                <th>confiança</th>
                <th>método</th>
              </tr>
            </thead>
            <tbody>
              {foot.landmarksFootFrame.map((c) => {
                const lm = foot.landmarks.find((l) => l.id === c.id)
                return (
                  <tr key={c.id} className={lm && lm.confidence < 0.55 ? 'low' : ''}>
                    <td>{c.id}</td>
                    <td>{c.uMm.toFixed(2)}</td>
                    <td>{c.vMm.toFixed(2)}</td>
                    <td>{c.tPct.toFixed(1)}</td>
                    <td>{lm ? `${Math.round(lm.confidence * 100)}%` : '—'}</td>
                    <td>{lm?.method ?? '—'}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </section>
      ))}

      <section className="debug-section">
        <h3>Tempos por etapa (ms)</h3>
        <dl className="kv">
          {Object.entries(analysis.timingsMs).map(([k, v]) => (
            <Item key={k} k={k} v={v.toFixed(1)} />
          ))}
        </dl>
      </section>
    </div>
  )
}

function Item({ k, v }: { k: string; v: string }) {
  return (
    <div className="kv-row">
      <dt>{k}</dt>
      <dd>{v}</dd>
    </div>
  )
}

function chunk<T>(arr: T[], size: number): T[][] {
  const out: T[][] = []
  for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size))
  return out
}

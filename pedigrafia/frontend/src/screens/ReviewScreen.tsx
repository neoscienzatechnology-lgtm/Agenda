import { useCallback, useEffect, useMemo, useState } from 'react'

import * as api from '../api/client'
import { EditorCanvas } from '../editor/EditorCanvas'
import { computeMeasurements, frameFromAxisPoints, mm, num } from '../geom/measure'
import {
  canRedo,
  canUndo,
  commit,
  fromAnalysis,
  initHistory,
  landmarkOf,
  live,
  redo,
  removeCallosity,
  restoreAutomatic,
  setLaterality,
  setToeCut,
  toReviewed,
  toggleCallosity,
  undo,
  updateFoot,
  type EditorState,
  type History,
  type Selection,
  type ToolId,
} from '../state/editorState'
import type { AnalyzeResponse } from '../types/pedigrafia'
import type { ApprovedState } from '../App'

const TOOLS: { id: ToolId; label: string; hint: string }[] = [
  { id: 'contour', label: 'Contorno', hint: 'Arraste os nós. Duplo toque: adicionar ou remover nó.' },
  { id: 'axis', label: 'Eixo', hint: 'Arraste as pontas para girar; arraste a linha para deslocar.' },
  { id: 'metatarsals', label: 'Metatarsos', hint: 'Arraste M1–M5 e os ápices T1–T5.' },
  { id: 'arches', label: 'Arcos', hint: 'Arraste os pontos de controle das curvas medial e lateral.' },
  { id: 'zones', label: 'Zonas de apoio', hint: 'Arraste, redimensione ou crie (duplo toque) uma zona.' },
  { id: 'crop', label: 'Recorte', hint: 'Ajuste a linha de corte dos pododáctilos e confira o molde final.' },
]

export function ReviewScreen({
  analysis,
  onApproved,
}: {
  analysis: AnalyzeResponse
  onApproved: (state: ApprovedState) => void
}) {
  const [history, setHistory] = useState<History>(() =>
    initHistory({
      feet: analysis.feet.map(fromAnalysis),
      activeFootIndex: 0,
      tool: 'contour',
      selection: null,
    }),
  )
  const [showBackground, setShowBackground] = useState(true)
  const [panelOpen, setPanelOpen] = useState(false)
  const [approving, setApproving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const state = history.present
  const foot = state.feet[state.activeFootIndex]

  const setLive = useCallback((next: EditorState) => setHistory((h) => live(h, next)), [])
  const setCommit = useCallback((next: EditorState) => setHistory((h) => commit(h, next)), [])
  const setSelection = useCallback(
    (selection: Selection | null) => setHistory((h) => live(h, { ...h.present, selection })),
    [],
  )
  const setTool = useCallback(
    (tool: ToolId) => setHistory((h) => live(h, { ...h.present, tool, selection: null })),
    [],
  )
  const setActiveFoot = useCallback(
    (index: number) =>
      setHistory((h) => live(h, { ...h.present, activeFootIndex: index, selection: null })),
    [],
  )
  const editActive = useCallback(
    (updater: Parameters<typeof updateFoot>[2]) =>
      setHistory((h) => commit(h, updateFoot(h.present, h.present.activeFootIndex, updater))),
    [],
  )

  // Medidas ao vivo, calculadas localmente pelo mesmo algoritmo do servidor.
  const liveMeasurements = useMemo(() => {
    if (!foot || foot.contourMm.length < 3) return null
    const frame = frameFromAxisPoints(foot.contourMm, foot.axis.aMm, foot.axis.bMm)
    return computeMeasurements(
      foot.contourMm,
      frame,
      landmarkOf(foot, 'M1'),
      landmarkOf(foot, 'M5'),
      foot.toeCutT,
    )
  }, [foot])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault()
        setHistory((h) => (e.shiftKey ? redo(h) : undo(h)))
      }
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') {
        e.preventDefault()
        setHistory(redo)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const approve = async () => {
    setApproving(true)
    setError(null)
    try {
      const feet = state.feet.map(toReviewed)
      const res = await api.approve(analysis.sessionId, feet)
      onApproved({ reviewToken: res.reviewToken, feet })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Não foi possível aprovar.')
    } finally {
      setApproving(false)
    }
  }

  if (!foot) return <p className="centered-state">Nenhum pé identificado nesta captura.</p>

  const activeTool = TOOLS.find((t) => t.id === state.tool)!
  const lowConfidence = foot.landmarks.filter(
    (l) => (l.id.startsWith('M') || l.id.startsWith('T')) && l.confidence < 0.55,
  )

  return (
    <div className="review-screen">
      <div className="review-toolbar">
        <div className="tool-group" role="tablist" aria-label="Ferramentas de revisão">
          {TOOLS.map((t) => (
            <button
              key={t.id}
              role="tab"
              aria-selected={state.tool === t.id}
              className={state.tool === t.id ? 'tool active' : 'tool'}
              type="button"
              onClick={() => setTool(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>
        <div className="tool-group right">
          <button className="ghost small" type="button" disabled={!canUndo(history)} onClick={() => setHistory(undo)}>
            Desfazer
          </button>
          <button className="ghost small" type="button" disabled={!canRedo(history)} onClick={() => setHistory(redo)}>
            Refazer
          </button>
          <button className="ghost small" type="button" onClick={() => setHistory(restoreAutomatic)}>
            Restaurar automático
          </button>
          <label className="toggle small">
            <input
              type="checkbox"
              checked={showBackground}
              onChange={(e) => setShowBackground(e.target.checked)}
            />
            Foto
          </label>
        </div>
      </div>

      <p className="tool-hint">{activeTool.hint}</p>

      <div className="review-body">
        <EditorCanvas
          state={state}
          backgroundUrl={showBackground ? analysis.rectifiedImageUrl : null}
          backgroundOriginMm={analysis.rectification?.originMm ?? null}
          backgroundPxPerMm={analysis.rectification?.pxPerMm ?? null}
          showBackground={showBackground}
          onLiveChange={setLive}
          onCommit={setCommit}
          onSelect={setSelection}
          onActivateFoot={setActiveFoot}
        />

        <aside className={panelOpen ? 'review-panel open' : 'review-panel'}>
          <button
            className="panel-handle"
            type="button"
            onClick={() => setPanelOpen((v) => !v)}
            aria-expanded={panelOpen}
          >
            {panelOpen ? 'Ocultar medidas' : 'Medidas e marcações'}
          </button>

          <div className="panel-content">
            {state.feet.length > 1 && (
              <div className="foot-switch">
                {state.feet.map((f, i) => (
                  <button
                    key={f.id}
                    type="button"
                    className={i === state.activeFootIndex ? 'active' : ''}
                    onClick={() => setActiveFoot(i)}
                  >
                    {f.laterality === 'right' ? 'Direito' : 'Esquerdo'}
                  </button>
                ))}
              </div>
            )}

            <section className="panel-section">
              <h3>Lateralidade</h3>
              <div className="segmented">
                <button
                  type="button"
                  className={foot.laterality === 'right' ? 'active' : ''}
                  onClick={() => editActive((f) => setLaterality(f, 'right'))}
                >
                  Direito
                </button>
                <button
                  type="button"
                  className={foot.laterality === 'left' ? 'active' : ''}
                  onClick={() => editActive((f) => setLaterality(f, 'left'))}
                >
                  Esquerdo
                </button>
              </div>
              <Confidence value={foot.lateralityConfidence} label="Confiança da classificação" />
            </section>

            <section className="panel-section">
              <h3>Medidas em tempo real</h3>
              {liveMeasurements && (
                <dl className="measure-list">
                  <Measure label="Comprimento" value={mm(liveMeasurements.lengthMm)} strong />
                  <Measure label="Largura do antepé" value={mm(liveMeasurements.forefootWidthMm)} />
                  <Measure label="Largura do mediopé" value={mm(liveMeasurements.midfootWidthMm)} />
                  <Measure label="Largura do calcâneo" value={mm(liveMeasurements.heelWidthMm)} />
                  <Measure
                    label="Calcâneo → linha metatarsal"
                    value={mm(liveMeasurements.heelToMetatarsalLineMm)}
                  />
                  <Measure
                    label="Índice de arco (projeção 2D)"
                    value={num(liveMeasurements.archIndex)}
                  />
                  <Measure
                    label="Área plantar"
                    value={`${(liveMeasurements.plantarAreaMm2 / 100).toFixed(1).replace('.', ',')} cm²`}
                  />
                </dl>
              )}
              <p className="note small">
                O índice de arco é calculado sobre a <strong>silhueta</strong> plantar, não
                sobre a área de contato: não é comparável a valores de pedigrafia de tinta
                nem representa altura de arco.
              </p>
            </section>

            {state.tool === 'crop' && (
              <section className="panel-section">
                <h3>Linha de corte dos pododáctilos</h3>
                <input
                  type="range"
                  min={0.55}
                  max={0.97}
                  step={0.005}
                  value={foot.toeCutT}
                  onChange={(e) => editActive((f) => setToeCut(f, Number(e.target.value)))}
                />
                <p className="muted small">
                  Posição: {(foot.toeCutT * 100).toFixed(1).replace('.', ',')}% do comprimento.
                  Define a região usada no índice de arco.
                </p>
              </section>
            )}

            {lowConfidence.length > 0 && (
              <section className="panel-section warn-section">
                <h3>Marcações a verificar</h3>
                <ul className="uncertain-list">
                  {lowConfidence.map((l) => (
                    <li key={l.id}>
                      <strong>{l.id}</strong>
                      <span>Confiança: {Math.round(l.confidence * 100)}%</span>
                      <em>Verifique esta marcação</em>
                    </li>
                  ))}
                </ul>
              </section>
            )}

            {foot.callosityHints.length > 0 && (
              <section className="panel-section">
                <h3>Sugestões visuais de hiperqueratose</h3>
                <p className="note small">Sugestão visual — não constitui diagnóstico.</p>
                <ul className="hint-list">
                  {foot.callosityHints.map((c) => (
                    <li key={c.id}>
                      <label className="toggle">
                        <input
                          type="checkbox"
                          checked={c.accepted}
                          onChange={() => editActive((f) => toggleCallosity(f, c.id))}
                        />
                        {c.areaMm2.toFixed(0)} mm² · {Math.round(c.confidence * 100)}%
                      </label>
                      <button
                        className="ghost small"
                        type="button"
                        onClick={() => editActive((f) => removeCallosity(f, c.id))}
                      >
                        Remover
                      </button>
                    </li>
                  ))}
                </ul>
              </section>
            )}

            {analysis.calibration && (
              <section
                className={
                  analysis.calibration.interpolated
                    ? 'panel-section'
                    : 'panel-section warn-section'
                }
              >
                <h3>Calibração</h3>
                <dl className="measure-list">
                  <Measure
                    label="Marcadores usados"
                    value={String(analysis.calibration.markerCount)}
                  />
                  <Measure
                    label="Escala na região dos pés"
                    value={analysis.calibration.interpolated ? 'interpolada' : 'extrapolada'}
                    strong={!analysis.calibration.interpolated}
                  />
                  {!analysis.calibration.interpolated && (
                    <Measure
                      label="Distância fora da área calibrada"
                      value={mm(analysis.calibration.extrapolationMm)}
                    />
                  )}
                  {!analysis.calibration.exact && (
                    <Measure
                      label="Resíduo do ajuste"
                      value={mm(analysis.calibration.residualMaxMm, 2)}
                    />
                  )}
                </dl>
                {!analysis.calibration.interpolated && (
                  <p className="note small">
                    A escala está sendo extrapolada a partir de um único marcador: a
                    exatidão cai com a distância até ele. Para medidas de fabricação,
                    use o alvo de quatro marcadores ao redor da área de apoio.
                  </p>
                )}
              </section>
            )}

            <section className="panel-section">
              <h3>Confiança da detecção</h3>
              <Confidence value={analysis.feet[state.activeFootIndex]?.confidence.segmentation ?? 0} label="Segmentação" />
              <Confidence value={analysis.feet[state.activeFootIndex]?.confidence.marker ?? 0} label="Marcador" />
              <Confidence value={analysis.feet[state.activeFootIndex]?.confidence.metatarsal ?? 0} label="Metatarsos" />
              <Confidence value={analysis.feet[state.activeFootIndex]?.confidence.supportArea ?? 0} label="Zonas de apoio" />
            </section>

            {(analysis.feet[state.activeFootIndex]?.warnings.length ?? 0) > 0 && (
              <section className="panel-section">
                <h3>Observações</h3>
                <ul className="warning-list">
                  {analysis.feet[state.activeFootIndex].warnings.map((w) => (
                    <li key={w}>{w}</li>
                  ))}
                </ul>
              </section>
            )}

            {analysis.shoeSizeCheck?.message && (
              <section className="panel-section">
                <h3>Conferência do número informado</h3>
                <p className="muted small">{analysis.shoeSizeCheck.message}</p>
              </section>
            )}
          </div>
        </aside>
      </div>

      {error && (
        <div className="banner error" role="alert">
          {error}
        </div>
      )}

      <div className="review-footer">
        <p className="muted small">
          Escala verificada pelo marcador: erro de ida-e-volta{' '}
          {analysis.marker.roundTripErrorMm >= 0
            ? `${analysis.marker.roundTripErrorMm.toFixed(3).replace('.', ',')} mm`
            : 'indisponível'}
          .
        </p>
        <button className="primary big" type="button" onClick={() => void approve()} disabled={approving}>
          {approving ? 'Aprovando…' : 'Aprovar pedigrafia'}
        </button>
      </div>
    </div>
  )
}

function Measure({ label, value, strong }: { label: string; value: string; strong?: boolean }) {
  return (
    <div className={strong ? 'measure strong' : 'measure'}>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  )
}

function Confidence({ value, label }: { value: number; label: string }) {
  const pct = Math.round(Math.min(1, Math.max(0, value)) * 100)
  const level = pct >= 75 ? 'high' : pct >= 50 ? 'mid' : 'low'
  return (
    <div className={`confidence ${level}`}>
      <span className="confidence-label">{label}</span>
      <span className="confidence-bar" aria-hidden="true">
        <i style={{ width: `${pct}%` }} />
      </span>
      <span className="confidence-value">{pct}%</span>
    </div>
  )
}

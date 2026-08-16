import { useCallback, useEffect, useState } from 'react'

import * as api from './api/client'
import { CaptureScreen } from './screens/CaptureScreen'
import { DebugScreen } from './screens/DebugScreen'
import { QualityRejectedScreen } from './screens/QualityRejectedScreen'
import { ResultScreen } from './screens/ResultScreen'
import { ReviewScreen } from './screens/ReviewScreen'
import { StartScreen } from './screens/StartScreen'
import type { AnalyzeResponse, CaptureRejected, ReviewedFoot } from './types/pedigrafia'

export type Stage = 'start' | 'capture' | 'analyzing' | 'rejected' | 'review' | 'result' | 'debug'

export interface ApprovedState {
  reviewToken: string
  feet: ReviewedFoot[]
}

export default function App() {
  const [stage, setStage] = useState<Stage>('start')
  const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null)
  const [rejection, setRejection] = useState<CaptureRejected | null>(null)
  const [approved, setApproved] = useState<ApprovedState | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const reset = useCallback(async () => {
    if (analysis) await api.deleteSession(analysis.sessionId)
    setAnalysis(null)
    setRejection(null)
    setApproved(null)
    setError(null)
    setStage('start')
  }, [analysis])

  // A sessão é temporária: ao fechar a aba, pedimos o descarte do material.
  useEffect(() => {
    if (!analysis) return
    const handler = () => {
      navigator.sendBeacon?.(`/api/session/${analysis.sessionId}`)
    }
    window.addEventListener('pagehide', handler)
    return () => window.removeEventListener('pagehide', handler)
  }, [analysis])

  const runAnalysis = useCallback(
    async (
      file: File | Blob,
      opts: {
        shoeSize?: string
        view?: 'below' | 'above'
        calibration?: api.CalibrationSource
      },
    ) => {
      setBusy(true)
      setError(null)
      setStage('analyzing')
      try {
        const res = await api.analyze(file, {
          view: opts.view ?? 'below',
          shoeSize: opts.shoeSize,
          calibration: opts.calibration ?? 'auto',
        })
        setAnalysis(res)
        setRejection(null)
        setApproved(null)
        setStage('review')
      } catch (err) {
        if (err instanceof api.CaptureRejectedError) {
          setRejection(err.rejection)
          setStage('rejected')
        } else {
          setError(err instanceof Error ? err.message : 'Falha ao analisar a imagem.')
          setStage('capture')
        }
      } finally {
        setBusy(false)
      }
    },
    [],
  )

  return (
    <div className="app-shell">
      <header className="app-header">
        <button className="brand" type="button" onClick={() => void reset()}>
          <span className="brand-mark" aria-hidden="true" />
          <span className="brand-text">
            Pedigrafia<strong>Digital</strong>
          </span>
        </button>
        <div className="header-actions">
          {analysis && stage !== 'debug' && (
            <button className="ghost small" type="button" onClick={() => setStage('debug')}>
              Depuração
            </button>
          )}
          {stage !== 'start' && (
            <button className="ghost small" type="button" onClick={() => void reset()}>
              Nova
            </button>
          )}
        </div>
      </header>

      {error && (
        <div className="banner error" role="alert">
          {error}
          <button className="ghost small" type="button" onClick={() => setError(null)}>
            Fechar
          </button>
        </div>
      )}

      <main className="app-main">
        {stage === 'start' && <StartScreen onStart={() => setStage('capture')} />}

        {stage === 'capture' && <CaptureScreen busy={busy} onSubmit={runAnalysis} />}

        {stage === 'analyzing' && (
          <div className="centered-state">
            <div className="spinner" aria-hidden="true" />
            <h2>Analisando a captura</h2>
            <p className="muted">
              Localizando a referência de escala, corrigindo a perspectiva e medindo em
              milímetros.
            </p>
          </div>
        )}

        {stage === 'rejected' && rejection && (
          <QualityRejectedScreen rejection={rejection} onRetry={() => setStage('capture')} />
        )}

        {stage === 'review' && analysis && (
          <ReviewScreen
            analysis={analysis}
            onApproved={(state) => {
              setApproved(state)
              setStage('result')
            }}
          />
        )}

        {stage === 'result' && analysis && approved && (
          <ResultScreen
            analysis={analysis}
            approved={approved}
            onBackToReview={() => setStage('review')}
            onNew={() => void reset()}
          />
        )}

        {stage === 'debug' && analysis && (
          <DebugScreen analysis={analysis} onClose={() => setStage('review')} />
        )}
      </main>
    </div>
  )
}

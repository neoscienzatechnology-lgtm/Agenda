import type { CaptureQuality } from '../types/pedigrafia'

export function QualityReport({ quality, compact = false }: { quality: CaptureQuality; compact?: boolean }) {
  const shown = compact ? quality.checks.filter((c) => !c.passed || c.score < 0.85) : quality.checks
  if (compact && shown.length === 0) {
    return <p className="muted small">Todas as verificações de captura passaram com folga.</p>
  }
  return (
    <ul className="check-list">
      {shown.map((c) => (
        <li key={c.id} className={c.passed ? 'ok' : c.severity === 'blocker' ? 'fail' : 'warn'}>
          <span className="mark" aria-hidden="true">
            {c.passed ? '✓' : '✕'}
          </span>
          <span className="check-label">{c.label}</span>
          <span className="check-bar" aria-hidden="true">
            <i style={{ width: `${Math.round(Math.min(1, Math.max(0, c.score)) * 100)}%` }} />
          </span>
          {!c.passed && c.hint && <span className="check-hint">{c.hint}</span>}
        </li>
      ))}
    </ul>
  )
}

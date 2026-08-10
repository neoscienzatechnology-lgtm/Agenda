import { QualityReport } from '../components/QualityReport'
import type { CaptureRejected } from '../types/pedigrafia'

export function QualityRejectedScreen({
  rejection,
  onRetry,
}: {
  rejection: CaptureRejected
  onRetry: () => void
}) {
  const q = rejection.captureQuality
  return (
    <div className="rejected-screen">
      <div className="score-badge fail">
        <span className="score-value">{q.score.toFixed(0)}</span>
        <span className="score-total">/100</span>
      </div>
      <h2>Captura inadequada</h2>
      <p className="lead">
        A análise não foi realizada. Medir a partir de uma foto inadequada produziria um
        molde errado — e o molde é uma peça física de fabricação.
      </p>

      <ul className="blocker-list">
        {q.blockers.map((b) => (
          <li key={b}>
            <span className="mark fail" aria-hidden="true">
              ✕
            </span>
            {b}
          </li>
        ))}
      </ul>

      <QualityReport quality={q} />

      <button className="primary big" type="button" onClick={onRetry}>
        Refazer foto
      </button>
    </div>
  )
}

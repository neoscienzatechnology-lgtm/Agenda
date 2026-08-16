import { markerPdfUrl, targetPdfUrl } from '../api/client'

export function StartScreen({ onStart }: { onStart: () => void }) {
  return (
    <div className="start-screen">
      <div className="start-hero">
        <p className="eyebrow">Pedigrafia calibrada</p>
        <h1>
          Molde plantar em <span className="accent">escala 1:1 real</span>
        </h1>
        <p className="lead">
          Posicione o(s) pé(s) sobre a plataforma e mantenha o marcador de referência de
          50 × 50 mm totalmente visível. A dimensão física vem exclusivamente da
          calibração geométrica do marcador — nunca do número do calçado.
        </p>

        <button className="primary big" type="button" onClick={onStart}>
          Nova pedigrafia
        </button>

        <a className="secondary link" href={targetPdfUrl()} target="_blank" rel="noreferrer">
          Baixar alvo de calibração — 4 marcadores (recomendado)
        </a>
        <a className="ghost link" href={markerPdfUrl()} target="_blank" rel="noreferrer">
          Marcador único (menos exato longe do marcador)
        </a>
        <p className="note small">
          Com quatro marcadores ao redor da área de apoio a escala é interpolada entre
          eles. Com um marcador só, ela é extrapolada, e o erro cresce com a distância
          até o marcador — medido em até 2 mm.
        </p>
      </div>

      <ol className="start-steps">
        <li>
          <span className="step-index">1</span>
          <div>
            <h3>Posicione</h3>
            <p>Pés sobre o podoscópio, marcador rígido no mesmo plano da planta.</p>
          </div>
        </li>
        <li>
          <span className="step-index">2</span>
          <div>
            <h3>Fotografe</h3>
            <p>Celular na mão ou em suporte. A distância não precisa ser fixa.</p>
          </div>
        </li>
        <li>
          <span className="step-index">3</span>
          <div>
            <h3>Revise</h3>
            <p>Ajuste contorno, eixo, metatarsos, arcos e zonas de apoio.</p>
          </div>
        </li>
        <li>
          <span className="step-index">4</span>
          <div>
            <h3>Aprove e imprima</h3>
            <p>Um pé por folha A4, em tamanho real. Sem “ajustar à página”.</p>
          </div>
        </li>
      </ol>
    </div>
  )
}

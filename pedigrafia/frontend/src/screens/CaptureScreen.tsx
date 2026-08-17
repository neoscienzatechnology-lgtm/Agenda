import { useRef, useState } from 'react'

import type { CalibrationSource } from '../api/client'

export interface CaptureScreenProps {
  busy: boolean
  onSubmit: (
    file: File | Blob,
    opts: { shoeSize?: string; view?: 'below' | 'above'; calibration?: CalibrationSource },
  ) => void
}

/** Cada opção declara de onde vem o milímetro. Os avisos são medidos, não retóricos:
 *  a tolerância citada é a da própria norma do objeto e vira erro de comprimento
 *  proporcional, que nenhum processamento de imagem remove. */
const SOURCES: {
  key: CalibrationSource
  label: string
  detail: string
  hint: string
}[] = [
  {
    key: 'aruco',
    label: 'Alvo impresso',
    detail: 'Marcadores de 50 mm',
    hint: 'Mais exato. Com os quatro marcadores ao redor da área de apoio, a escala é interpolada.',
  },
  {
    key: 'card',
    label: 'Cartão',
    detail: '85,60 × 53,98 mm',
    hint:
      'Sem imprimir nada. Dimensão bem controlada (ISO/IEC 7810 ID-1, ±0,13 mm ≈ ±0,4 mm em um pé de 265 mm), mas objeto pequeno: espalhe quatro ao redor da área de apoio para a cobertura ficar equivalente à do alvo impresso.',
  },
  {
    key: 'a4',
    label: 'Folha A4',
    detail: '210 × 297 mm',
    hint:
      'Cobre bem a área, mas o corte do papel tem tolerância de ±2 mm (ISO 216) — cerca de ±2,5 mm em um pé de 265 mm, que nenhum algoritmo remove. E confirme que a folha é mesmo A4: toda a série ISO A tem a mesma proporção, e uma A5 aumentaria as medidas em 42 % sem disparar alarme.',
  },
]

export function CaptureScreen({ busy, onSubmit }: CaptureScreenProps) {
  const cameraRef = useRef<HTMLInputElement | null>(null)
  const fileRef = useRef<HTMLInputElement | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [shoeSize, setShoeSize] = useState('')
  const [view, setView] = useState<'below' | 'above'>('below')
  const [calibration, setCalibration] = useState<CalibrationSource>('aruco')
  const active = SOURCES.find((s) => s.key === calibration) ?? SOURCES[0]

  const pick = (f: File | null) => {
    if (!f) return
    setFile(f)
    setPreview((old) => {
      if (old) URL.revokeObjectURL(old)
      return URL.createObjectURL(f)
    })
  }

  return (
    <div className="capture-screen">
      <div className="capture-stage">
        {preview ? (
          <img className="capture-preview" src={preview} alt="Pré-visualização da captura" />
        ) : (
          <div className="capture-guide" aria-hidden="true">
            <div className="guide-frame">
              <span className="guide-corner tl" />
              <span className="guide-corner tr" />
              <span className="guide-corner bl" />
              <span className="guide-corner br" />
              <div className="guide-feet">
                <span />
                <span />
              </div>
              <div className="guide-marker">
                {calibration === 'aruco' ? '50 × 50 mm' : active.detail}
              </div>
            </div>
            <p className="guide-hint">
              Enquadre os dois pés <strong>e</strong> a referência de escala. Fotografe o
              mais perpendicular possível à plataforma.
            </p>
          </div>
        )}
      </div>

      <div className="capture-controls">
        <fieldset className="calibration-picker">
          <legend>De onde vem a medida</legend>
          <div className="calibration-options" role="radiogroup" aria-label="Referência de escala">
            {SOURCES.map((source) => (
              <button
                key={source.key}
                type="button"
                role="radio"
                aria-checked={calibration === source.key}
                className={`calibration-option${calibration === source.key ? ' selected' : ''}`}
                onClick={() => setCalibration(source.key)}
                disabled={busy}
              >
                <strong>{source.label}</strong>
                <span>{source.detail}</span>
              </button>
            ))}
          </div>
          <p className="note small">{active.hint}</p>
          <p className="note small">
            É obrigatório declarar qual objeto foi usado: um cartão e uma folha A4 têm
            razões largura/altura a ~11 % uma da outra, dentro do ruído de uma foto de
            celular, e confundi-los erraria a escala em 2,45× sem nenhum outro sintoma.
          </p>
        </fieldset>

        <div className="capture-buttons">
          <button className="primary" type="button" onClick={() => cameraRef.current?.click()} disabled={busy}>
            Tirar foto
          </button>
          <button className="secondary" type="button" onClick={() => fileRef.current?.click()} disabled={busy}>
            Selecionar foto
          </button>
        </div>

        <input
          ref={cameraRef}
          type="file"
          accept="image/*"
          capture="environment"
          hidden
          onChange={(e) => pick(e.target.files?.[0] ?? null)}
        />
        <input
          ref={fileRef}
          type="file"
          accept="image/jpeg,image/png,image/webp,image/heic,image/heif"
          hidden
          onChange={(e) => pick(e.target.files?.[0] ?? null)}
        />

        <details className="capture-extra">
          <summary>Opções</summary>
          <label className="field">
            <span>Ponto de vista da câmera</span>
            <select value={view} onChange={(e) => setView(e.target.value as 'below' | 'above')}>
              <option value="below">Podoscópio — câmera sob o vidro</option>
              <option value="above">Câmera acima da plataforma</option>
            </select>
          </label>
          <label className="field">
            <span>
              Número do calçado <em>(opcional, apenas conferência)</em>
            </span>
            <input
              type="number"
              inputMode="numeric"
              min={15}
              max={55}
              placeholder="ex.: 40"
              value={shoeSize}
              onChange={(e) => setShoeSize(e.target.value)}
            />
          </label>
          <p className="note">
            O número do calçado <strong>não</strong> dimensiona o molde. Ele serve apenas
            como verificação de sanidade contra a medida calibrada.
          </p>
        </details>

        <button
          className="primary big"
          type="button"
          disabled={!file || busy}
          onClick={() =>
            file && onSubmit(file, { shoeSize: shoeSize || undefined, view, calibration })
          }
        >
          {busy ? 'Analisando…' : 'Analisar captura'}
        </button>
      </div>
    </div>
  )
}

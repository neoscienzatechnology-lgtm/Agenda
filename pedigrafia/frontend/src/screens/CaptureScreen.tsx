import { useRef, useState } from 'react'

export interface CaptureScreenProps {
  busy: boolean
  onSubmit: (file: File | Blob, opts: { shoeSize?: string; view?: 'below' | 'above' }) => void
}

export function CaptureScreen({ busy, onSubmit }: CaptureScreenProps) {
  const cameraRef = useRef<HTMLInputElement | null>(null)
  const fileRef = useRef<HTMLInputElement | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [shoeSize, setShoeSize] = useState('')
  const [view, setView] = useState<'below' | 'above'>('below')

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
              <div className="guide-marker">50 × 50 mm</div>
            </div>
            <p className="guide-hint">
              Enquadre os dois pés <strong>e</strong> o marcador. Fotografe o mais
              perpendicular possível à plataforma.
            </p>
          </div>
        )}
      </div>

      <div className="capture-controls">
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
          onClick={() => file && onSubmit(file, { shoeSize: shoeSize || undefined, view })}
        >
          {busy ? 'Analisando…' : 'Analisar captura'}
        </button>
      </div>
    </div>
  )
}

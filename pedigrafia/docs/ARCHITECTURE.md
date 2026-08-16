# Pedigrafia Digital — Arquitetura

> **Princípio norteador:** este software produz uma **referência física de fabricação**.
> A dimensão física NUNCA é inventada, estimada por número de calçado, nem ajustada
> para "caber" na folha. Ela vem exclusivamente de uma cadeia geométrica determinística
> ancorada em um marcador fiducial de **50,00 × 50,00 mm**.

---

## 1. Visão geral

```
┌──────────────┐   multipart/form-data    ┌─────────────────────────────┐
│  PWA (React) │ ───────────────────────► │  API FastAPI (Python)       │
│  Vite + TS   │                          │                             │
│              │ ◄─────────────────────── │  pipeline de visão (OpenCV) │
│  Editor      │   AnalyzeResponse (mm)   │                             │
│  vetorial    │                          └──────────────┬──────────────┘
│              │   geometria revisada (mm)               │
│              │ ───────────────────────►  /api/export-pdf│
│              │ ◄─────────────────────── PDF A4 1:1 ─────┘
└──────────────┘
```

* **Todo** o processamento de imagem acontece **no servidor**.
* **Toda** a geometria trafega em **milímetros no plano físico da plataforma** — nunca em pixels.
* O frontend nunca reescala geometria; apenas aplica uma transformação de *view*
  (zoom/pan) para desenhar, mantendo o modelo em mm intacto.

## 2. Cadeia metrológica (o coração do sistema)

```
 foto original (px)
   │  ① normalização EXIF (rotação apenas — sem reescala)
   ▼
 imagem normalizada (px)
   │  ② detecção da referência física, por um de dois caminhos:
   │      • ArUco/AprilTag + refino subpixel (cornerSubPix)          — mais exato
   │      • retângulo normalizado (cartão ISO 7810, folha A4):
   │        ajuste de retas nas arestas + interseção                 — sem impressora
   ▼
 4 cantos por objeto (px, subpixel)
   │  ③ correspondência com o modelo físico conhecido:
   │       marcador:  (0,0) (50,0) (50,50) (0,50)          [mm]
   │       cartão:    (0,0) (85.6,0) (85.6,53.98) (0,53.98) [mm]
   │     com 2+ objetos soltos, a pose de cada um entra como incógnita
   │     do ajuste conjunto (fit_free_rectangles)
   ▼
 H_img→mm  (homografia 3×3, exata para os 4 pontos)
   │  ④ retificação: amostragem do plano em RECTIFIED_PX_PER_MM px/mm
   ▼
 imagem retificada (px)  com  mm = origem_mm + px / px_per_mm   (afim, exata)
   │  ⑤ segmentação → máscara → contorno subpixel (marching squares)
   ▼
 contorno em px retificado
   │  ⑥ px → mm (afim)
   ▼
 contourMm : PointMm[]        ◄── única fonte de verdade dimensional
   │  ⑦ revisão humana obrigatória (edição em mm)
   ▼
 geometria aprovada (mm)
   │  ⑧ mm → pt  (pt = mm × 72 / 25.4), transformação RÍGIDA apenas
   ▼
 PDF A4 210×297 mm, escala 1:1
```

**Invariantes verificados por teste automatizado**

| # | Invariante | Onde é testado |
|---|---|---|
| I1 | O marcador re-detectado na imagem retificada mede 50,00 mm ± 0,3 mm de lado | `test_calibration.py::test_rectified_marker_roundtrip` |
| I2 | Comprimento sintético conhecido é recuperado com erro ≤ 1,0 mm | `test_metrology_end_to_end.py` |
| I3 | Suavização do contorno altera o comprimento em ≤ 0,5 mm | `test_contour.py::test_smoothing_preserves_length` |
| I4 | Simplificação (nós editáveis) altera o comprimento em ≤ 0,3 mm | `test_contour.py::test_simplify_preserves_length` |
| I5 | MediaBox do PDF == 595,2756 × 841,8898 pt (A4 exato) | `test_pdf_metrology.py` |
| I6 | Comprimento do path do contorno **relido do PDF** == comprimento em mm ± 0,2 mm, para 200/240/260/265/270 mm | `test_pdf_metrology.py` |
| I7 | O posicionamento na página é **rígido** (todas as distâncias par-a-par preservadas) | `test_pdf_metrology.py::test_placement_is_rigid` |
| I8 | Medidas do TypeScript == medidas do Python (paridade) | `tests/fixtures/measurement_parity.json` + pytest + vitest |
| I9 | Nenhum PDF é emitido sem `reviewApproved` | `test_api.py::test_export_requires_review` |
| I10 | Número do calçado não altera nenhuma dimensão | `test_shoe_size_is_advisory.py` |
| I11 | A referência normalizada re-medida na imagem retificada confere lados **e** diagonais | `test_reference_object.py::test_round_trip_re_measures_the_object_on_the_rectified_image` |
| I12 | O modo automático recusa identificar a referência quando a forma não prova qual é | `test_reference_object.py::test_auto_mode_refuses_instead_of_guessing_the_object` |
| I13 | Pedir alvo impresso nunca cai, em silêncio, numa escala de outra origem | `test_reference_object.py::test_aruco_mode_never_falls_back_to_a_reference` |

## 3. Sistemas de coordenadas

Ver `COORDINATE_SYSTEM.md` para a definição formal. Resumo:

| Frame | Unidade | Origem | Uso |
|---|---|---|---|
| `image` | px | canto sup. esq. da foto normalizada | somente entrada |
| `plane` | **mm** | canto sup. esq. da referência (marcador ou objeto) | **frame canônico de troca** |
| `rect` | px | canto sup. esq. do raster retificado | somente rasterização |
| `foot` | mm | ponto mais posterior do calcâneo, eixo *u* → dedos | relatório de landmarks |
| `page` | mm → pt | canto inf. esq. da folha A4 | somente PDF |

Transformações `plane ↔ rect` são **afins puras** (escala uniforme + translação),
portanto não introduzem distorção. `image → plane` é a única projetiva.

## 4. Módulos do backend

```
app/
  config.py          Settings (thresholds, TTL, limites) — 100% configurável por env
  schemas.py         Contratos Pydantic (espelhados em frontend/src/types)
  security.py        Validação de upload, sniffing MIME, nomes aleatórios
  storage.py         Sessão temporária em disco + reaper TTL
  calibration/       marker.py (ArUco/AprilTag), reference.py (retângulo normalizado),
                     target.py (posições físicas), fit.py (homografia por mínimos
                     quadrados), homography.py (retificação, px/mm)
  quality/           metrics.py (foco, blur, exposição, glare…), gate.py (score 0–100)
  segmentation/      base.py (interface), classical.py, onnx_model.py, factory.py
  geometry/          contour.py, frame.py, separation.py, laterality.py,
                     arches.py, support_zones.py, polygon.py
  landmarks/         toes.py, metatarsals.py, detect.py
  measurements/      compute.py (fonte de verdade), arch_index.py
  callosity/         detect.py (sugestão visual, NÃO diagnóstico)
  render/            annotated.py (PNG anotado)
  pdf/               builder.py (ReportLab 1:1), inspect.py (parser de content stream)
  synth/             generator.py (cenas sintéticas de geometria conhecida)
  api/               routes_*.py
```

**Regra de dependência:** `api → measurements/pdf/render → geometry/landmarks →
calibration/segmentation → numpy/opencv`. Nenhum módulo de geometria importa FastAPI.

## 5. Segmentação plugável

```python
class Segmenter(Protocol):
    name: str
    def segment(self, rect_bgr, ctx: SegmentationContext) -> SegmentationResult: ...
```

* `ClassicalSegmenter` (padrão): fusão de pistas (prior de pele YCrCb, contraste local
  contra modelo de fundo das bordas, Otsu em L*/a*) → trimapa → GrabCut → limpeza
  morfológica com limiares em **mm²**.
* `OnnxSegmenter` (`ml-or-vision/`): carrega qualquer modelo de segmentação exportado
  para ONNX (YOLO-seg, U-Net, SAM decoder…) via `PEDIGRAFIA_SEG_MODEL`.
* `factory.get_segmenter()` resolve por env `PEDIGRAFIA_SEGMENTER`, com *fallback*
  automático e registrado para o clássico. Trocar de modelo = trocar 1 variável de
  ambiente; nenhuma outra camada muda.

## 6. Estratégia de PDF 1:1

* ReportLab, `pagesize=A4` → MediaBox exatamente `595.2755905511812 × 841.8897637795277 pt`.
* `MM_TO_PT = 72 / 25.4` aplicado a cada vértice — sem `scale()`, sem `fit`, sem bbox.
* Posicionamento = **rotação rígida** (alinhar o eixo do pé à vertical da folha) +
  **translação**. Ambas preservam distâncias exatamente; a matriz é registrada nos
  metadados do PDF e a rigidez é verificada por teste.
* Se o pé não couber em A4 a 1:1, o sistema **não reescala**: marca `fitsOnPage=false`,
  emite aviso e mantém 1:1.
* Sem régua/quadrado de calibração na folha final (a escala vem da referência física).
* `pdf/inspect.py` relê o PDF gerado, decodifica o content stream, aplica o CTM e
  devolve os paths em mm — usado nos testes e no endpoint de QA `/api/verify-pdf`.

## 7. Revisão humana obrigatória

O backend só aceita `POST /api/export-pdf` com `review.approved == true` **e** um
`reviewToken` emitido por `POST /api/review/approve`, que por sua vez exige que a
geometria enviada seja válida (polígono simples, ≥ 3 pontos, todos os landmarks
presentes). Não há caminho de exportação que ignore a revisão.

## 8. O que a IA pode e não pode fazer

| Pode | Não pode |
|---|---|
| Segmentar a região do pé | Definir escala física |
| Sugerir landmarks (com *confidence*) | Definir comprimento final |
| Sugerir zonas de apoio e calosidades | Definir dimensões do molde |
| Ordenar/rotular estruturas | Definir tamanho no PDF |

Todo valor dimensional publicado vem de `measurements/compute.py`, que opera **apenas**
sobre coordenadas em mm derivadas do marcador.

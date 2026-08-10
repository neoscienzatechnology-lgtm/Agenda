# Pedigrafia Digital

PWA para produzir um **molde plantar em escala 1:1 real** a partir de uma fotografia
de podoscópio, com a dimensão física derivada exclusivamente de um marcador fiducial
de **50,00 × 50,00 mm**.

> **O número do calçado nunca dimensiona nada.** Ele existe apenas como conferência
> de sanidade. Se a calibração medir 264,3 mm, o PDF sai com 264,3 mm.

```
1 Nova pedigrafia → 2 Tirar foto → 3 Verificação automática → 4 Análise
                 → 5 Revisar marcações → 6 Aprovar → 7 Baixar PDF 1:1
```

---

## Estado atual

| Área | Situação |
|---|---|
| Cadeia metrológica marcador → mm | Verificada: erro de ida-e-volta 0,005–0,072 mm |
| Comprimento fim a fim (sintético, perpendicular) | −0,04 a +0,12 mm |
| Comprimento fim a fim (câmera inclinada 10–20°) | até +2,1 mm — ver limitações |
| PDF A4 1:1 relido do arquivo | erro 0,0000 mm em 200/240/260/265/270 mm |
| Testes | 118 pytest · 10 vitest · 4 end-to-end (desktop + mobile) |
| Validação com hardware/pés reais | **não realizada** — ver `docs/VALIDATION_CHECKLIST.md` |

**Este sistema não tem precisão clínica nem metrológica validada.** Os números acima
vêm de cenas sintéticas de geometria conhecida. Antes de usar para fabricação,
execute o checklist de validação física.

## Estrutura

```
pedigrafia/
├── frontend/         PWA React + TypeScript + Vite (editor vetorial em canvas)
├── backend/          API FastAPI + pipeline de visão (OpenCV/NumPy/scikit-image)
│   └── app/
│       ├── calibration/   marcador ArUco/AprilTag, homografia, escala px→mm
│       ├── quality/       métricas objetivas e quality gate 0–100
│       ├── segmentation/  interface plugável + clássico + ponte ONNX
│       ├── geometry/      polígonos em mm, frame do pé, contorno sub-pixel
│       ├── landmarks/     T1–T5 (observados) e M1–M5 (M2–M4 estimados)
│       ├── measurements/  fonte única de verdade dimensional
│       ├── callosity/     sugestão visual (não diagnóstico)
│       ├── render/        PNG anotado para análise visual
│       ├── pdf/           construtor A4 1:1 + inspetor de content stream
│       ├── synth/         gerador de cenas com verdade geométrica conhecida
│       └── api/           rotas
├── ml-or-vision/     adaptadores de modelos (ONNX) e guia de treino
├── tests/            pytest, fixture de paridade, end-to-end Playwright
└── docs/             arquitetura, metrologia, limitações, validação física
```

## Documentação

| Documento | Conteúdo |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Arquitetura, cadeia metrológica, invariantes testados |
| [`docs/COORDINATE_SYSTEM.md`](docs/COORDINATE_SYSTEM.md) | Os cinco frames e as conversões exatas |
| [`docs/RUNNING.md`](docs/RUNNING.md) | Rodar, testar, configurar e implantar |
| [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) | Exatidão medida e limitações conhecidas |
| [`docs/VALIDATION_CHECKLIST.md`](docs/VALIDATION_CHECKLIST.md) | Validação com régua e impressão reais |
| [`docs/TRAINING.md`](docs/TRAINING.md) | Treinar um modelo específico de segmentação |

## Início rápido

```bash
cd pedigrafia
python3 -m venv .venv && .venv/bin/pip install -r backend/requirements-dev.txt
PYTHONPATH=backend .venv/bin/uvicorn app.main:app --port 8000 &
cd frontend && npm install && npm run dev      # http://localhost:5173
```

Imprima o marcador em `http://localhost:8000/api/marker.pdf` (**100 %, sem ajustar à
página**), confira com régua que mede 50,00 mm, e cole em superfície rígida.

## Princípios que o código impõe

1. **A escala vem do marcador.** Nenhum outro caminho define dimensão física.
2. **Geometria em milímetros, sempre.** Pixels existem só para rasterizar; a única
   conversão px↔mm vive em `calibration/homography.py` e em `geom/units.ts`.
3. **Posicionamento no PDF é isometria.** `|det(M)| = 1`, verificado em teste; as
   distâncias par-a-par são preservadas em 1,1 × 10⁻¹³ mm.
4. **Revisão humana é obrigatória.** `/api/export-pdf` exige um `reviewToken` emitido
   por `/api/review/approve`. Não existe caminho alternativo.
5. **Estimativa nunca se disfarça de observação.** Todo landmark carrega `method` e
   `confidence`; M2–M4 saem como `interpolated_model` e a UI os destaca.
6. **Nomes honestos.** "Índice geométrico do arco em projeção 2D" (não altura de
   arco); "zonas visuais de contato/apoio" (não pressão); "sugestão visual" (não
   diagnóstico).
7. **IA não define dimensão.** O modelo decide onde está o pé; a geometria calibrada
   decide quanto ele mede.

## API

| Endpoint | Função |
|---|---|
| `POST /api/analyze` | Foto → qualidade, marcador, retificação, pés em mm |
| `POST /api/measure` | Recomputa medidas a partir da geometria revisada |
| `POST /api/review/approve` | Emite o `reviewToken` (revisão obrigatória) |
| `POST /api/export-pdf` | PDF A4 1:1, um pé por folha |
| `POST /api/render-annotated` | PNG anotado (análise visual) |
| `GET /api/marker.pdf` | Folha de calibração 50 × 50 mm |
| `POST /api/verify-pdf` | QA: relê um PDF e confere a dimensão física |
| `DELETE /api/session/{id}` | Apaga a sessão temporária |

Documentação interativa em `/docs`.

## Fora do escopo deste MVP

Login, cadastro de pacientes, prontuário, histórico, agenda, pagamentos,
multiempresa, faturamento, estoque e CRM — deliberadamente não implementados.
A sessão é temporária e apagada por TTL.

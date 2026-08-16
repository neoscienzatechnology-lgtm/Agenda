# Rodar localmente, testar e implantar

## Requisitos

* Python 3.11+
* Node 20+
* (opcional) `onnxruntime`, apenas se for usar um modelo neural de segmentação

## 1. Backend

```bash
cd pedigrafia
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements-dev.txt

# API em http://127.0.0.1:8000  (docs interativas em /docs)
PYTHONPATH=backend .venv/bin/uvicorn app.main:app --reload --port 8000
```

## 2. Frontend (PWA)

```bash
cd pedigrafia/frontend
npm install
npm run dev          # http://localhost:5173 — já faz proxy de /api para :8000
```

Build de produção e pré-visualização (é o que o teste end-to-end usa):

```bash
npm run build
npm run preview      # http://127.0.0.1:4173, também com proxy de /api
```

Para instalar como PWA: abra em HTTPS (ou `localhost`) e use "Instalar aplicativo".
O *service worker* faz cache do shell da aplicação; **respostas da API nunca são
cacheadas** — uma medida servida de cache seria uma medida potencialmente errada.

## 3. Testes

```bash
cd pedigrafia

# Backend — 132 testes (calibração, geometria, PDF, API, segurança, quality gate)
.venv/bin/python -m pytest tests/backend -q

# Paridade das medidas em TypeScript — 10 testes
cd frontend && npm test && cd ..

# End-to-end no navegador (desktop + mobile), com verificação do PDF baixado
.venv/bin/python tests/e2e/make_scene.py
# … com backend em :8000 e `npm run preview` em :4173 …
npm install
PW_CHROMIUM=/caminho/para/chrome npx playwright test
```

Regenerar o *fixture* de paridade após alterar as medidas (obrigatório: os dois lados
falham se divergirem):

```bash
.venv/bin/python tests/make_parity_fixture.py
```

## 4. Configuração

Tudo via variáveis de ambiente com prefixo `PEDIGRAFIA_`. As principais:

| Variável | Padrão | Efeito |
|---|---|---|
| `PEDIGRAFIA_MARKER_SIZE_MM` | `50.0` | Aresta física do marcador |
| `PEDIGRAFIA_MARKER_DICTIONARY` | `DICT_4X4_50` | ArUco/AprilTag (`DICT_APRILTAG_36H11`, …) |
| `PEDIGRAFIA_MARKER_ID` | `-1` | `-1` aceita qualquer id do dicionário |
| `PEDIGRAFIA_TARGET` | `auto` | `auto`, `single`, `board4` ou caminho de um JSON de alvo |
| `PEDIGRAFIA_REFERENCE_OBJECT` | `auto` | Referência sem impressão: `auto`, `off`, `card`, `a4`, `a5` ou `"LxA"` |
| `PEDIGRAFIA_REFERENCE_CUSTOM_MM` | — | Retângulo próprio: `"85.6x53.98@0.1"` |
| `PEDIGRAFIA_PRINTED_TARGET_TOLERANCE_MM` | `0.20` | Tolerância assumida do alvo impresso |
| `PEDIGRAFIA_MAX_EXTRAPOLATION_WARN_MM` | `60` | Acima disso o score cai |
| `PEDIGRAFIA_MAX_EXTRAPOLATION_BLOCK_MM` | `450` | Acima disso a captura é recusada |
| `PEDIGRAFIA_RECTIFIED_PX_PER_MM` | `6.0` | Amostragem do raster retificado |
| `PEDIGRAFIA_WORKING_AREA_MM` | `700.0` | Janela física máxima retificada |
| `PEDIGRAFIA_SEGMENTER` | `classical` | `classical` ou `onnx` |
| `PEDIGRAFIA_SEG_MODEL` | — | Caminho do `.onnx` |
| `PEDIGRAFIA_SESSION_TTL_SECONDS` | `1800` | Vida da sessão temporária |
| `PEDIGRAFIA_MAX_UPLOAD_BYTES` | `26214400` | Limite de upload |
| `PEDIGRAFIA_CORS_ORIGINS` | `http://localhost:5173,…` | Origens permitidas |
| `PEDIGRAFIA_RATE_LIMIT_REQUESTS` | `40` | Por janela, por IP |
| `PEDIGRAFIA_DEBUG_ARTIFACTS` | `false` | Grava estágios intermediários |
| `PEDIGRAFIA_PDF_MARGIN_MM` | `6.0` | Margem mínima na folha |

## 5. Deploy

### Backend (contêiner)

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ /srv/
ENV PYTHONPATH=/srv PEDIGRAFIA_STORAGE_DIR=/tmp/pedigrafia
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

**`--workers 1` é obrigatório nesta versão.** As sessões vivem em memória de processo;
com mais de um worker uma requisição de exportação pode cair em um processo que não
conhece o `reviewToken`. Para escalar horizontalmente, substitua `app/storage.py` por
um store compartilhado (Redis/S3) — a interface já está isolada.

Dimensionamento: uma análise leva ~2–3 s de CPU (imagem de 7 MP). Prefira mais
instâncias de 1 worker a um processo com muitos workers.

### Frontend (estático)

`npm run build` gera `dist/`, servível por qualquer CDN ou servidor estático.
Configure o servidor para encaminhar `/api` ao backend, ou publique o backend em outro
domínio e defina `VITE_API_BASE` no build (e inclua o domínio em
`PEDIGRAFIA_CORS_ORIGINS`).

**HTTPS é obrigatório** em produção: sem ele a câmera não é liberada pelo navegador e
a PWA não instala.

### Higiene de dados

* As sessões expiram em 30 min e um *reaper* apaga o diretório em disco.
* `DELETE /api/session/{id}` apaga imediatamente; a PWA dispara isso ao fechar a aba.
* Aponte `PEDIGRAFIA_STORAGE_DIR` para um volume efêmero (`tmpfs` de preferência).
* Nenhuma imagem é registrada em log. As exceções do pipeline registram apenas o
  *traceback*.

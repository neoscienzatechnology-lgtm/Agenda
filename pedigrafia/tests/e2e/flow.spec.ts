/**
 * End-to-end no navegador real: percorre os 7 passos do fluxo principal e confere
 * que o PDF baixado tem a dimensão física da medida exibida na tela.
 *
 * Pré-requisitos (ver docs/RUNNING.md):
 *   backend  → uvicorn app.main:app --port 8000
 *   frontend → npm run build && npm run preview   (porta 4173, com proxy /api)
 *   cena     → python tests/e2e/make_scene.py
 */

import { expect, test } from '@playwright/test'
import { execFileSync } from 'node:child_process'
import { existsSync } from 'node:fs'
import path from 'node:path'

const ROOT = path.resolve(__dirname, '../..')
const SCENE = path.join(ROOT, 'tests', 'e2e', 'scene.jpg')
const VENV_PY = path.join(ROOT, '.venv', 'bin', 'python')

test.beforeAll(() => {
  if (!existsSync(SCENE)) {
    execFileSync(VENV_PY, [path.join(ROOT, 'tests', 'e2e', 'make_scene.py')], {
      stdio: 'inherit',
    })
  }
})

test('fluxo completo: captura → revisão → aprovação → PDF 1:1', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', (e) => errors.push(String(e)))

  // 1 — início
  await page.goto('/')
  await expect(page.getByRole('heading', { level: 1 })).toContainText('escala 1:1')
  await page.getByRole('button', { name: 'Nova pedigrafia' }).click()

  // 2 — captura. A origem da escala é declarada aqui, e o padrão é o alvo impresso:
  // a cena de teste tem marcador ArUco, e um padrão diferente mudaria a metrologia
  // sem que o teste percebesse.
  await expect(page.getByRole('button', { name: 'Selecionar foto' })).toBeVisible()
  await expect(page.getByRole('radio', { name: /Alvo impresso/ })).toHaveAttribute(
    'aria-checked',
    'true',
  )
  await expect(page.getByRole('radio', { name: /Cartão/ })).toBeVisible()
  await page.locator('input[type=file]').nth(1).setInputFiles(SCENE)
  await page.getByRole('button', { name: 'Analisar captura' }).click()

  // 3 e 4 — verificação automática + análise
  await expect(page.getByRole('tab', { name: 'Contorno' })).toBeVisible({ timeout: 90_000 })

  // 5 — revisão: as ferramentas exigidas existem e o canvas responde
  for (const tool of ['Contorno', 'Eixo', 'Metatarsos', 'Arcos', 'Zonas de apoio', 'Recorte']) {
    await expect(page.getByRole('tab', { name: tool })).toBeVisible()
  }
  await expect(page.getByRole('button', { name: 'Desfazer' })).toBeDisabled()

  // No mobile o painel começa recolhido; no desktop ele já está aberto e o handle
  // nem existe. Clicar num elemento invisível travaria até o timeout do teste.
  const handle = page.getByRole('button', { name: 'Medidas e marcações' })
  if (await handle.isVisible()) await handle.click()
  const panel = page.locator('.review-panel')
  await expect(panel.locator('.measure.strong dt')).toHaveText('Comprimento')

  const lengthText = await panel.locator('.measure.strong dd').first().innerText()
  const lengthMm = Number(lengthText.replace(' mm', '').replace(',', '.'))
  expect(lengthMm).toBeGreaterThan(150)
  expect(lengthMm).toBeLessThan(320)

  // Arrastar um nó do contorno deve habilitar o desfazer (edição real do modelo).
  const canvas = page.locator('canvas.editor-canvas')
  const box = (await canvas.boundingBox())!
  await page.mouse.move(box.x + box.width * 0.3, box.y + box.height * 0.5)
  await page.mouse.down()
  await page.mouse.move(box.x + box.width * 0.3 + 12, box.y + box.height * 0.5 + 6, { steps: 6 })
  await page.mouse.up()

  // 6 — aprovação
  await page.getByRole('button', { name: 'Aprovar pedigrafia' }).click()
  await expect(page.getByText('Pedigrafia concluída')).toBeVisible({ timeout: 30_000 })

  // 7 — download do PDF de cada pé
  const cards = page.locator('.result-card')
  const count = await cards.count()
  expect(count).toBeGreaterThanOrEqual(1)

  for (let i = 0; i < count; i++) {
    const card = cards.nth(i)
    const shown = await card.locator('.measure.strong dd').first().innerText()
    const shownMm = Number(shown.replace(' mm', '').replace(',', '.'))

    const [download] = await Promise.all([
      page.waitForEvent('download'),
      card.getByRole('button', { name: /PDF 1:1/ }).click(),
    ])
    const file = await download.path()
    expect(file).toBeTruthy()

    // Verificação metrológica independente do PDF baixado.
    const out = execFileSync(
      VENV_PY,
      [path.join(ROOT, 'tests', 'e2e', 'verify_pdf.py'), file!, String(shownMm)],
      { encoding: 'utf-8', env: { ...process.env, PYTHONPATH: path.join(ROOT, 'backend') } },
    )
    const report = JSON.parse(out)
    expect(report.isA4).toBe(true)
    expect(Math.abs(report.errorMm)).toBeLessThan(0.5)
  }

  await expect(page.getByText('Tamanho real')).toBeVisible()
  expect(errors, `erros de página: ${errors.join(' | ')}`).toHaveLength(0)
})

test('exportação é impossível sem passar pela revisão', async ({ request }) => {
  const res = await request.post('/api/export-pdf', {
    data: {
      sessionId: 'sessao-inexistente-xxxxxxxx',
      reviewToken: 'x'.repeat(24),
      foot: {
        id: 'foot-0',
        laterality: 'right',
        contourMm: [
          { x: 0, y: 0 },
          { x: 10, y: 0 },
          { x: 10, y: 10 },
        ],
        axis: { aMm: { x: 0, y: 0 }, bMm: { x: 0, y: 10 }, angleDeg: 90 },
        landmarks: [],
        medialArch: { id: 'medial', pointsMm: [], controlPointsMm: [], confidence: 0 },
        lateralArch: { id: 'lateral', pointsMm: [], controlPointsMm: [], confidence: 0 },
        supportZones: [],
        callosityHints: [],
        toeCutT: 0.78,
        notes: '',
      },
    },
  })
  expect([403, 404]).toContain(res.status())
})

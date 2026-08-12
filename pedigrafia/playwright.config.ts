import { defineConfig, devices } from '@playwright/test'

/**
 * Os servidores precisam estar no ar (ver docs/RUNNING.md):
 *   backend  → uvicorn app.main:app --port 8000
 *   frontend → npm run build && npm run preview   (4173, com proxy /api)
 */
export default defineConfig({
  testDir: './tests/e2e',
  timeout: 180_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: process.env.E2E_BASE_URL ?? 'http://127.0.0.1:4173',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    acceptDownloads: true,
    // O ambiente já traz o Chromium; PW_CHROMIUM aponta para ele quando a versão
    // empacotada pelo Playwright não é a mesma que está instalada.
    launchOptions: process.env.PW_CHROMIUM
      ? { executablePath: process.env.PW_CHROMIUM }
      : {},
  },
  projects: [
    { name: 'desktop', use: { ...devices['Desktop Chrome'] } },
    { name: 'mobile', use: { ...devices['Pixel 7'] } },
  ],
})

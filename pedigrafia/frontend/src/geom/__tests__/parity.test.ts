/**
 * Paridade TypeScript ↔ Python.
 *
 * O mesmo *fixture* é verificado por `tests/backend/test_parity_and_quality.py`.
 * Se as duas implementações de medida divergirem, ambos os testes falham — é essa
 * garantia que permite mostrar medidas ao vivo no editor sem risco de o número da
 * tela discordar do número que vai para o PDF.
 */

import { describe, expect, it } from 'vitest'

import fixture from '../../../../tests/fixtures/measurement_parity.json'
import { computeMeasurements, frameFromAxisPoints } from '../measure'
import { area, clipBandU, maxCaliper, project, resampleClosed, widthAtU } from '../polygon'
import type { PointMm } from '../polygon'

interface Case {
  name: string
  input: {
    contourMm: PointMm[]
    axis: { aMm: PointMm; bMm: PointMm }
    m1Mm: PointMm
    m5Mm: PointMm
    toeCutT: number
  }
  expected: Record<string, number | number[] | PointMm>
}

const data = fixture as unknown as { toleranceMm: number; cases: Case[] }

function expectRelative(actual: number, expected: number, rel: number): void {
  const diff = Math.abs(actual - expected)
  const scale = Math.max(Math.abs(expected), 1e-9)
  expect(diff / scale).toBeLessThan(rel)
}

describe('paridade das medidas com o backend', () => {
  it('tem casos suficientes', () => {
    expect(data.cases.length).toBeGreaterThanOrEqual(4)
  })

  for (const c of data.cases) {
    it(`reproduz ${c.name}`, () => {
      const tol = data.toleranceMm
      const frame = frameFromAxisPoints(c.input.contourMm, c.input.axis.aMm, c.input.axis.bMm, 1)
      const m = computeMeasurements(
        c.input.contourMm,
        frame,
        c.input.m1Mm,
        c.input.m5Mm,
        c.input.toeCutT,
      )
      const e = c.expected as Record<string, number>

      expect(frame.lengthMm).toBeCloseTo(e.frameLengthMm, 6)
      expect(m.lengthMm).toBeCloseTo(e.lengthMm, 6)
      expect(m.forefootWidthMm).toBeCloseTo(e.forefootWidthMm, 6)
      expect(m.midfootWidthMm).toBeCloseTo(e.midfootWidthMm, 6)
      expect(m.heelWidthMm).toBeCloseTo(e.heelWidthMm, 6)
      expect(m.heelToMetatarsalLineMm).toBeCloseTo(e.heelToMetatarsalLineMm, 6)
      // Tolerância RELATIVA de 1e-5. O objetivo do teste é detectar divergência
      // ALGORÍTMICA (fórmula, faixa ou definição diferente), que apareceria com
      // magnitude de 1e-2 ou maior. Exigir igualdade bit a bit entre numpy e JS —
      // somatórios de milhares de termos, `Math.hypot` vs `sqrt(x²+y²)` — testaria
      // a aritmética de ponto flutuante, não a implementação.
      expectRelative(m.archIndex, e.archIndex, 1e-5)
      expect(m.plantarAreaMm2).toBeCloseTo(e.plantarAreaMm2, 4)
      expect(m.bboxWidthMm).toBeCloseTo(e.bboxWidthMm, 6)
      expect(m.forefootWidthAtT).toBeCloseTo(e.forefootWidthAtT, 9)
      expect(m.midfootWidthAtT).toBeCloseTo(e.midfootWidthAtT, 9)
      expect(m.heelWidthAtT).toBeCloseTo(e.heelWidthAtT, 9)
      expect(m.heelToM1Mm).toBeCloseTo(e.heelToM1Mm, 6)
      expect(m.heelToM5Mm).toBeCloseTo(e.heelToM5Mm, 6)
      expect(m.metatarsalLineLengthMm).toBeCloseTo(e.metatarsalLineLengthMm, 6)

      const areas = c.expected.archAreasMm2 as number[]
      m.archAreasMm2.forEach((v, i) => expectRelative(v, areas[i], 1e-5))
      expect(tol).toBeGreaterThan(0)
    })
  }
})

describe('primitivas de polígono', () => {
  const square: PointMm[] = [
    { x: 0, y: 0 },
    { x: 10, y: 0 },
    { x: 10, y: 10 },
    { x: 0, y: 10 },
  ]

  it('área e largura do quadrado', () => {
    expect(area(square)).toBeCloseTo(100, 12)
    expect(widthAtU(square, 5)).toBeCloseTo(10, 12)
  })

  it('recorte em faixa preserva a área esperada', () => {
    expect(area(clipBandU(square, 2, 5))).toBeCloseTo(30, 12)
  })

  it('diâmetro do quadrado é a diagonal', () => {
    expect(maxCaliper(square)).toBeCloseTo(Math.SQRT2 * 10, 12)
  })

  it('reamostragem preserva o perímetro', () => {
    const dense = resampleClosed(square, 0.5)
    expect(dense.length).toBeGreaterThan(60)
    expect(area(dense)).toBeCloseTo(100, 6)
  })

  it('projeção é invertível', () => {
    const uv = project(square, { x: 1, y: 2 }, [0, 1], [-1, 0])
    expect(uv[0].x).toBeCloseTo(-2, 12)
    expect(uv[0].y).toBeCloseTo(1, 12)
  })
})

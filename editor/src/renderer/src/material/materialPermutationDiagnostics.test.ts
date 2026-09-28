import { describe, expect, it } from 'vitest'
import { diagnoseMaterialPermutations } from './materialPermutationDiagnostics'

describe('diagnoseMaterialPermutations', () => {
  it('reports one permutation when no compile-time switches contribute', () => {
    expect(
      diagnoseMaterialPermutations([
        { id: 'fixed', name: 'Fixed', fixed: true },
        { id: 'invalid', name: 'Invalid', variantCount: 0 },
      ]),
    ).toMatchObject({ permutationCount: 1, severity: 'normal', causes: [] })
  })

  it('multiplies independent switch variants and explains their cost', () => {
    const result = diagnoseMaterialPermutations([
      { id: 'normal-map', name: 'Normal Map' },
      { id: 'quality', name: 'Quality', variantCount: 3 },
      { id: 'clear-coat', name: 'Clear Coat' },
    ])

    expect(result.permutationCount).toBe(12)
    expect(result.causes.map(({ id, variantCount }) => [id, variantCount])).toEqual([
      ['clear-coat', 2],
      ['normal-map', 2],
      ['quality', 3],
    ])
    expect(result.message).toContain('12 shader permutations')
    expect(result.message).toContain('Quality (3x)')
  })

  it('deduplicates stable switch IDs deterministically', () => {
    const result = diagnoseMaterialPermutations([
      { id: 'feature', name: 'Feature', variantCount: 2 },
      { id: 'feature', name: 'Stale duplicate', variantCount: 8 },
    ])

    expect(result.permutationCount).toBe(2)
    expect(result.causes).toEqual([{ id: 'feature', name: 'Feature', variantCount: 2 }])
  })

  it('warns when authored combinations cross configured budgets', () => {
    const switches = Array.from({ length: 5 }, (_, index) => ({
      id: `switch-${index}`,
      name: `Switch ${index}`,
    }))

    expect(diagnoseMaterialPermutations(switches).severity).toBe('warning')
    expect(
      diagnoseMaterialPermutations(switches, { warning: 8, critical: 32 }).severity,
    ).toBe('critical')
  })

  it('saturates pathological permutation counts instead of overflowing', () => {
    const result = diagnoseMaterialPermutations([
      { id: 'a', name: 'A', variantCount: Number.MAX_SAFE_INTEGER },
      { id: 'b', name: 'B', variantCount: 2 },
    ])

    expect(result.permutationCount).toBe(Number.MAX_SAFE_INTEGER)
    expect(result.severity).toBe('critical')
  })
})

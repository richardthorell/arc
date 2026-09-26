import { describe, expect, it } from 'vitest'

import {
  DEFAULT_MATERIAL_PARAMETER_GROUP_ID,
  groupMaterialParameters,
} from './materialParameterGroups'

describe('groupMaterialParameters', () => {
  it('places ungrouped parameters in General and sorts groups deterministically', () => {
    const groups = groupMaterialParameters([
      { id: 'roughness', name: 'Roughness', group: 'Surface' },
      { id: 'tint', name: 'Tint' },
      { id: 'emissive', name: 'Emissive', group: 'Advanced' },
    ])

    expect(groups.map((group) => group.id)).toEqual([
      DEFAULT_MATERIAL_PARAMETER_GROUP_ID,
      'advanced',
      'surface',
    ])
    expect(groups[0]?.parameters.map((parameter) => parameter.id)).toEqual(['tint'])
  })

  it('normalizes group labels without changing parameter identity', () => {
    const groups = groupMaterialParameters([
      { id: 'a', name: 'A', group: ' Surface ' },
      { id: 'b', name: 'B', group: 'surface' },
    ])

    expect(groups).toHaveLength(1)
    expect(groups[0]?.parameters.map((parameter) => parameter.id)).toEqual(['a', 'b'])
  })

  it('orders parameters by authored order, then name and stable id', () => {
    const groups = groupMaterialParameters([
      { id: 'z', name: 'Same', group: 'Surface', order: 2 },
      { id: 'b', name: 'Beta', group: 'Surface', order: 1 },
      { id: 'a', name: 'Alpha', group: 'Surface', order: 1 },
      { id: 'c', name: 'Same', group: 'Surface', order: 2 },
    ])

    expect(groups[0]?.parameters.map((parameter) => parameter.id)).toEqual(['a', 'b', 'c', 'z'])
  })

  it('does not mutate the source parameter array', () => {
    const parameters = [
      { id: 'b', name: 'Beta', order: 2 },
      { id: 'a', name: 'Alpha', order: 1 },
    ]

    groupMaterialParameters(parameters)

    expect(parameters.map((parameter) => parameter.id)).toEqual(['b', 'a'])
  })
})

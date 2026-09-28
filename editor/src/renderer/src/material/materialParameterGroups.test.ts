import { describe, expect, it } from 'vitest';

import {
  assignMaterialParameterGroup,
  DEFAULT_MATERIAL_PARAMETER_GROUP_ID,
  groupMaterialParameters,
  reorderMaterialParameter,
} from './materialParameterGroups';

describe('groupMaterialParameters', () => {
  it('places ungrouped parameters in General and sorts groups deterministically', () => {
    const groups = groupMaterialParameters([
      { id: 'roughness', name: 'Roughness', group: 'Surface' },
      { id: 'tint', name: 'Tint' },
      { id: 'emissive', name: 'Emissive', group: 'Advanced' },
    ]);

    expect(groups.map((group) => group.id)).toEqual([DEFAULT_MATERIAL_PARAMETER_GROUP_ID, 'advanced', 'surface']);
    expect(groups[0]?.parameters.map((parameter) => parameter.id)).toEqual(['tint']);
  });

  it('normalizes group labels without changing parameter identity', () => {
    const groups = groupMaterialParameters([
      { id: 'a', name: 'A', group: ' Surface ' },
      { id: 'b', name: 'B', group: 'surface' },
    ]);

    expect(groups).toHaveLength(1);
    expect(groups[0]?.parameters.map((parameter) => parameter.id)).toEqual(['a', 'b']);
  });

  it('orders parameters by authored order, then name and stable id', () => {
    const groups = groupMaterialParameters([
      { id: 'z', name: 'Same', group: 'Surface', order: 2 },
      { id: 'b', name: 'Beta', group: 'Surface', order: 1 },
      { id: 'a', name: 'Alpha', group: 'Surface', order: 1 },
      { id: 'c', name: 'Same', group: 'Surface', order: 2 },
    ]);

    expect(groups[0]?.parameters.map((parameter) => parameter.id)).toEqual(['a', 'b', 'c', 'z']);
  });

  it('does not mutate the source parameter array', () => {
    const parameters = [
      { id: 'b', name: 'Beta', order: 2 },
      { id: 'a', name: 'Alpha', order: 1 },
    ];

    groupMaterialParameters(parameters);

    expect(parameters.map((parameter) => parameter.id)).toEqual(['b', 'a']);
  });
});

describe('material parameter metadata editing', () => {
  it('assigns a trimmed group by stable id and resets stale order', () => {
    const parameters = [
      { id: 'roughness', name: 'Roughness', group: 'Surface', order: 4 },
      { id: 'tint', name: 'Tint', group: 'Surface', order: 1 },
    ];

    const updated = assignMaterialParameterGroup(parameters, 'roughness', '  Advanced  ');

    expect(updated[0]).toEqual({ id: 'roughness', name: 'Roughness', group: 'Advanced', order: undefined });
    expect(updated[1]).toBe(parameters[1]);
    expect(parameters[0]?.group).toBe('Surface');
  });

  it('maps blank group names back to General', () => {
    const updated = assignMaterialParameterGroup([{ id: 'tint', name: 'Tint', group: 'Surface' }], 'tint', '   ');

    expect(groupMaterialParameters(updated)[0]?.id).toBe(DEFAULT_MATERIAL_PARAMETER_GROUP_ID);
  });

  it('reorders only the current group and compacts authored order', () => {
    const parameters = [
      { id: 'a', name: 'Alpha', group: 'Surface', order: 0 },
      { id: 'b', name: 'Beta', group: 'Surface', order: 4 },
      { id: 'c', name: 'Gamma', group: 'Surface', order: 8 },
      { id: 'x', name: 'Other', group: 'Advanced', order: 7 },
    ];

    const updated = reorderMaterialParameter(parameters, 'c', 0);

    expect(groupMaterialParameters(updated)[1]?.parameters.map((parameter) => parameter.id)).toEqual(['c', 'a', 'b']);
    expect(updated.find((parameter) => parameter.id === 'c')?.order).toBe(0);
    expect(updated.find((parameter) => parameter.id === 'a')?.order).toBe(1);
    expect(updated.find((parameter) => parameter.id === 'b')?.order).toBe(2);
    expect(updated.find((parameter) => parameter.id === 'x')).toBe(parameters[3]);
  });

  it('clamps reorder targets and leaves unknown ids unchanged', () => {
    const parameters = [
      { id: 'a', name: 'Alpha' },
      { id: 'b', name: 'Beta' },
    ];

    expect(groupMaterialParameters(reorderMaterialParameter(parameters, 'a', 99))[0]?.parameters.map((p) => p.id)).toEqual([
      'b',
      'a',
    ]);
    expect(reorderMaterialParameter(parameters, 'missing', 0)).toEqual(parameters);
  });
});

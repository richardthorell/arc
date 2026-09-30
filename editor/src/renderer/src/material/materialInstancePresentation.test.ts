import { describe, expect, it } from 'vitest';

import { materialInstancePresentation } from './materialInstancePresentation';

describe('material instance presentation', () => {
  const parent = [
    { id: 'base-color', name: 'Base Color', value: '#ffffff' },
    { id: 'roughness', name: 'Roughness', value: '0.6' },
    { id: 'metallic', name: 'Metallic', value: '0.0' },
  ];

  it('preserves parent order and marks inherited values', () => {
    const result = materialInstancePresentation(parent, []);

    expect(result.parameters.map((parameter) => parameter.id)).toEqual(['base-color', 'roughness', 'metallic']);
    expect(result.parameters).toEqual(
      parent.map((parameter) => ({
        ...parameter,
        inheritedValue: parameter.value,
        isOverridden: false,
        canResetToParent: false,
      })),
    );
  });

  it('shows override values without losing inherited values or parameter identity', () => {
    const result = materialInstancePresentation(parent, [{ parameterId: 'roughness', value: '0.2' }]);
    const roughness = result.parameters[1];

    expect(roughness).toEqual({
      id: 'roughness',
      name: 'Roughness',
      inheritedValue: '0.6',
      value: '0.2',
      isOverridden: true,
      canResetToParent: true,
    });
    expect(result.parameters[0].value).toBe('#ffffff');
  });

  it('reports orphan and duplicate overrides instead of presenting them as valid parameters', () => {
    const result = materialInstancePresentation(parent, [
      { parameterId: 'roughness', value: '0.2' },
      { parameterId: 'missing-z', value: '1' },
      { parameterId: 'roughness', value: '0.1' },
      { parameterId: 'missing-a', value: '2' },
    ]);

    expect(result.orphanOverrideIds).toEqual(['missing-a', 'missing-z']);
    expect(result.duplicateOverrideIds).toEqual(['roughness']);
    expect(result.parameters[1].value).toBe('0.2');
  });
});

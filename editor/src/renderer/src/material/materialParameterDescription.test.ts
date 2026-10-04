import { describe, expect, it } from 'vitest';

import {
  normalizeMaterialParameterDescription,
  setMaterialParameterDescription,
} from './materialParameterDescription';

describe('material parameter descriptions', () => {
  it('normalizes authored help text and treats blank text as unset', () => {
    expect(normalizeMaterialParameterDescription('  Controls surface roughness.  ')).toBe(
      'Controls surface roughness.',
    );
    expect(normalizeMaterialParameterDescription('   ')).toBeUndefined();
    expect(normalizeMaterialParameterDescription(undefined)).toBeUndefined();
  });

  it('updates help text without changing stable parameter authoring metadata', () => {
    const parameters = [
      { id: 'roughness', name: 'Roughness', group: 'Surface', order: 2 },
      { id: 'metallic', name: 'Metallic', group: 'Surface', order: 1, description: 'Existing' },
    ];

    expect(setMaterialParameterDescription(parameters, 'roughness', '  Micro-surface response  ')).toEqual([
      {
        id: 'roughness',
        name: 'Roughness',
        group: 'Surface',
        order: 2,
        description: 'Micro-surface response',
      },
      parameters[1],
    ]);
    expect(parameters[0]).not.toHaveProperty('description');
  });

  it('clears descriptions without disturbing grouping or ordering', () => {
    expect(
      setMaterialParameterDescription(
        [{ id: 'roughness', name: 'Roughness', group: 'Surface', order: 4, description: 'Help' }],
        'roughness',
        ' ',
      ),
    ).toEqual([{ id: 'roughness', name: 'Roughness', group: 'Surface', order: 4, description: undefined }]);
  });

  it('leaves unrelated parameters untouched', () => {
    const parameters = [{ id: 'roughness', name: 'Roughness' }];
    const updated = setMaterialParameterDescription(parameters, 'missing', 'Help');
    expect(updated).toEqual(parameters);
    expect(updated[0]).toBe(parameters[0]);
  });
});

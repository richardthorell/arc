import { describe, expect, it } from 'vitest';

import {
  applyMaterialParameterMetadata,
  materialParameterMetadataFromAsset,
  withMaterialParameterMetadata,
} from './materialParameterMetadata';

describe('material parameter authoring metadata', () => {
  it('persists group/order by stable parameter id in deterministic order', () => {
    const asset = withMaterialParameterMetadata(
      { version: 4, name: 'Test' },
      [
        { id: 'roughness', name: 'Roughness', group: ' Surface ', order: 2 },
        { id: 'base-color', name: 'Base Color', group: 'Surface', order: 0 },
      ],
    );

    expect(asset.parameterMetadata).toEqual({
      version: 1,
      parameters: {
        'base-color': { group: 'Surface', order: 0 },
        roughness: { group: 'Surface', order: 2 },
      },
    });
  });

  it('loads only valid versioned entries and ignores malformed metadata', () => {
    const metadata = materialParameterMetadataFromAsset({
      parameterMetadata: {
        version: 1,
        parameters: {
          valid: { group: ' Surface ', order: 3 },
          negative: { order: -1 },
          junk: 'bad',
        },
      },
    });

    expect(metadata.parameters).toEqual({ valid: { group: 'Surface', order: 3 } });
    expect(materialParameterMetadataFromAsset({ parameterMetadata: { version: 99, parameters: {} } }).parameters).toEqual({});
  });

  it('applies presentation metadata without changing stable ids or names', () => {
    const descriptors = [
      { id: 'base-color', name: 'Base Color' },
      { id: 'roughness', name: 'Roughness' },
    ];
    const updated = applyMaterialParameterMetadata(descriptors, {
      version: 1,
      parameters: { roughness: { group: 'Surface', order: 1 } },
    });

    expect(updated).toEqual([
      { id: 'base-color', name: 'Base Color' },
      { id: 'roughness', name: 'Roughness', group: 'Surface', order: 1 },
    ]);
    expect(descriptors[1]).toEqual({ id: 'roughness', name: 'Roughness' });
  });

  it('removes empty metadata instead of persisting layout noise', () => {
    const asset = withMaterialParameterMetadata(
      { name: 'Test', parameterMetadata: { version: 1, parameters: { stale: { group: 'Old' } } } },
      [{ id: 'base-color', name: 'Base Color' }],
    );

    expect(asset.parameterMetadata).toBeUndefined();
  });
});

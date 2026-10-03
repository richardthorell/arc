import { describe, expect, it } from 'vitest';

import { buildAssetDependencyIndex } from './assetDependencyOperations';
import { buildAssetFindUsagesResult } from './assetFindUsages';

describe('asset Find Usages projection', () => {
  it('projects direct and transitive dependents with stable identity', () => {
    const index = buildAssetDependencyIndex([
      { sourceAssetId: 'material-a', targetAssetId: 'texture-a', kind: 'texture' },
      { sourceAssetId: 'scene-a', targetAssetId: 'material-a', kind: 'material' },
    ]);
    const descriptors = new Map([
      ['material-a', { assetId: 'material-a', label: 'Brushed Metal', logicalPath: 'Materials/Brushed.arc' }],
      ['scene-a', { assetId: 'scene-a', label: 'Showroom', logicalPath: 'Scenes/Showroom.arc' }],
    ]);

    expect(buildAssetFindUsagesResult(index, 'texture-a', descriptors)).toEqual({
      assetId: 'texture-a',
      directUsageCount: 1,
      rows: [
        {
          assetId: 'material-a',
          label: 'Brushed Metal',
          logicalPath: 'Materials/Brushed.arc',
          kind: 'texture',
          depth: 1,
          direct: true,
        },
        {
          assetId: 'scene-a',
          label: 'Showroom',
          logicalPath: 'Scenes/Showroom.arc',
          kind: 'material',
          depth: 2,
          direct: false,
        },
      ],
    });
  });

  it('falls back to stable IDs when registry presentation data is missing', () => {
    const index = buildAssetDependencyIndex([
      { sourceAssetId: 'unknown-dependent', targetAssetId: 'texture-a', kind: 'texture' },
    ]);

    expect(buildAssetFindUsagesResult(index, 'texture-a', new Map())).toEqual({
      assetId: 'texture-a',
      directUsageCount: 1,
      rows: [
        {
          assetId: 'unknown-dependent',
          label: 'unknown-dependent',
          logicalPath: undefined,
          kind: 'texture',
          depth: 1,
          direct: true,
        },
      ],
    });
  });

  it('returns an empty result for assets without dependents', () => {
    expect(buildAssetFindUsagesResult(buildAssetDependencyIndex([]), 'unused', new Map())).toEqual({
      assetId: 'unused',
      directUsageCount: 0,
      rows: [],
    });
  });
});

import { describe, expect, it } from 'vitest';
import { buildAssetDependencyIndex } from './assetDependencyOperations';
import { describeAssetUsages } from './assetUsagePresentation';

describe('asset usage presentation', () => {
  it('groups direct usages by asset while preserving individual references', () => {
    const index = buildAssetDependencyIndex([
      { sourceAssetId: 'material-a', targetAssetId: 'texture-a', kind: 'normal' },
      { sourceAssetId: 'material-a', targetAssetId: 'texture-a', kind: 'albedo' },
      { sourceAssetId: 'material-b', targetAssetId: 'texture-a', kind: 'albedo' },
    ]);

    const usages = describeAssetUsages(index, 'texture-a');

    expect(usages.directAssetIds).toEqual(['material-a', 'material-b']);
    expect(usages.directReferences).toHaveLength(3);
    expect(usages.transitiveAssetIds).toEqual([]);
    expect(usages.hasUsages).toBe(true);
  });

  it('separates transitive dependents from direct usages', () => {
    const index = buildAssetDependencyIndex([
      { sourceAssetId: 'material-a', targetAssetId: 'texture-a', kind: 'texture' },
      { sourceAssetId: 'scene-a', targetAssetId: 'material-a', kind: 'material' },
      { sourceAssetId: 'prefab-a', targetAssetId: 'scene-a', kind: 'scene' },
    ]);

    expect(describeAssetUsages(index, 'texture-a')).toMatchObject({
      directAssetIds: ['material-a'],
      transitiveAssetIds: ['prefab-a', 'scene-a'],
      hasUsages: true,
    });
  });

  it('returns an explicit empty presentation for an unused asset', () => {
    expect(describeAssetUsages(buildAssetDependencyIndex([]), 'texture-a')).toEqual({
      assetId: 'texture-a',
      directReferences: [],
      directAssetIds: [],
      transitiveAssetIds: [],
      hasUsages: false,
    });
  });
});

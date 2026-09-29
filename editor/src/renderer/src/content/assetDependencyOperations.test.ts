import { describe, expect, it } from 'vitest';
import {
  buildAssetDependencyIndex,
  describeAssetDependencyImpact,
  findAssetUsages,
  findTransitiveDependentAssetIds,
  planAssetBulkDelete,
  planAssetDelete,
  planAssetRelocation,
} from './assetDependencyOperations';

describe('asset dependency operations', () => {
  const references = [
    { sourceAssetId: 'scene-b', targetAssetId: 'material-a', kind: 'material' },
    { sourceAssetId: 'scene-a', targetAssetId: 'material-a', kind: 'material' },
    { sourceAssetId: 'material-a', targetAssetId: 'texture-a', kind: 'texture' },
  ];

  it('finds usages deterministically by stable asset identity', () => {
    const index = buildAssetDependencyIndex(references);

    expect(findAssetUsages(index, 'material-a')).toEqual([
      { sourceAssetId: 'scene-a', targetAssetId: 'material-a', kind: 'material' },
      { sourceAssetId: 'scene-b', targetAssetId: 'material-a', kind: 'material' },
    ]);
    expect(findAssetUsages(index, 'missing')).toEqual([]);
  });

  it('finds transitive dependents without duplicating shared paths', () => {
    const index = buildAssetDependencyIndex([
      ...references,
      { sourceAssetId: 'level-a', targetAssetId: 'scene-a' },
      { sourceAssetId: 'level-a', targetAssetId: 'scene-b' },
      { sourceAssetId: 'package-a', targetAssetId: 'level-a' },
    ]);

    expect(findTransitiveDependentAssetIds(index, 'texture-a')).toEqual([
      'material-a',
      'scene-a',
      'level-a',
      'package-a',
      'scene-b',
    ]);
    expect(describeAssetDependencyImpact(index, 'material-a')).toEqual({
      assetId: 'material-a',
      directDependents: [
        { sourceAssetId: 'scene-a', targetAssetId: 'material-a', kind: 'material' },
        { sourceAssetId: 'scene-b', targetAssetId: 'material-a', kind: 'material' },
      ],
      transitiveDependentAssetIds: ['scene-a', 'level-a', 'package-a', 'scene-b'],
    });
  });

  it('handles dependency cycles without reporting the queried asset as its own dependent', () => {
    const index = buildAssetDependencyIndex([
      { sourceAssetId: 'asset-b', targetAssetId: 'asset-a' },
      { sourceAssetId: 'asset-c', targetAssetId: 'asset-b' },
      { sourceAssetId: 'asset-a', targetAssetId: 'asset-c' },
    ]);

    expect(findTransitiveDependentAssetIds(index, 'asset-a')).toEqual(['asset-b', 'asset-c']);
  });

  it('blocks deletion planning while dependents exist', () => {
    const index = buildAssetDependencyIndex(references);

    expect(planAssetDelete(index, 'material-a')).toMatchObject({ safe: false });
    expect(planAssetDelete(index, 'unused')).toEqual({
      assetId: 'unused',
      dependents: [],
      safe: true,
    });
  });

  it('allows bulk deletion when all dependents are selected together', () => {
    const index = buildAssetDependencyIndex([
      { sourceAssetId: 'material-a', targetAssetId: 'texture-a', kind: 'texture' },
      { sourceAssetId: 'scene-a', targetAssetId: 'material-a', kind: 'material' },
    ]);

    expect(planAssetBulkDelete(index, ['material-a', 'texture-a', 'material-a'])).toEqual({
      assetIds: ['material-a', 'texture-a'],
      internalReferences: [{ sourceAssetId: 'material-a', targetAssetId: 'texture-a', kind: 'texture' }],
      blockingReferences: [{ sourceAssetId: 'scene-a', targetAssetId: 'material-a', kind: 'material' }],
      safe: false,
    });

    expect(planAssetBulkDelete(index, ['scene-a', 'texture-a', 'material-a'])).toEqual({
      assetIds: ['material-a', 'scene-a', 'texture-a'],
      internalReferences: [
        { sourceAssetId: 'material-a', targetAssetId: 'texture-a', kind: 'texture' },
        { sourceAssetId: 'scene-a', targetAssetId: 'material-a', kind: 'material' },
      ],
      blockingReferences: [],
      safe: true,
    });
  });

  it('keeps stable identity explicit when planning rename or move', () => {
    expect(planAssetRelocation('asset-guid', 'Materials/Old.arc', 'Materials/New.arc')).toEqual({
      assetId: 'asset-guid',
      fromPath: 'Materials/Old.arc',
      toPath: 'Materials/New.arc',
      preserveIdentity: true,
    });
  });

  it('rejects no-op relocation plans', () => {
    expect(() => planAssetRelocation('asset-guid', 'A.arc', 'A.arc')).toThrow(
      'Asset relocation requires a different destination path',
    );
  });
});

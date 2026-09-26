import { describe, expect, it } from 'vitest';
import {
  buildAssetDependencyIndex,
  findAssetUsages,
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

  it('blocks deletion planning while dependents exist', () => {
    const index = buildAssetDependencyIndex(references);

    expect(planAssetDelete(index, 'material-a')).toMatchObject({ safe: false });
    expect(planAssetDelete(index, 'unused')).toEqual({
      assetId: 'unused',
      dependents: [],
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

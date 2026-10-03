import { describe, expect, it } from 'vitest';
import {
  buildAssetDependencyIndex,
  describeAssetDependencyImpact,
  findAssetUsages,
  findTransitiveDependentAssetIds,
  planAssetBulkDelete,
  planAssetDelete,
  planAssetDeleteTransaction,
  planAssetRelocation,
  planAssetRelocationTransaction,
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

  it('plans delete mutations as explicit atomic transactions', () => {
    const index = buildAssetDependencyIndex([
      { sourceAssetId: 'material-a', targetAssetId: 'texture-a', kind: 'texture' },
      { sourceAssetId: 'scene-a', targetAssetId: 'material-a', kind: 'material' },
    ]);

    expect(planAssetDeleteTransaction(index, ['texture-a', 'material-a'])).toEqual({
      assetIds: ['material-a', 'texture-a'],
      internalReferences: [{ sourceAssetId: 'material-a', targetAssetId: 'texture-a', kind: 'texture' }],
      blockingReferences: [{ sourceAssetId: 'scene-a', targetAssetId: 'material-a', kind: 'material' }],
      atomic: true,
      executable: false,
    });

    expect(planAssetDeleteTransaction(index, ['scene-a', 'texture-a', 'material-a'])).toEqual({
      assetIds: ['material-a', 'scene-a', 'texture-a'],
      internalReferences: [
        { sourceAssetId: 'material-a', targetAssetId: 'texture-a', kind: 'texture' },
        { sourceAssetId: 'scene-a', targetAssetId: 'material-a', kind: 'material' },
      ],
      blockingReferences: [],
      atomic: true,
      executable: true,
    });
  });

  it('never treats an empty delete selection as an executable transaction', () => {
    expect(planAssetDeleteTransaction(buildAssetDependencyIndex([]), [])).toEqual({
      assetIds: [],
      internalReferences: [],
      blockingReferences: [],
      atomic: true,
      executable: false,
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

  it('normalizes relocation paths before checking for no-op moves', () => {
    expect(() => planAssetRelocation('asset-guid', './Materials/Old.arc', 'Materials\\Old.arc')).toThrow(
      'Asset relocation requires a different destination path',
    );
  });

  it('plans bulk relocations as one deterministic identity-preserving transaction', () => {
    expect(
      planAssetRelocationTransaction([
        { assetId: 'texture-b', fromPath: 'Textures/B.arc', toPath: 'Shared\\B.arc' },
        { assetId: 'material-a', fromPath: './Materials/A.arc', toPath: 'Shared/A.arc' },
      ]),
    ).toEqual({
      atomic: true,
      operations: [
        { assetId: 'material-a', fromPath: 'Materials/A.arc', toPath: 'Shared/A.arc', preserveIdentity: true },
        { assetId: 'texture-b', fromPath: 'Textures/B.arc', toPath: 'Shared/B.arc', preserveIdentity: true },
      ],
    });
  });

  it('rejects ambiguous bulk relocation transactions before mutation', () => {
    expect(() =>
      planAssetRelocationTransaction([
        { assetId: 'asset-a', fromPath: 'A.arc', toPath: 'Moved/A.arc' },
        { assetId: 'asset-a', fromPath: 'B.arc', toPath: 'Moved/B.arc' },
      ]),
    ).toThrow('Asset relocation contains duplicate asset id: asset-a');

    expect(() =>
      planAssetRelocationTransaction([
        { assetId: 'asset-a', fromPath: 'A.arc', toPath: 'Moved/Same.arc' },
        { assetId: 'asset-b', fromPath: 'B.arc', toPath: 'Moved\\Same.arc' },
      ]),
    ).toThrow('Asset relocation contains duplicate destination path: Moved/Same.arc');

    expect(() =>
      planAssetRelocationTransaction([
        { assetId: 'asset-a', fromPath: 'Same.arc', toPath: 'Moved/A.arc' },
        { assetId: 'asset-b', fromPath: './Same.arc', toPath: 'Moved/B.arc' },
      ]),
    ).toThrow('Asset relocation contains duplicate source path: Same.arc');
  });

  it('rejects relocation into an occupied asset path before mutation', () => {
    expect(() =>
      planAssetRelocationTransaction(
        [{ assetId: 'asset-a', fromPath: 'A.arc', toPath: 'Existing\\B.arc' }],
        ['A.arc', './Existing/B.arc'],
      ),
    ).toThrow('Asset relocation destination is already occupied: Existing/B.arc');
  });

  it('allows atomic swaps when every occupied destination is vacated by the transaction', () => {
    expect(
      planAssetRelocationTransaction(
        [
          { assetId: 'asset-b', fromPath: 'B.arc', toPath: 'A.arc' },
          { assetId: 'asset-a', fromPath: 'A.arc', toPath: 'B.arc' },
        ],
        ['A.arc', 'B.arc', 'Untouched.arc'],
      ),
    ).toEqual({
      atomic: true,
      operations: [
        { assetId: 'asset-a', fromPath: 'A.arc', toPath: 'B.arc', preserveIdentity: true },
        { assetId: 'asset-b', fromPath: 'B.arc', toPath: 'A.arc', preserveIdentity: true },
      ],
    });
  });
});

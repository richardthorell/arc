import { describe, expect, it } from 'vitest';

import { normalizeAgentAssetOperation } from './agentAssetOperations';

describe('normalizeAgentAssetOperation', () => {
  it('normalizes move operations around stable asset identities', () => {
    expect(
      normalizeAgentAssetOperation({
        kind: 'asset.move',
        assetIds: [' asset-a ', 'asset-b', 'asset-a'],
        destinationFolder: './Materials/Characters/',
        expectedProjectRevision: 12,
      }),
    ).toEqual({
      kind: 'asset.move',
      assetIds: ['asset-a', 'asset-b'],
      destinationFolder: 'Materials/Characters',
      expectedProjectRevision: 12,
    });
  });

  it('keeps delete operations identity and revision based', () => {
    expect(
      normalizeAgentAssetOperation({
        kind: 'asset.delete',
        assetIds: [' texture-guid '],
        expectedProjectRevision: 4,
      }),
    ).toEqual({
      kind: 'asset.delete',
      assetIds: ['texture-guid'],
      expectedProjectRevision: 4,
    });
  });

  it('rejects mutations without stable asset identities', () => {
    expect(() =>
      normalizeAgentAssetOperation({
        kind: 'asset.delete',
        assetIds: ['  '],
        expectedProjectRevision: 1,
      }),
    ).toThrow('at least one stable asset ID');
  });

  it('rejects move destinations outside the project asset workspace', () => {
    expect(() =>
      normalizeAgentAssetOperation({
        kind: 'asset.move',
        assetIds: ['asset-a'],
        destinationFolder: '../External',
        expectedProjectRevision: 1,
      }),
    ).toThrow('inside the project asset workspace');
  });

  it('rejects invalid project revisions', () => {
    expect(() =>
      normalizeAgentAssetOperation({
        kind: 'asset.delete',
        assetIds: ['asset-a'],
        expectedProjectRevision: -1,
      }),
    ).toThrow('Project revision');
  });
});

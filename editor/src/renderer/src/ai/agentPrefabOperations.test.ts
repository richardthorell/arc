import { describe, expect, it } from 'vitest';

import { normalizeAgentPrefabOperation } from './agentPrefabOperations';

describe('normalizeAgentPrefabOperation', () => {
  it('normalizes prefab creation around stable source identities', () => {
    expect(
      normalizeAgentPrefabOperation({
        kind: 'prefab.create',
        entityIds: [' entity-a ', 'entity-b', 'entity-a'],
        destinationPath: 'Prefabs/Enemy.arcprefab',
        expectedSceneRevision: 12,
      }),
    ).toEqual({
      kind: 'prefab.create',
      entityIds: ['entity-a', 'entity-b'],
      destinationPath: 'Prefabs/Enemy.arcprefab',
      expectedSceneRevision: 12,
    });
  });

  it('keeps instantiate identity and revisions explicit', () => {
    expect(
      normalizeAgentPrefabOperation({
        kind: 'prefab.instantiate',
        assetId: ' prefab-guid ',
        expectedAssetRevision: 4,
        expectedSceneRevision: 20,
        parentEntityId: ' root ',
      }),
    ).toEqual({
      kind: 'prefab.instantiate',
      assetId: 'prefab-guid',
      expectedAssetRevision: 4,
      expectedSceneRevision: 20,
      parentEntityId: 'root',
    });
  });

  it('rejects creation without source entities', () => {
    expect(() =>
      normalizeAgentPrefabOperation({
        kind: 'prefab.create',
        entityIds: [],
        destinationPath: 'Prefabs/Empty.arcprefab',
        expectedSceneRevision: 1,
      }),
    ).toThrow('at least one source entity');
  });

  it('rejects destinations that escape the project asset workspace', () => {
    expect(() =>
      normalizeAgentPrefabOperation({
        kind: 'prefab.create',
        entityIds: ['entity-a'],
        destinationPath: '../Outside.arcprefab',
        expectedSceneRevision: 1,
      }),
    ).toThrow('inside the project asset workspace');
  });

  it('rejects stale-contract revisions that are not valid revision numbers', () => {
    expect(() =>
      normalizeAgentPrefabOperation({
        kind: 'prefab.instantiate',
        assetId: 'prefab-guid',
        expectedAssetRevision: -1,
        expectedSceneRevision: 1,
      }),
    ).toThrow('Prefab asset revision');
  });
});

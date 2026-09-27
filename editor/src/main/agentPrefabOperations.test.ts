import { describe, expect, it } from 'vitest';

import { parseAgentPrefabOperation, requireAgentPrefabPath } from './agentPrefabOperations';

describe('agent prefab operations', () => {
  it('normalizes prefab creation around a stable root guid and project-relative asset path', () => {
    expect(
      parseAgentPrefabOperation('createPrefab', {
        rootGuid: '9db41d89-0f30-48e0-bf86-62899a279d7d',
        path: 'Content/Prefabs/Crate.arcprefab',
      }),
    ).toEqual({
      action: 'createPrefab',
      value: {
        rootGuid: '9db41d89-0f30-48e0-bf86-62899a279d7d',
        path: 'Content/Prefabs/Crate.arcprefab',
      },
    });
  });

  it('supports prefab instantiation with an optional stable parent guid', () => {
    expect(
      parseAgentPrefabOperation('instantiatePrefab', {
        path: 'Content/Prefabs/Crate.arcprefab',
        parentGuid: 'parent-guid',
      }),
    ).toEqual({
      action: 'instantiatePrefab',
      value: { path: 'Content/Prefabs/Crate.arcprefab', parentGuid: 'parent-guid' },
    });
  });

  it.each([
    '../Crate.arcprefab',
    '/Content/Crate.arcprefab',
    'C:/Content/Crate.arcprefab',
    'Content\\Crate.arcprefab',
    'Content/Crate.arcscene',
  ])('rejects prefab paths outside the project prefab contract: %s', (path) => {
    expect(() => requireAgentPrefabPath(path)).toThrow();
  });

  it('rejects missing stable entity identity', () => {
    expect(() => parseAgentPrefabOperation('createPrefab', { path: 'Content/Crate.arcprefab' })).toThrow(
      'value.rootGuid must be a non-empty string',
    );
  });
});

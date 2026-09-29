import { describe, expect, it } from 'vitest';

import {
  assetLibraryScope,
  assetLibraryScopes,
  assetMutationScope,
  assetsInScope,
  isAssetScopeWritable,
} from './assetLibraryScopes';

describe('assetLibraryScopes', () => {
  it('defines the logical library mounts independently from asset identity', () => {
    expect(assetLibraryScopes.map((scope) => scope.id)).toEqual(['builtin', 'project', 'user', 'organization']);
    expect(isAssetScopeWritable('builtin')).toBe(false);
    expect(isAssetScopeWritable('project')).toBe(true);
    expect(isAssetScopeWritable('user')).toBe(true);
    expect(isAssetScopeWritable('organization')).toBe(false);
  });

  it('treats legacy assets without a scope as project assets', () => {
    const assets = [
      { scope: undefined, guid: 'same-guid' },
      { scope: 'builtin', guid: 'same-guid' },
      { scope: 'user', guid: 'user-guid' },
    ] as const;

    expect(assetsInScope(assets, 'project')).toEqual([assets[0]]);
    expect(assetsInScope(assets, 'builtin')).toEqual([assets[1]]);
    expect(assetsInScope(assets, 'user')).toEqual([assets[2]]);
  });

  it('falls back safely when an older host reports an unknown scope', () => {
    expect(assetLibraryScope('legacy').id).toBe('project');
  });

  it('routes mutations only to writable logical scopes', () => {
    expect(assetMutationScope('project')).toBe('project');
    expect(assetMutationScope('user')).toBe('user');
    expect(assetMutationScope('builtin')).toBe('project');
    expect(assetMutationScope('organization')).toBe('project');
  });

  it('preserves legacy project mutation behavior for missing or unknown scopes', () => {
    expect(assetMutationScope(undefined)).toBe('project');
    expect(assetMutationScope(null)).toBe('project');
    expect(assetMutationScope('legacy')).toBe('project');
  });
});

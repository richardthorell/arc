import { describe, expect, it } from 'vitest';

import { buildAssetLibraryScopeNavigation } from './assetLibraryScopeNavigation';

describe('buildAssetLibraryScopeNavigation', () => {
  it('keeps the shared logical scope order and exposes writable state', () => {
    const navigation = buildAssetLibraryScopeNavigation([
      { scope: 'organization' },
      { scope: 'project' },
      { scope: 'user' },
      { scope: 'project' },
      { scope: 'builtin' },
    ]);

    expect(navigation.map((scope) => scope.id)).toEqual(['builtin', 'project', 'user', 'organization']);
    expect(navigation.map((scope) => scope.writable)).toEqual([false, true, true, false]);
    expect(navigation.map((scope) => scope.assetCount)).toEqual([1, 2, 1, 1]);
  });

  it('keeps empty configured scopes visible and treats legacy unscoped assets as project assets', () => {
    const navigation = buildAssetLibraryScopeNavigation([{ scope: undefined }, { scope: undefined }]);

    expect(navigation).toHaveLength(4);
    expect(navigation.find((scope) => scope.id === 'project')?.assetCount).toBe(2);
    expect(navigation.find((scope) => scope.id === 'builtin')?.assetCount).toBe(0);
    expect(navigation.find((scope) => scope.id === 'user')?.assetCount).toBe(0);
    expect(navigation.find((scope) => scope.id === 'organization')?.assetCount).toBe(0);
  });
});

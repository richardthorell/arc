import { describe, expect, it } from 'vitest';

import { buildAssetLibraryScopeNavigation, visibleAssetLibraryScopeNavigation } from './assetLibraryNavigation';

describe('asset library scope navigation', () => {
  it('keeps project and built-in roots stable for an empty project', () => {
    expect(visibleAssetLibraryScopeNavigation([]).map((scope) => scope.id)).toEqual(['builtin', 'project']);
  });

  it('surfaces optional user and organization mounts when assets are available', () => {
    const navigation = visibleAssetLibraryScopeNavigation([
      { scope: 'user' },
      { scope: 'organization' },
      { scope: 'organization' },
    ]);

    expect(navigation.map((scope) => scope.id)).toEqual(['builtin', 'project', 'user', 'organization']);
    expect(navigation.find((scope) => scope.id === 'user')).toMatchObject({ writable: true, assetCount: 1 });
    expect(navigation.find((scope) => scope.id === 'organization')).toMatchObject({
      writable: false,
      assetCount: 2,
    });
  });

  it('treats legacy unscoped assets as project assets without changing identity', () => {
    const navigation = buildAssetLibraryScopeNavigation([{ scope: undefined }, { scope: 'project' }]);

    expect(navigation.find((scope) => scope.id === 'project')).toMatchObject({ assetCount: 2, available: true });
  });

  it('does not make scope part of an asset identity contract', () => {
    const asset = { id: 'stable-asset', scope: 'project' as const };
    const moved = { ...asset, scope: 'user' as const };

    expect(asset.id).toBe(moved.id);
  });
});

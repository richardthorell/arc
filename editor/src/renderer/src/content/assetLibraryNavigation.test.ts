import { describe, expect, it } from 'vitest';

import {
  buildAssetLibraryNavigation,
  buildAssetLibraryScopeNavigation,
  buildAssetLibraryVirtualNavigation,
  visibleAssetLibraryScopeNavigation,
} from './assetLibraryNavigation';

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
    expect(navigation.find((scope) => scope.id === 'user')).toMatchObject({
      kind: 'scope',
      writable: true,
      assetCount: 1,
    });
    expect(navigation.find((scope) => scope.id === 'organization')).toMatchObject({
      kind: 'scope',
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

describe('asset library virtual navigation', () => {
  it('keeps virtual views distinct from storage scopes and counts stable membership', () => {
    const navigation = buildAssetLibraryVirtualNavigation({
      favorites: ['asset-a', 'asset-b', 'asset-a'],
      recent: ['asset-b'],
      downloads: [],
    });

    expect(navigation.map((view) => view.id)).toEqual(['favorites', 'recent', 'downloads']);
    expect(navigation.find((view) => view.id === 'favorites')).toMatchObject({
      kind: 'virtual-view',
      assetCount: 2,
      persistent: true,
    });
  });

  it('only exposes transient Search Results while results exist', () => {
    expect(buildAssetLibraryVirtualNavigation({}, []).some((view) => view.id === 'search-results')).toBe(false);

    expect(buildAssetLibraryVirtualNavigation({}, ['asset-a', 'asset-a', 'asset-b'])).toContainEqual(
      expect.objectContaining({
        id: 'search-results',
        kind: 'virtual-view',
        assetCount: 2,
        persistent: false,
      }),
    );
  });

  it('combines virtual views and logical scopes without conflating their identities', () => {
    const navigation = buildAssetLibraryNavigation(
      [{ scope: 'project' }, { scope: 'user' }],
      { favorites: ['asset-a'], recent: [], downloads: [] },
    );

    expect(navigation.filter((item) => item.kind === 'virtual-view').map((item) => item.id)).toEqual([
      'favorites',
      'recent',
      'downloads',
    ]);
    expect(navigation.filter((item) => item.kind === 'scope').map((item) => item.id)).toEqual([
      'builtin',
      'project',
      'user',
    ]);
  });
});

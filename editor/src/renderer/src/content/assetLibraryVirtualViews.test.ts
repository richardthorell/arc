import { describe, expect, it } from 'vitest';

import {
  assetIdsForVirtualView,
  assetLibraryVirtualView,
  assetLibraryVirtualViews,
  assetsForVirtualView,
  recordDownloadedAsset,
  recordRecentAsset,
  setAssetFavorite,
} from './assetLibraryVirtualViews';

describe('asset library virtual views', () => {
  it('keeps virtual collections distinct from storage semantics', () => {
    expect(assetLibraryVirtualViews().map((view) => view.id)).toEqual([
      'favorites',
      'recent',
      'downloads',
      'search-results',
    ]);
    expect(assetLibraryVirtualView('favorites')).toMatchObject({
      persistent: true,
      acceptsMembershipChanges: true,
    });
    expect(assetLibraryVirtualView('recent').acceptsMembershipChanges).toBe(false);
    expect(assetLibraryVirtualView('downloads').acceptsMembershipChanges).toBe(false);
    expect(assetLibraryVirtualView('search-results').persistent).toBe(false);
  });

  it('projects stable asset IDs into multiple views without duplication', () => {
    const membership = {
      favorites: ['asset-a', 'asset-b', 'asset-a'],
      recent: ['asset-b', 'asset-c'],
      downloads: ['asset-c'],
    };

    expect(assetIdsForVirtualView('favorites', membership)).toEqual(['asset-a', 'asset-b']);
    expect(assetIdsForVirtualView('recent', membership)).toEqual(['asset-b', 'asset-c']);
    expect(assetIdsForVirtualView('downloads', membership)).toEqual(['asset-c']);
  });

  it('resolves current assets in view order and ignores stale membership', () => {
    const assets = [
      { id: 'asset-a', scope: 'project' },
      { id: 'asset-b', scope: 'builtin' },
      { id: 'asset-c', scope: 'user' },
    ] as const;
    const membership = { favorites: ['asset-c', 'missing', 'asset-a', 'asset-c'] };

    const resolved = assetsForVirtualView(assets, 'favorites', membership);
    expect(resolved).toEqual([assets[2], assets[0]]);
    expect(resolved[0]).toBe(assets[2]);
    expect(assets[2].scope).toBe('user');
  });

  it('keeps search results transient instead of persisting collection membership', () => {
    const membership = { 'search-results': ['stale-result'] };
    expect(assetIdsForVirtualView('search-results', membership, ['asset-a', 'asset-a', 'asset-c'])).toEqual([
      'asset-a',
      'asset-c',
    ]);

    const assets = [{ id: 'asset-a' }, { id: 'asset-c' }] as const;
    expect(assetsForVirtualView(assets, 'search-results', membership, ['asset-c', 'asset-a'])).toEqual([
      assets[1],
      assets[0],
    ]);
  });

  it('toggles favorites without mutating other virtual views', () => {
    const membership = { favorites: ['asset-a'], recent: ['asset-b'] };
    const added = setAssetFavorite(membership, 'asset-b', true);
    expect(added).toEqual({ favorites: ['asset-a', 'asset-b'], recent: ['asset-b'] });
    expect(setAssetFavorite(added, 'asset-a', false)).toEqual({ favorites: ['asset-b'], recent: ['asset-b'] });
    expect(membership).toEqual({ favorites: ['asset-a'], recent: ['asset-b'] });
  });

  it('records bounded recent assets in newest-first order without duplicates', () => {
    const membership = { recent: ['asset-a', 'asset-b', 'asset-c'] };
    expect(recordRecentAsset(membership, 'asset-b', 3)).toEqual({
      recent: ['asset-b', 'asset-a', 'asset-c'],
    });
    expect(recordRecentAsset(membership, 'asset-d', 2)).toEqual({ recent: ['asset-d', 'asset-a'] });
  });

  it('records completed downloads as a derived newest-first view', () => {
    const membership = { downloads: ['asset-a', 'asset-b', 'asset-a'], favorites: ['asset-b'] };
    expect(recordDownloadedAsset(membership, 'asset-b')).toEqual({
      downloads: ['asset-b', 'asset-a'],
      favorites: ['asset-b'],
    });
  });
});

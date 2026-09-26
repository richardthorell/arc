import { describe, expect, it } from 'vitest';

import {
  assetIdsForVirtualView,
  assetLibraryVirtualView,
  assetLibraryVirtualViews,
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

  it('keeps search results transient instead of persisting collection membership', () => {
    const membership = { 'search-results': ['stale-result'] };
    expect(assetIdsForVirtualView('search-results', membership, ['asset-a', 'asset-a', 'asset-c'])).toEqual([
      'asset-a',
      'asset-c',
    ]);
  });
});

import { describe, expect, it } from 'vitest';

import {
  assetIdsMatchingImportedPaths,
  assetsForVirtualView,
  assetVirtualViewContains,
  assetVirtualViewStorageKey,
  defaultAssetVirtualViews,
  loadAssetVirtualViews,
  recordDownloadedAssets,
  recordRecentAsset,
  removeAssetFromVirtualCollection,
  saveAssetVirtualViews,
  setFavoriteAsset,
  type AssetVirtualViewStorage,
} from './assetVirtualViewIntegration';

const createStorage = (): AssetVirtualViewStorage & { values: Map<string, string> } => {
  const values = new Map<string, string>();
  return {
    values,
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => void values.set(key, value),
    removeItem: (key) => void values.delete(key),
  };
};

const assets = [
  { id: 'rock', guid: 'rock-guid', path: 'Content/Props/rock.glb', sourcePath: 'D:/Project/Content/Props/rock.glb' },
  { id: 'sky', guid: 'sky-guid', path: 'Content/Environment/sky.hdr', sourcePath: undefined },
];

describe('assetVirtualViewIntegration', () => {
  it('persists durable collections per project and never persists search results', () => {
    const storage = createStorage();
    let views = setFavoriteAsset(defaultAssetVirtualViews(), 'rock', true);
    views = recordRecentAsset(views, 'sky');
    views = recordDownloadedAssets(views, ['rock']);

    saveAssetVirtualViews(storage, 'D:/Project', views);

    expect(storage.values.has(assetVirtualViewStorageKey('D:/Project'))).toBe(true);
    expect(loadAssetVirtualViews(storage, 'D:/Project', assets)).toEqual(views);
    expect(loadAssetVirtualViews(storage, 'D:/Other', assets)).toEqual(defaultAssetVirtualViews());
    expect(storage.values.get(assetVirtualViewStorageKey('D:/Project'))).not.toContain('search-results');
  });

  it('migrates the legacy Favorites key to stable asset IDs once', () => {
    const storage = createStorage();
    storage.setItem('arc.content.favorites', JSON.stringify(['rock-guid', 'Content/Environment/sky.hdr']));

    const views = loadAssetVirtualViews(storage, 'D:/Project', assets);

    expect(assetVirtualViewContains(views, 'favorites', 'rock')).toBe(true);
    expect(assetVirtualViewContains(views, 'favorites', 'sky')).toBe(true);
    expect(storage.getItem('arc.content.favorites')).toBeNull();
    expect(storage.getItem(assetVirtualViewStorageKey('D:/Project'))).not.toBeNull();
  });

  it('records recent assets newest-first without duplication and keeps the history bounded', () => {
    let views = defaultAssetVirtualViews();
    views = recordRecentAsset(views, 'rock', 2);
    views = recordRecentAsset(views, 'sky', 2);
    views = recordRecentAsset(views, 'rock', 2);

    expect(assetsForVirtualView(views, 'recent', assets).map((asset) => asset.id)).toEqual(['rock', 'sky']);
  });

  it('records downloads independently from Favorites and supports collection-only removal', () => {
    let views = setFavoriteAsset(defaultAssetVirtualViews(), 'rock', true);
    views = recordDownloadedAssets(views, ['rock', 'sky']);
    views = removeAssetFromVirtualCollection(views, 'downloads', 'rock');

    expect(assetVirtualViewContains(views, 'favorites', 'rock')).toBe(true);
    expect(assetsForVirtualView(views, 'downloads', assets).map((asset) => asset.id)).toEqual(['sky']);
  });

  it('resolves imported file paths back to stable asset IDs', () => {
    expect(assetIdsMatchingImportedPaths(assets, ['Props/rock.glb', 'Content/Environment/sky.hdr'])).toEqual([
      'rock',
      'sky',
    ]);
  });

  it('projects transient search results without adding them to durable state', () => {
    const views = defaultAssetVirtualViews();
    const projected = assetsForVirtualView(views, 'search-results', assets, ['sky', 'rock', 'sky']);

    expect(projected.map((asset) => asset.id)).toEqual(['sky', 'rock']);
    expect(views).toEqual(defaultAssetVirtualViews());
  });
});

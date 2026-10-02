import { describe, expect, it } from 'vitest';

import { createAssetVirtualView } from './assetVirtualViews';
import { deserializeAssetVirtualViews, serializeAssetVirtualViews } from './assetVirtualViewPersistence';

describe('assetVirtualViewPersistence', () => {
  it('round-trips durable virtual views while keeping search results transient', () => {
    const serialized = serializeAssetVirtualViews([
      createAssetVirtualView('favorites', ['asset-a', 'asset-b']),
      createAssetVirtualView('recent', ['asset-b']),
      createAssetVirtualView('downloads', ['asset-c']),
      createAssetVirtualView('search-results', ['asset-d']),
    ]);

    expect(deserializeAssetVirtualViews(serialized)).toEqual([
      createAssetVirtualView('favorites', ['asset-a', 'asset-b']),
      createAssetVirtualView('recent', ['asset-b']),
      createAssetVirtualView('downloads', ['asset-c']),
    ]);
  });

  it('normalizes persisted membership through the virtual-view model', () => {
    const views = deserializeAssetVirtualViews(
      JSON.stringify({ version: 1, views: [{ kind: 'favorites', assetIds: [' asset-a ', '', 'asset-a'] }] }),
    );

    expect(views).toEqual([createAssetVirtualView('favorites', ['asset-a'])]);
  });

  it('rejects unsupported schemas and malformed payloads without inventing state', () => {
    expect(deserializeAssetVirtualViews('not-json')).toEqual([]);
    expect(deserializeAssetVirtualViews(JSON.stringify({ version: 2, views: [] }))).toEqual([]);
    expect(deserializeAssetVirtualViews(JSON.stringify({ version: 1, views: 'favorites' }))).toEqual([]);
  });

  it('ignores transient, unknown, malformed, and duplicate persisted collections', () => {
    const views = deserializeAssetVirtualViews(
      JSON.stringify({
        version: 1,
        views: [
          { kind: 'search-results', assetIds: ['asset-search'] },
          { kind: 'unknown', assetIds: ['asset-x'] },
          { kind: 'favorites', assetIds: ['asset-a'] },
          { kind: 'favorites', assetIds: ['asset-b'] },
          { kind: 'recent', assetIds: [42] },
        ],
      }),
    );

    expect(views).toEqual([createAssetVirtualView('favorites', ['asset-a'])]);
  });
});

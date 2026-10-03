import { describe, expect, it } from 'vitest';

import { createAssetVirtualView } from './assetVirtualViews';
import { getAssetVirtualViewActions } from './assetVirtualViewActions';

describe('assetVirtualViewActions', () => {
  it('preserves normal asset actions instead of treating a virtual view as storage', () => {
    const actions = getAssetVirtualViewActions(createAssetVirtualView('favorites', ['asset-a']), {
      assetId: 'asset-a',
      writable: true,
    });

    expect(actions).toEqual([
      { action: 'open', enabled: true },
      { action: 'reveal', enabled: true },
      { action: 'rename', enabled: true },
      { action: 'move', enabled: true },
      { action: 'delete', enabled: true },
      { action: 'remove-from-view', enabled: true },
    ]);
  });

  it('keeps storage mutations disabled for read-only assets while retaining safe actions', () => {
    const actions = getAssetVirtualViewActions(createAssetVirtualView('downloads', ['asset-a']), {
      assetId: 'asset-a',
      writable: false,
    });

    expect(actions).toEqual([
      { action: 'open', enabled: true },
      { action: 'reveal', enabled: true },
      { action: 'rename', enabled: false },
      { action: 'move', enabled: false },
      { action: 'delete', enabled: false },
      { action: 'remove-from-view', enabled: true },
    ]);
  });

  it('does not expose membership mutation for transient search results or derived recent history', () => {
    for (const kind of ['search-results', 'recent'] as const) {
      const actions = getAssetVirtualViewActions(createAssetVirtualView(kind, ['asset-a']), {
        assetId: 'asset-a',
        writable: true,
      });

      expect(actions.some(({ action }) => action === 'remove-from-view')).toBe(false);
      expect(actions.filter(({ action }) => action !== 'remove-from-view').every(({ enabled }) => enabled)).toBe(true);
    }
  });
});

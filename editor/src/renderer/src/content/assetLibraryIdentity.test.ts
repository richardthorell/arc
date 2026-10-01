import { describe, expect, it } from 'vitest';

import {
  assetLibraryIdentity,
  assetLibraryLogicalReference,
  referencesPreserveAssetIdentity,
} from './assetLibraryIdentity';

describe('assetLibraryIdentity', () => {
  const asset = { id: 'asset-42', guid: 'guid-42' };

  it('uses the ARC asset id without including logical scope', () => {
    expect(assetLibraryIdentity(asset)).toBe('asset-42');

    const project = assetLibraryLogicalReference(asset, 'project');
    const user = assetLibraryLogicalReference(asset, 'user');

    expect(project.assetId).toBe(user.assetId);
    expect(project.scope).not.toBe(user.scope);
  });

  it('keeps identity stable across virtual-view membership', () => {
    const favorites = assetLibraryLogicalReference(asset, 'project', 'favorites');
    const recent = assetLibraryLogicalReference(asset, 'project', 'recent');

    expect(favorites.assetId).toBe('asset-42');
    expect(recent.assetId).toBe('asset-42');
    expect(favorites.viewId).not.toBe(recent.viewId);
  });

  it('validates rebuilt logical references against authoritative asset ids', () => {
    expect(
      referencesPreserveAssetIdentity(
        [asset, { id: 'asset-99', guid: 'guid-99' }],
        [
          { assetId: 'asset-42' },
          { assetId: 'asset-99' },
        ],
      ),
    ).toBe(true);

    expect(referencesPreserveAssetIdentity([asset], [{ assetId: 'project:asset-42' }])).toBe(false);
  });
});

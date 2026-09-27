import { describe, expect, it } from 'vitest';

import type { AssetItem } from '../services/editorHostTypes';
import { assetMatchesMetadataSearch } from './assetMetadataSearch';

const asset: AssetItem = {
  id: 'asset-1',
  guid: 'guid-wood-floor',
  name: 'oak_floor',
  title: 'Warm Oak Floor',
  description: 'Scanned hardwood surface for interior scenes',
  tags: ['wood', 'flooring', 'photogrammetry'],
  path: 'Content/Materials/oak_floor.arcmaterial',
  kind: 'material',
  status: 'ready',
};

describe('assetMatchesMetadataSearch', () => {
  it('matches title, description, and tags in addition to identity fields', () => {
    expect(assetMatchesMetadataSearch(asset, 'warm')).toBe(true);
    expect(assetMatchesMetadataSearch(asset, 'interior')).toBe(true);
    expect(assetMatchesMetadataSearch(asset, 'photogrammetry')).toBe(true);
    expect(assetMatchesMetadataSearch(asset, 'guid-wood-floor')).toBe(true);
  });

  it('requires every search term while allowing terms to match different metadata fields', () => {
    expect(assetMatchesMetadataSearch(asset, 'warm flooring interior')).toBe(true);
    expect(assetMatchesMetadataSearch(asset, 'warm metal')).toBe(false);
  });

  it('is case-insensitive and treats blank queries as a match', () => {
    expect(assetMatchesMetadataSearch(asset, 'OAK FLOORING')).toBe(true);
    expect(assetMatchesMetadataSearch(asset, '   ')).toBe(true);
  });

  it('handles assets without optional metadata', () => {
    const minimal: AssetItem = {
      id: 'asset-2',
      name: 'sky',
      path: 'Content/sky.arcscene',
      kind: 'scene',
      status: 'ready',
    };
    expect(assetMatchesMetadataSearch(minimal, 'sky')).toBe(true);
    expect(assetMatchesMetadataSearch(minimal, 'tagged')).toBe(false);
  });
});

import { describe, expect, it } from 'vitest';

import { buildAssetMetadataFacets } from './assetMetadataFacets';

const assets = [
  { assetId: 'a', title: 'Brick', assetType: 'Material', tags: ['Metal', 'Industrial'] },
  { assetId: 'b', title: 'Steel', assetType: 'material', tags: ['metal', ' hard-surface '] },
  { assetId: 'c', title: 'Crate', assetType: 'Mesh', tags: ['Industrial', ''] },
];

describe('buildAssetMetadataFacets', () => {
  it('normalizes, counts, and deterministically orders metadata facets', () => {
    expect(buildAssetMetadataFacets(assets)).toEqual({
      assetTypes: [
        { value: 'material', count: 2 },
        { value: 'mesh', count: 1 },
      ],
      tags: [
        { value: 'industrial', count: 2 },
        { value: 'metal', count: 2 },
        { value: 'hard-surface', count: 1 },
      ],
    });
  });

  it('does not mutate source metadata', () => {
    const snapshot = JSON.stringify(assets);
    buildAssetMetadataFacets(assets);
    expect(JSON.stringify(assets)).toBe(snapshot);
  });
});

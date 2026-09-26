import { describe, expect, it } from 'vitest';
import { collectAssetFacets, filterAssets, matchesAssetSearch, type AssetSearchMetadata } from './assetSearch';

const assets: AssetSearchMetadata[] = [
  {
    id: 'stone',
    name: 'rough_stone',
    title: 'Rough Stone',
    description: 'Weathered cliff material',
    path: 'Assets/Materials/rough_stone.arcmat',
    kind: 'material',
    tags: ['Environment', 'Rock'],
  },
  {
    id: 'hero',
    name: 'hero.glb',
    title: 'Hero',
    description: 'Player character',
    path: 'Assets/Models/hero.glb',
    kind: 'scene',
    tags: ['Character'],
  },
];

describe('asset search', () => {
  it('matches normalized terms across metadata fields and tags', () => {
    expect(matchesAssetSearch(assets[0], 'CLIFF rock')).toBe(true);
    expect(matchesAssetSearch(assets[0], 'rough environment')).toBe(true);
    expect(matchesAssetSearch(assets[0], 'character')).toBe(false);
  });

  it('combines search terms with presentation-kind and tag facets deterministically', () => {
    expect(filterAssets(assets, 'hero', { kind: 'model' }).map((asset) => asset.id)).toEqual(['hero']);
    expect(filterAssets(assets, '', { tags: ['ROCK', 'environment'] }).map((asset) => asset.id)).toEqual(['stone']);
    expect(filterAssets(assets, 'stone', { kind: 'model' })).toEqual([]);
  });

  it('collects stable sorted facet values without duplicate tags', () => {
    const facets = collectAssetFacets([...assets, { ...assets[0], id: 'stone-2', tags: ['rock', 'Architecture'] }]);
    expect(facets.kinds).toEqual(['material', 'model']);
    expect(facets.tags).toEqual(['architecture', 'character', 'environment', 'rock']);
  });
});

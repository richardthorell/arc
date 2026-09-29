import { describe, expect, it } from 'vitest';
import type { AssetItem } from '../services/editorHostTypes';
import { searchAssetLibrary } from './assetLibrarySearch';

const asset = (overrides: Partial<AssetItem>): AssetItem => ({
  id: 'asset',
  name: 'Asset',
  path: 'Assets/Asset',
  kind: 'unknown',
  status: 'ready',
  ...overrides,
});

const assets: AssetItem[] = [
  asset({
    id: 'm1',
    name: 'Rock',
    title: 'Cliff Rock',
    description: 'Granite scan',
    tags: ['Environment', 'Stone'],
    path: 'Assets/Rocks/Rock',
    kind: 'mesh',
  }),
  asset({
    id: 't1',
    name: 'RockNormal',
    title: 'Rock Normal',
    tags: ['stone', 'PBR'],
    path: 'Assets/Rocks/RockNormal',
    kind: 'texture',
  }),
  asset({
    id: 'm2',
    name: 'Tree',
    description: 'Forest oak',
    tags: ['Environment'],
    path: 'Assets/Trees/Tree',
    kind: 'mesh',
  }),
];

describe('searchAssetLibrary', () => {
  it('searches name, title, description, path, and tags case-insensitively', () => {
    expect(searchAssetLibrary(assets, { text: 'granite environment' }).assets.map((item) => item.id)).toEqual(['m1']);
    expect(searchAssetLibrary(assets, { text: 'ROCK normal' }).assets.map((item) => item.id)).toEqual(['t1']);
  });

  it('requires all selected tags while normalizing case and whitespace', () => {
    expect(searchAssetLibrary(assets, { tags: [' stone ', 'ENVIRONMENT'] }).assets.map((item) => item.id)).toEqual([
      'm1',
    ]);
  });

  it('filters by kind without hiding useful kind facet counts', () => {
    const result = searchAssetLibrary(assets, { text: 'rock', kinds: ['mesh'] });
    expect(result.assets.map((item) => item.id)).toEqual(['m1']);
    expect(result.facets).toEqual([
      { kind: 'mesh', count: 1 },
      { kind: 'texture', count: 1 },
    ]);
  });

  it('preserves stable source ordering and does not mutate assets', () => {
    const before = assets.slice();
    expect(searchAssetLibrary(assets, {}).assets.map((item) => item.id)).toEqual(['m1', 't1', 'm2']);
    expect(assets).toEqual(before);
  });
});

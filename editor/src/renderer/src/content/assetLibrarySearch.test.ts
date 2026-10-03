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

  it('supports deterministic path-prefix filtering at folder boundaries', () => {
    expect(searchAssetLibrary(assets, { pathPrefix: ' assets\\rocks/ ' }).assets.map((item) => item.id)).toEqual([
      'm1',
      't1',
    ]);
    expect(searchAssetLibrary(assets, { pathPrefix: 'Assets/Rock' }).assets).toEqual([]);
  });

  it('excludes assets carrying any excluded tag', () => {
    expect(searchAssetLibrary(assets, { excludeTags: [' pbr ', 'environment'] }).assets.map((item) => item.id)).toEqual(
      [],
    );
    expect(searchAssetLibrary(assets, { tags: ['stone'], excludeTags: ['PBR'] }).assets.map((item) => item.id)).toEqual(
      ['m1'],
    );
  });

  it('filters by kind without hiding useful kind facet counts', () => {
    const result = searchAssetLibrary(assets, { text: 'rock', kinds: ['mesh'] });
    expect(result.assets.map((item) => item.id)).toEqual(['m1']);
    expect(result.facets).toEqual([
      { kind: 'mesh', count: 1 },
      { kind: 'texture', count: 1 },
    ]);
  });

  it('computes facets after metadata predicates and before kind filtering', () => {
    const result = searchAssetLibrary(assets, { pathPrefix: 'Assets/Rocks', excludeTags: ['PBR'], kinds: ['texture'] });
    expect(result.assets).toEqual([]);
    expect(result.facets).toEqual([{ kind: 'mesh', count: 1 }]);
  });

  it('preserves stable source ordering and does not mutate assets', () => {
    const before = assets.slice();
    expect(searchAssetLibrary(assets, {}).assets.map((item) => item.id)).toEqual(['m1', 't1', 'm2']);
    expect(assets).toEqual(before);
  });

  it('keeps large metadata libraries deterministic without requiring raw-path queries', () => {
    const library = Array.from({ length: 10_000 }, (_, index) =>
      asset({
        id: `asset-${index.toString().padStart(5, '0')}`,
        name: `Asset ${index}`,
        title: index % 250 === 0 ? `Landmark ${index}` : undefined,
        description: index % 2 === 0 ? 'Outdoor environment asset' : 'Interior gameplay asset',
        tags: [index % 2 === 0 ? 'Environment' : 'Gameplay', `Batch-${index % 20}`],
        path: `Assets/Generated/Batch-${index % 20}/Asset-${index}`,
        kind: index % 3 === 0 ? 'mesh' : 'texture',
      }),
    );

    const query = { text: 'landmark', tags: ['environment'] } as const;
    const first = searchAssetLibrary(library, query);
    const second = searchAssetLibrary(library, query);

    expect(first.assets.length).toBe(40);
    expect(second.assets.map((item) => item.id)).toEqual(first.assets.map((item) => item.id));
    expect(first.assets.every((item) => item.tags?.includes('Environment'))).toBe(true);
    expect(first.facets).toEqual(second.facets);
  });
});

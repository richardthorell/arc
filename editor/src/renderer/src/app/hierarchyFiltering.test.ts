import { describe, expect, it } from 'vitest';
import { filterHierarchy, type HierarchyFilterEntry } from './hierarchyFiltering';

const entries: readonly HierarchyFilterEntry[] = [
  { id: 'root', parentId: null, name: 'World' },
  { id: 'player', parentId: 'root', name: 'Player' },
  { id: 'weapon', parentId: 'player', name: 'Plasma Rifle' },
  { id: 'camera', parentId: 'root', name: 'Gameplay Camera' },
  { id: 'light', parentId: 'root', name: 'Sun Light' },
];

const sorted = (values: ReadonlySet<string>): string[] => [...values].sort();

describe('filterHierarchy', () => {
  it('returns all entries for an empty query', () => {
    const result = filterHierarchy(entries, '   ');
    expect(sorted(result.visibleIds)).toEqual(['camera', 'light', 'player', 'root', 'weapon']);
    expect(sorted(result.matchedIds)).toEqual(['camera', 'light', 'player', 'root', 'weapon']);
  });

  it('matches names case-insensitively and preserves ancestor context', () => {
    const result = filterHierarchy(entries, 'RIFLE');
    expect(sorted(result.matchedIds)).toEqual(['weapon']);
    expect(sorted(result.visibleIds)).toEqual(['player', 'root', 'weapon']);
  });

  it('keeps multiple matching branches visible', () => {
    const result = filterHierarchy(entries, 'play');
    expect(sorted(result.matchedIds)).toEqual(['camera', 'player']);
    expect(sorted(result.visibleIds)).toEqual(['camera', 'player', 'root']);
  });

  it('returns an empty result when nothing matches', () => {
    const result = filterHierarchy(entries, 'missing');
    expect(result.matchedIds.size).toBe(0);
    expect(result.visibleIds.size).toBe(0);
  });

  it('terminates safely for malformed parent cycles', () => {
    const cyclic: readonly HierarchyFilterEntry[] = [
      { id: 'a', parentId: 'b', name: 'Alpha' },
      { id: 'b', parentId: 'a', name: 'Beta' },
    ];
    const result = filterHierarchy(cyclic, 'alpha');
    expect(sorted(result.visibleIds)).toEqual(['a', 'b']);
  });
});

import { describe, expect, it } from 'vitest';
import { filterHierarchy, type HierarchyFilterEntry } from './hierarchyFilter';

const entries: readonly HierarchyFilterEntry[] = [
  { id: 'root', name: 'World', parentId: null },
  { id: 'camera', name: 'Main Camera', parentId: 'root' },
  { id: 'group', name: 'Gameplay', parentId: 'root' },
  { id: 'player', name: 'Player Character', parentId: 'group' },
  { id: 'light', name: 'Key Light', parentId: 'root' },
];

describe('filterHierarchy', () => {
  it('preserves authored order when no filter is active', () => {
    const result = filterHierarchy(entries, '   ');
    expect(result.visibleIds).toEqual(['root', 'camera', 'group', 'player', 'light']);
    expect([...result.matchedIds]).toEqual([]);
    expect([...result.expandedAncestorIds]).toEqual([]);
  });

  it('matches case-insensitively and keeps ancestor context visible', () => {
    const result = filterHierarchy(entries, 'PLAYER');
    expect(result.visibleIds).toEqual(['root', 'group', 'player']);
    expect([...result.matchedIds]).toEqual(['player']);
    expect([...result.expandedAncestorIds]).toEqual(['group', 'root']);
  });

  it('supports multiple matches without exposing unrelated siblings', () => {
    const result = filterHierarchy(entries, 'a');
    expect(result.visibleIds).toEqual(['root', 'camera', 'group', 'player']);
    expect([...result.matchedIds]).toEqual(['camera', 'group', 'player']);
  });

  it('does not loop forever on malformed parent cycles', () => {
    const cyclic: readonly HierarchyFilterEntry[] = [
      { id: 'a', name: 'Alpha', parentId: 'b' },
      { id: 'b', name: 'Beta', parentId: 'a' },
    ];
    const result = filterHierarchy(cyclic, 'alpha');
    expect(result.visibleIds).toEqual(['a', 'b']);
    expect([...result.matchedIds]).toEqual(['a']);
  });
});

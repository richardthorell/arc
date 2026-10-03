import { describe, expect, it } from 'vitest';
import { planHierarchyGroup, type HierarchyGroupEntry } from './hierarchyGrouping';

const entries: readonly HierarchyGroupEntry[] = [
  { id: 'root', parentId: null },
  { id: 'camera', parentId: 'root' },
  { id: 'player', parentId: 'root' },
  { id: 'light', parentId: 'root' },
  { id: 'weapon', parentId: 'player' },
];

describe('planHierarchyGroup', () => {
  it('preserves authored hierarchy order for selected siblings', () => {
    expect(planHierarchyGroup(entries, ['light', 'camera'])).toEqual({
      parentId: 'root',
      memberIds: ['camera', 'light'],
    });
  });

  it('allows grouping root-level siblings', () => {
    expect(
      planHierarchyGroup(
        [
          { id: 'a', parentId: null },
          { id: 'b', parentId: null },
        ],
        ['a', 'b'],
      ),
    ).toEqual({ parentId: null, memberIds: ['a', 'b'] });
  });

  it('rejects selections from different parents', () => {
    expect(planHierarchyGroup(entries, ['player', 'weapon'])).toBeNull();
  });

  it('rejects missing, duplicate, or single selections', () => {
    expect(planHierarchyGroup(entries, ['camera'])).toBeNull();
    expect(planHierarchyGroup(entries, ['camera', 'camera'])).toBeNull();
    expect(planHierarchyGroup(entries, ['camera', 'missing'])).toBeNull();
  });
});

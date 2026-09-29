import { describe, expect, it } from 'vitest';

import { filterHierarchy, planHierarchyMove } from './hierarchyOperations';

const entities = [
  { guid: 'root', parentGuid: '', siblingOrder: 0 },
  { guid: 'a', parentGuid: 'root', siblingOrder: 0 },
  { guid: 'b', parentGuid: 'root', siblingOrder: 1 },
  { guid: 'child', parentGuid: 'a', siblingOrder: 0 },
];

describe('hierarchy move planning', () => {
  it('reorders siblings deterministically without changing stable ids', () => {
    const plan = planHierarchyMove(entities, 'b', 'root', 0);
    expect(plan?.siblings).toEqual([
      { guid: 'b', parentGuid: 'root', siblingOrder: 0 },
      { guid: 'a', parentGuid: 'root', siblingOrder: 1 },
    ]);
    expect(plan?.moved.guid).toBe('b');
  });

  it('reparents and clamps insertion to the destination sibling range', () => {
    const plan = planHierarchyMove(entities, 'child', 'root', 99);
    expect(plan?.moved).toEqual({ guid: 'child', parentGuid: 'root', siblingOrder: 2 });
    expect(plan?.siblings.map((entity) => entity.guid)).toEqual(['a', 'b', 'child']);
  });

  it('rejects cycles, self-parenting, missing entities, and missing parents', () => {
    expect(planHierarchyMove(entities, 'root', 'child', 0)).toBeNull();
    expect(planHierarchyMove(entities, 'a', 'a', 0)).toBeNull();
    expect(planHierarchyMove(entities, 'missing', '', 0)).toBeNull();
    expect(planHierarchyMove(entities, 'a', 'missing', 0)).toBeNull();
  });
});

describe('hierarchy filtering', () => {
  const searchable = [
    { guid: 'root', parentGuid: '', siblingOrder: 0, label: 'World' },
    { guid: 'player', parentGuid: 'root', siblingOrder: 0, label: 'Player', searchTerms: ['Character', 'Camera'] },
    { guid: 'weapon', parentGuid: 'player', siblingOrder: 0, label: 'Sword', searchTerms: ['Mesh Renderer'] },
    { guid: 'light', parentGuid: 'root', siblingOrder: 1, label: 'Sun', searchTerms: ['Directional Light'] },
  ];

  it('matches labels and metadata case-insensitively while preserving source order', () => {
    expect(filterHierarchy(searchable, 'directional LIGHT')).toEqual({
      visibleGuids: ['root', 'light'],
      matchedGuids: ['light'],
    });
  });

  it('keeps ancestors visible so filtered matches retain hierarchy context', () => {
    expect(filterHierarchy(searchable, 'mesh renderer')).toEqual({
      visibleGuids: ['root', 'player', 'weapon'],
      matchedGuids: ['weapon'],
    });
  });

  it('supports multi-term matching across label and metadata fields', () => {
    expect(filterHierarchy(searchable, 'player camera')).toEqual({
      visibleGuids: ['root', 'player'],
      matchedGuids: ['player'],
    });
  });

  it('returns the complete hierarchy for an empty query and no rows for no match', () => {
    expect(filterHierarchy(searchable, '   ').visibleGuids).toEqual(['root', 'player', 'weapon', 'light']);
    expect(filterHierarchy(searchable, 'missing')).toEqual({ visibleGuids: [], matchedGuids: [] });
  });
});

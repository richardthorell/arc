import { describe, expect, it } from 'vitest';

import { planHierarchyMove } from './hierarchyOperations';

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

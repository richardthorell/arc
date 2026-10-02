import { describe, expect, it } from 'vitest';
import { buildHierarchyContextActions } from './hierarchyContextActions';

describe('buildHierarchyContextActions', () => {
  it('disables selection actions when no entity is selected', () => {
    const actions = buildHierarchyContextActions({ selectedIds: [], primaryId: null });
    expect(actions.every((action) => !action.enabled)).toBe(true);
  });

  it('enables rename and create-child only for a valid single primary selection', () => {
    const actions = buildHierarchyContextActions({ selectedIds: ['player'], primaryId: 'player' });
    expect(actions).toEqual([
      { id: 'hierarchy.rename', label: 'Rename', enabled: true },
      { id: 'hierarchy.duplicate', label: 'Duplicate', enabled: true },
      { id: 'hierarchy.create-child', label: 'Create Child', enabled: true },
      { id: 'hierarchy.delete', label: 'Delete', enabled: true, destructive: true },
    ]);
  });

  it('uses one deterministic multi-selection action set', () => {
    const actions = buildHierarchyContextActions({ selectedIds: ['a', 'b', 'c'], primaryId: 'b' });
    expect(actions).toEqual([
      { id: 'hierarchy.rename', label: 'Rename', enabled: false },
      { id: 'hierarchy.duplicate', label: 'Duplicate 3 Entities', enabled: true },
      { id: 'hierarchy.create-child', label: 'Create Child', enabled: false },
      { id: 'hierarchy.delete', label: 'Delete 3 Entities', enabled: true, destructive: true },
    ]);
  });

  it('does not enable primary-only actions for a stale primary entity', () => {
    const actions = buildHierarchyContextActions({ selectedIds: ['live'], primaryId: 'deleted' });
    expect(actions.find((action) => action.id === 'hierarchy.rename')?.enabled).toBe(false);
    expect(actions.find((action) => action.id === 'hierarchy.create-child')?.enabled).toBe(false);
  });
});

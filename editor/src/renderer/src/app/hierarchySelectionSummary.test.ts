import { describe, expect, it } from 'vitest';
import {
  summarizeHierarchySelection,
  type HierarchySelectionEntry,
} from './hierarchySelectionSummary';

const entries: readonly HierarchySelectionEntry[] = [
  { id: 'root', parentId: null, label: 'Root' },
  { id: 'camera', parentId: 'root', label: 'Camera' },
  { id: 'player', parentId: 'root', label: 'Player' },
  { id: 'weapon', parentId: 'player', label: 'Weapon' },
];

describe('summarizeHierarchySelection', () => {
  it('preserves authored order and reports a common parent', () => {
    expect(summarizeHierarchySelection(entries, ['player', 'camera'])).toEqual({
      count: 2,
      validIds: ['camera', 'player'],
      staleIds: [],
      commonParentId: 'root',
      hasCommonParent: true,
      label: '2 entities',
    });
  });

  it('uses the entity label for a single valid selection', () => {
    expect(summarizeHierarchySelection(entries, ['weapon']).label).toBe('Weapon');
  });

  it('separates stale IDs from the visible selection', () => {
    expect(summarizeHierarchySelection(entries, ['camera', 'deleted'])).toEqual({
      count: 1,
      validIds: ['camera'],
      staleIds: ['deleted'],
      commonParentId: 'root',
      hasCommonParent: true,
      label: 'Camera',
    });
  });

  it('reports mixed-parent selections without inventing a common parent', () => {
    const summary = summarizeHierarchySelection(entries, ['camera', 'weapon']);
    expect(summary.hasCommonParent).toBe(false);
    expect(summary.commonParentId).toBe('root');
  });

  it('returns a stable empty summary', () => {
    expect(summarizeHierarchySelection(entries, [])).toEqual({
      count: 0,
      validIds: [],
      staleIds: [],
      commonParentId: null,
      hasCommonParent: false,
      label: 'No selection',
    });
  });
});

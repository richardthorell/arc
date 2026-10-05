import { describe, expect, it } from 'vitest';
import { reconcileHierarchySelection, updateHierarchySelection } from './hierarchySelection';

const visible = ['root', 'camera', 'player', 'light', 'props'];

describe('updateHierarchySelection', () => {
  it('replaces selection for a plain click', () => {
    expect(updateHierarchySelection(['camera', 'light'], 'player', visible)).toEqual(['player']);
  });

  it('toggles entities without disturbing the remaining selection order', () => {
    expect(updateHierarchySelection(['camera', 'light'], 'player', visible, { toggle: true })).toEqual([
      'camera',
      'light',
      'player',
    ]);
    expect(updateHierarchySelection(['camera', 'light'], 'camera', visible, { toggle: true })).toEqual(['light']);
  });

  it('selects a contiguous range in visible hierarchy order', () => {
    expect(updateHierarchySelection(['root'], 'light', visible, { range: true, anchorId: 'camera' })).toEqual([
      'camera',
      'player',
      'light',
    ]);
  });

  it('adds a range when toggle and range modifiers are combined', () => {
    expect(
      updateHierarchySelection(['root'], 'light', visible, {
        toggle: true,
        range: true,
        anchorId: 'camera',
      }),
    ).toEqual(['root', 'camera', 'player', 'light']);
  });

  it('falls back to a plain target selection when the range anchor is not visible', () => {
    expect(updateHierarchySelection(['root'], 'light', visible, { range: true, anchorId: 'hidden-child' })).toEqual([
      'light',
    ]);
  });
});

describe('reconcileHierarchySelection', () => {
  it('drops deleted entities and duplicate selection entries deterministically', () => {
    expect(reconcileHierarchySelection(['player', 'deleted', 'player', 'light'], visible)).toEqual(['player', 'light']);
  });
});

import { describe, expect, it } from 'vitest';
import { pruneHierarchySelection, updateHierarchySelection } from './hierarchySelection';

const visible = ['root', 'a', 'b', 'c', 'd'] as const;

describe('updateHierarchySelection', () => {
  it('replaces selection and establishes the range anchor', () => {
    const result = updateHierarchySelection({ ids: new Set(['a', 'b']), anchorId: 'a' }, visible, 'c', 'replace');
    expect([...result.ids]).toEqual(['c']);
    expect(result.anchorId).toBe('c');
  });

  it('toggles one row without discarding the rest of the selection', () => {
    const result = updateHierarchySelection({ ids: new Set(['a', 'b']), anchorId: 'a' }, visible, 'b', 'toggle');
    expect([...result.ids]).toEqual(['a']);
    expect(result.anchorId).toBe('b');
  });

  it('selects the contiguous visible range from the anchor', () => {
    const result = updateHierarchySelection({ ids: new Set(['a']), anchorId: 'a' }, visible, 'd', 'range');
    expect([...result.ids]).toEqual(['a', 'b', 'c', 'd']);
    expect(result.anchorId).toBe('a');
  });

  it('falls back to replacement when the range anchor is no longer visible', () => {
    const result = updateHierarchySelection({ ids: new Set(['root']), anchorId: 'hidden' }, visible, 'b', 'range');
    expect([...result.ids]).toEqual(['b']);
    expect(result.anchorId).toBe('b');
  });

  it('ignores targets that are not in the visible hierarchy', () => {
    const selection = { ids: new Set(['a']), anchorId: 'a' };
    expect(updateHierarchySelection(selection, visible, 'hidden', 'replace')).toBe(selection);
  });
});

describe('pruneHierarchySelection', () => {
  it('drops deleted entities and invalidates a deleted anchor', () => {
    const result = pruneHierarchySelection({ ids: new Set(['a', 'b']), anchorId: 'b' }, new Set(['a', 'c']));
    expect([...result.ids]).toEqual(['a']);
    expect(result.anchorId).toBeNull();
  });
});

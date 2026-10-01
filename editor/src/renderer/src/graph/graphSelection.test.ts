import { describe, expect, it } from 'vitest';

import {
  createGraphClipboardSnapshot,
  createGraphSelection,
  navigateGraphSelection,
  reconcileGraphSelection,
  remapGraphClipboardSnapshot,
  selectGraphMarquee,
  selectGraphNode,
  selectGraphRange,
} from './graphSelection';

describe('graphSelection', () => {
  it('supports replace and additive toggle selection', () => {
    let selection = selectGraphNode(createGraphSelection(), 'a');
    selection = selectGraphNode(selection, 'b', true);
    expect([...selection.ids]).toEqual(['a', 'b']);

    selection = selectGraphNode(selection, 'a', true);
    expect([...selection.ids]).toEqual(['b']);
    expect(selection.anchor).toBe('a');
  });

  it('reconciles stale selection and anchor after graph mutations', () => {
    const selection = createGraphSelection(['removed', 'b', 'a'], 'removed');
    const reconciled = reconcileGraphSelection(selection, ['a', 'b', 'c']);
    expect([...reconciled.ids]).toEqual(['b', 'a']);
    expect(reconciled.anchor).toBe('a');

    const empty = reconcileGraphSelection(createGraphSelection(['removed'], 'removed'), ['a']);
    expect([...empty.ids]).toEqual([]);
    expect(empty.anchor).toBeNull();
  });

  it('selects a contiguous range from the stable anchor', () => {
    const initial = selectGraphNode(createGraphSelection(), 'b');
    const selection = selectGraphRange(initial, ['a', 'b', 'c', 'd'], 'd');
    expect([...selection.ids]).toEqual(['b', 'c', 'd']);
    expect(selection.anchor).toBe('b');
  });

  it('selects graph nodes intersecting a marquee independent of drag direction', () => {
    const selectable = [
      { id: 'a', bounds: { x: 0, y: 0, width: 20, height: 20 } },
      { id: 'b', bounds: { x: 30, y: 30, width: 20, height: 20 } },
      { id: 'c', bounds: { x: 80, y: 80, width: 20, height: 20 } },
    ];

    const forward = selectGraphMarquee(createGraphSelection(), selectable, {
      x: 10,
      y: 10,
      width: 50,
      height: 50,
    });
    expect([...forward.ids]).toEqual(['a', 'b']);
    expect(forward.anchor).toBe('a');

    const reverse = selectGraphMarquee(createGraphSelection(), selectable, {
      x: 60,
      y: 60,
      width: -50,
      height: -50,
    });
    expect([...reverse.ids]).toEqual(['a', 'b']);
  });

  it('supports additive marquee selection while preserving the existing anchor', () => {
    const initial = createGraphSelection(['existing'], 'existing');
    const selection = selectGraphMarquee(
      initial,
      [{ id: 'new', bounds: { x: 5, y: 5, width: 10, height: 10 } }],
      { x: 0, y: 0, width: 20, height: 20 },
      true,
    );
    expect([...selection.ids]).toEqual(['existing', 'new']);
    expect(selection.anchor).toBe('existing');
  });

  it('navigates selection deterministically in domain-provided order', () => {
    const orderedIds = ['a', 'b', 'c'];
    let selection = navigateGraphSelection(createGraphSelection(), orderedIds, 'next');
    expect([...selection.ids]).toEqual(['a']);

    selection = navigateGraphSelection(selection, orderedIds, 'next');
    expect([...selection.ids]).toEqual(['b']);
    expect(navigateGraphSelection(selection, orderedIds, 'previous').anchor).toBe('a');
    expect(navigateGraphSelection(selection, orderedIds, 'last').anchor).toBe('c');
    expect(navigateGraphSelection(selection, orderedIds, 'first').anchor).toBe('a');
  });

  it('extends keyboard navigation as one contiguous selection from a stable anchor', () => {
    const orderedIds = ['a', 'b', 'c', 'd'];
    const initial = createGraphSelection(['b'], 'b');
    const extended = navigateGraphSelection(initial, orderedIds, 'last', true);
    expect([...extended.ids]).toEqual(['b', 'c', 'd']);
    expect(extended.anchor).toBe('b');

    const contracted = navigateGraphSelection(extended, orderedIds, 'next', true);
    expect([...contracted.ids]).toEqual(['b', 'c']);
    expect(contracted.anchor).toBe('b');
  });

  it('handles empty and stale selection anchors without wrapping unexpectedly', () => {
    const empty = createGraphSelection();
    expect(navigateGraphSelection(empty, [], 'next')).toBe(empty);
    expect(navigateGraphSelection(empty, ['a', 'b'], 'previous').anchor).toBe('b');

    const stale = createGraphSelection(['removed'], 'removed');
    expect(navigateGraphSelection(stale, ['a', 'b'], 'next').anchor).toBe('a');
  });

  it('copies only selected nodes while preserving source order', () => {
    const nodes = [
      { id: 'a', value: { x: 1 } },
      { id: 'b', value: { x: 2 } },
      { id: 'c', value: { x: 3 } },
    ];
    const snapshot = createGraphClipboardSnapshot(nodes, createGraphSelection(['c', 'a']));
    expect(snapshot.nodes.map((node) => node.id)).toEqual(['a', 'c']);
  });

  it('requires pasted nodes to receive unique stable ids', () => {
    const snapshot = createGraphClipboardSnapshot(
      [
        { id: 'a', value: 1 },
        { id: 'b', value: 2 },
      ],
      createGraphSelection(['a', 'b']),
    );
    expect(remapGraphClipboardSnapshot(snapshot, (id) => `copy-${id}`).nodes.map((node) => node.id)).toEqual([
      'copy-a',
      'copy-b',
    ]);
    expect(() => remapGraphClipboardSnapshot(snapshot, () => 'duplicate')).toThrow(/Duplicate graph node id/);
  });
});

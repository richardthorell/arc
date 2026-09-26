import { describe, expect, it } from 'vitest';

import {
  createGraphClipboardSnapshot,
  createGraphSelection,
  remapGraphClipboardSnapshot,
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

  it('selects a contiguous range from the stable anchor', () => {
    const initial = selectGraphNode(createGraphSelection(), 'b');
    const selection = selectGraphRange(initial, ['a', 'b', 'c', 'd'], 'd');
    expect([...selection.ids]).toEqual(['b', 'c', 'd']);
    expect(selection.anchor).toBe('b');
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

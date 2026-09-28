import { describe, expect, it } from 'vitest';

import {
  copyGraphSelection,
  deleteGraphSelection,
  duplicateGraphSelection,
  pasteGraphSelection,
  type GraphIdFactory,
} from './graphClipboard';

type Node = { id: string; position: { x: number; y: number }; label: string };
type Edge = { id: string; sourceNodeId: string; targetNodeId: string; pin: string };

const nodes: Node[] = [
  { id: 'a', position: { x: 10, y: 20 }, label: 'A' },
  { id: 'b', position: { x: 50, y: 80 }, label: 'B' },
  { id: 'c', position: { x: 90, y: 120 }, label: 'C' },
];
const edges: Edge[] = [
  { id: 'ab', sourceNodeId: 'a', targetNodeId: 'b', pin: 'value' },
  { id: 'bc', sourceNodeId: 'b', targetNodeId: 'c', pin: 'value' },
];

function idFactory(): GraphIdFactory {
  let node = 0;
  let edge = 0;
  return (kind) => (kind === 'node' ? `node-${++node}` : `edge-${++edge}`);
}

describe('graph clipboard operations', () => {
  it('copies selected nodes and only connections wholly inside the selection', () => {
    const copied = copyGraphSelection(nodes, edges, new Set(['a', 'b']));

    expect(copied.nodes.map((node) => node.id)).toEqual(['a', 'b']);
    expect(copied.edges.map((edge) => edge.id)).toEqual(['ab']);
    expect(copied.nodes[0]).not.toBe(nodes[0]);
  });

  it('pastes with fresh stable IDs, remapped edges, and a predictable offset', () => {
    const copied = copyGraphSelection(nodes, edges, new Set(['a', 'b']));
    const pasted = pasteGraphSelection(copied, idFactory());

    expect(pasted.nodes).toEqual([
      { id: 'node-1', position: { x: 34, y: 44 }, label: 'A' },
      { id: 'node-2', position: { x: 74, y: 104 }, label: 'B' },
    ]);
    expect(pasted.edges).toEqual([
      { id: 'edge-1', sourceNodeId: 'node-1', targetNodeId: 'node-2', pin: 'value' },
    ]);
    expect([...pasted.selectedNodeIds]).toEqual(['node-1', 'node-2']);
    expect(pasted.nodeIdMap.get('a')).toBe('node-1');
    expect(pasted.nodeIdMap.get('b')).toBe('node-2');
  });

  it('duplicates through the same copy/paste contract', () => {
    const duplicated = duplicateGraphSelection(nodes, edges, new Set(['b', 'c']), idFactory(), { x: 8, y: 12 });

    expect(duplicated.nodes.map((node) => [node.id, node.position])).toEqual([
      ['node-1', { x: 58, y: 92 }],
      ['node-2', { x: 98, y: 132 }],
    ]);
    expect(duplicated.edges[0]).toMatchObject({ sourceNodeId: 'node-1', targetNodeId: 'node-2' });
  });

  it('deletes selected nodes and every incident connection without mutating inputs', () => {
    const remaining = deleteGraphSelection(nodes, edges, new Set(['b']));

    expect(remaining.nodes.map((node) => node.id)).toEqual(['a', 'c']);
    expect(remaining.edges).toEqual([]);
    expect(nodes).toHaveLength(3);
    expect(edges).toHaveLength(2);
  });
});

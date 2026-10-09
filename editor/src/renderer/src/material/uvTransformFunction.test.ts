import fs from 'node:fs';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

type FunctionNode = { id: string; type: string; values: Record<string, unknown> };
type FunctionConnection = {
  from: { nodeId: string; pin: string };
  to: { nodeId: string; pin: string };
};
type MaterialFunction = {
  kind: string;
  name: string;
  inputs: Array<{ id: string; type: string; default: number | number[] }>;
  outputs: Array<{ id: string; type: string }>;
  graph: { nodes: FunctionNode[]; connections: FunctionConnection[] };
};

const functionPath = path.resolve(process.cwd(), '..', 'assets', 'material_functions', 'uv_transform.arcmatfn');
const loadFunction = () => JSON.parse(fs.readFileSync(functionPath, 'utf8')) as MaterialFunction;

describe('built-in UV Transform material function', () => {
  it('has identity defaults and a typed UV output', () => {
    const fn = loadFunction();
    expect(fn.kind).toBe('materialFunction');
    expect(fn.name).toBe('UV Transform');
    expect(fn.inputs).toEqual([
      { id: 'tiling', name: 'Tiling', type: 'vec2', default: [1, 1] },
      { id: 'offset', name: 'Offset', type: 'vec2', default: [0, 0] },
      { id: 'pivot', name: 'Pivot', type: 'vec2', default: [0.5, 0.5] },
      { id: 'rotation', name: 'Rotation (radians)', type: 'float', default: 0 },
    ]);
    expect(fn.outputs).toEqual([{ id: 'uv', name: 'UV', type: 'vec2' }]);
  });

  it('connects a complete tiling, pivot, rotation and translation graph', () => {
    const { graph } = loadFunction();
    const nodes = new Map(graph.nodes.map((node) => [node.id, node]));
    const input = new Map(
      graph.connections.map((connection) => [
        `${connection.to.nodeId}.${connection.to.pin}`,
        connection.from.nodeId,
      ]),
    );

    expect(nodes.size).toBe(graph.nodes.length);
    for (const connection of graph.connections) {
      expect(nodes.has(connection.from.nodeId)).toBe(true);
      expect(nodes.has(connection.to.nodeId)).toBe(true);
    }
    expect(nodes.get('texcoord')?.type).toBe('texCoord');
    expect(nodes.get('cos')?.type).toBe('cosine');
    expect(nodes.get('sin')?.type).toBe('sine');
    expect(input.get('tiled.a')).toBe('texcoord');
    expect(input.get('tiled.b')).toBe('input-tiling');
    expect(input.get('centered.b')).toBe('input-pivot');
    expect(input.get('restored-pivot.b')).toBe('input-pivot');
    expect(input.get('translated.b')).toBe('input-offset');
    expect(input.get('function-output.uv')).toBe('translated');
    expect(input.get('rotated-x.a')).toBe('xcos');
    expect(input.get('rotated-x.b')).toBe('ysin');
    expect(input.get('rotated-y.a')).toBe('xsin');
    expect(input.get('rotated-y.b')).toBe('ycos');

    const reachable = new Set<string>();
    const pending = ['function-output'];
    while (pending.length > 0) {
      const id = pending.pop()!;
      if (reachable.has(id)) continue;
      reachable.add(id);
      pending.push(...graph.connections.filter((connection) => connection.to.nodeId === id).map((c) => c.from.nodeId));
    }
    expect(reachable.size).toBe(nodes.size);
  });
});

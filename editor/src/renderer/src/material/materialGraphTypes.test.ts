import { describe, expect, it } from 'vitest';

import {
  createDefaultMaterialGraph,
  isMaterialGraph,
  materialGraphFromAsset,
  materialNodeDefinitions,
} from './materialGraphTypes';

describe('material graph schema', () => {
  it('creates an editable starter graph', () => {
    const graph = createDefaultMaterialGraph();

    expect(graph.version).toBe(1);
    expect(graph.nodes.find((node) => node.type === 'output')?.id).toBe('material-output');
    expect(graph.nodes.filter((node) => node.parameter?.exposed).map((node) => node.parameter?.name)).toEqual([
      'Base Color',
      'Metallic',
      'Roughness',
    ]);
    expect(graph.connections).toHaveLength(3);
  });

  it('defines texture sample UV input and channel outputs', () => {
    expect(materialNodeDefinitions.textureSample.inputs.map((pin) => [pin.id, pin.type])).toEqual([['uv', 'vec2']]);
    expect(materialNodeDefinitions.textureSample.outputs.map((pin) => [pin.id, pin.type])).toEqual([
      ['rgb', 'vec3'],
      ['r', 'float'],
      ['g', 'float'],
      ['b', 'float'],
      ['a', 'float'],
      ['rgba', 'vec4'],
    ]);
  });

  it('recognizes and reuses the stable graph stored in a material asset', () => {
    const stored = createDefaultMaterialGraph();
    stored.viewport = { x: 120, y: 80, zoom: 0.8 };

    expect(isMaterialGraph(stored)).toBe(true);
    expect(materialGraphFromAsset({ graph: stored })).toEqual(stored);
    expect(materialGraphFromAsset({ graph: stored })).not.toBe(stored);
  });

  it('rejects material assets without a native graph', () => {
    expect(() => materialGraphFromAsset({ graph: null })).toThrow('valid native material graph');
  });
});

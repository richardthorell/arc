import { describe, expect, it } from 'vitest';

import { createDefaultMaterialGraph, isMaterialGraph, materialGraphFromAsset } from './materialGraphTypes';

describe('material graph schema', () => {
  it('upgrades a legacy descriptor material into an editable starter graph', () => {
    const graph = createDefaultMaterialGraph({
      surface: {
        baseColor: { r: 0.1, g: 0.2, b: 0.3, a: 1 },
        metallic: 0.4,
        roughness: 0.6,
      },
      graph: null,
    });

    expect(graph.version).toBe(1);
    expect(graph.nodes.find((node) => node.type === 'output')?.id).toBe('material-output');
    expect(graph.nodes.filter((node) => node.parameter?.exposed).map((node) => node.parameter?.name)).toEqual([
      'Base Color',
      'Metallic',
      'Roughness',
    ]);
    expect(graph.connections).toHaveLength(3);
  });

  it('recognizes and reuses the stable graph stored in a material asset', () => {
    const stored = createDefaultMaterialGraph({});
    stored.viewport = { x: 120, y: 80, zoom: 0.8 };

    expect(isMaterialGraph(stored)).toBe(true);
    expect(materialGraphFromAsset({ graph: stored })).toEqual(stored);
    expect(materialGraphFromAsset({ graph: stored })).not.toBe(stored);
  });
});

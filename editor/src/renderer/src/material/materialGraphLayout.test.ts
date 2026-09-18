import { describe, expect, it } from 'vitest';

import {
  autoArrangeMaterialGraph,
  frameMaterialGraphViewport,
  materialGraphBounds,
  materialNodeHeight,
  materialNodeWidth,
  snapMaterialGraphPoint,
} from './materialGraphLayout';
import { createDefaultMaterialGraph, createMaterialNode, type MaterialGraph } from './materialGraphTypes';

const branchedGraph = (): MaterialGraph => {
  const upper = { ...createMaterialNode('constant', [460, 420]), id: 'upper' };
  const lower = { ...createMaterialNode('constant', [-140, -60]), id: 'lower' };
  const multiply = { ...createMaterialNode('multiply', [20, 780]), id: 'multiply' };
  const output = { ...createMaterialNode('output', [-600, 300]), id: 'material-output' };
  return {
    version: 1,
    nodes: [upper, lower, multiply, output],
    connections: [
      {
        id: 'upper-multiply',
        from: { nodeId: upper.id, pin: 'value' },
        to: { nodeId: multiply.id, pin: 'a' },
      },
      {
        id: 'lower-multiply',
        from: { nodeId: lower.id, pin: 'value' },
        to: { nodeId: multiply.id, pin: 'b' },
      },
      {
        id: 'multiply-output',
        from: { nodeId: multiply.id, pin: 'result' },
        to: { nodeId: output.id, pin: 'roughness' },
      },
    ],
    viewport: { x: 0, y: 0, zoom: 1 },
  };
};

const fanInGraph = (connectionCount: number): MaterialGraph => {
  const source = { ...createMaterialNode('textureSample2D', [0, 0]), id: 'source' };
  const output = { ...createMaterialNode('output', [0, 0]), id: 'material-output' };
  const outputPins = [
    'baseColor',
    'metallic',
    'roughness',
    'normal',
    'ao',
    'emissive',
    'opacity',
    'clearCoat',
  ];
  return {
    version: 1,
    nodes: [source, output],
    connections: Array.from({ length: connectionCount }, (_, index) => ({
      id: `connection-${index}`,
      from: { nodeId: source.id, pin: 'rgb' },
      to: { nodeId: output.id, pin: outputPins[index % outputPins.length] },
    })),
    viewport: { x: 0, y: 0, zoom: 1 },
  };
};

describe('material graph layout', () => {
  it('keeps simple nodes compact and widens controls that need more editing space', () => {
    expect(materialNodeWidth('constant')).toBe(214);
    expect(materialNodeWidth('vector2')).toBeGreaterThan(materialNodeWidth('constant'));
    expect(materialNodeWidth('vector3')).toBeGreaterThan(materialNodeWidth('vector2'));
    expect(materialNodeWidth('vector4')).toBeGreaterThan(materialNodeWidth('vector3'));
    expect(materialNodeWidth('textureSample')).toBeGreaterThan(materialNodeWidth('constant'));
    expect(materialNodeWidth('colorRgba')).toBeGreaterThan(materialNodeWidth('textureSample'));
    expect(materialNodeWidth('output')).toBeGreaterThan(materialNodeWidth('constant'));
  });

  it('snaps node positions to the graph grid', () => {
    expect(snapMaterialGraphPoint([31, 49])).toEqual([40, 40]);
    expect(snapMaterialGraphPoint([-11, 11])).toEqual([-20, 20]);
  });

  it('frames every node while keeping small graphs at or below 100% zoom', () => {
    const graph = createDefaultMaterialGraph();
    const viewport = frameMaterialGraphViewport(graph, 1000, 700);
    const bounds = materialGraphBounds(graph);
    expect(bounds).not.toBeNull();
    expect(viewport.zoom).toBeLessThanOrEqual(1);

    const left = viewport.x + bounds!.left * viewport.zoom;
    const top = viewport.y + bounds!.top * viewport.zoom;
    const right = viewport.x + bounds!.right * viewport.zoom;
    const bottom = viewport.y + bounds!.bottom * viewport.zoom;
    expect(left).toBeGreaterThanOrEqual(0);
    expect(top).toBeGreaterThanOrEqual(0);
    expect(right).toBeLessThanOrEqual(1000);
    expect(bottom).toBeLessThanOrEqual(700);
  });

  it('auto-arranges connected nodes into flow columns without overlap', () => {
    const arranged = autoArrangeMaterialGraph(branchedGraph());
    const byId = new Map(arranged.nodes.map((node) => [node.id, node]));
    const upper = byId.get('upper')!;
    const lower = byId.get('lower')!;
    const multiply = byId.get('multiply')!;
    const output = byId.get('material-output')!;

    expect(upper.position[0]).toBeLessThan(multiply.position[0]);
    expect(lower.position[0]).toBeLessThan(multiply.position[0]);
    expect(multiply.position[0]).toBeLessThan(output.position[0]);
    expect(upper.position[0]).toBe(lower.position[0]);

    const [first, second] = [upper, lower].sort((left, right) => left.position[1] - right.position[1]);
    expect(first.position[1] + materialNodeHeight(first)).toBeLessThan(second.position[1]);

    for (const node of arranged.nodes) {
      expect(node.position[0] % 20).toBe(0);
      expect(node.position[1] % 20).toBe(0);
    }
  });

  it('adds horizontal routing room as fan-in pressure increases', () => {
    const connectionGap = (graph: MaterialGraph) => {
      const arranged = autoArrangeMaterialGraph(graph);
      const source = arranged.nodes.find((node) => node.id === 'source')!;
      const output = arranged.nodes.find((node) => node.id === 'material-output')!;
      return output.position[0] - (source.position[0] + materialNodeWidth(source.type));
    };

    const sparseGap = connectionGap(fanInGraph(1));
    const denseGap = connectionGap(fanInGraph(8));
    expect(denseGap).toBeGreaterThan(sparseGap + 100);
  });
});

import { describe, expect, it } from 'vitest';

import {
  autoArrangeMaterialGraph,
  frameMaterialGraphViewport,
  materialGraphBounds,
  materialNodeHeight,
  materialNodePinOffsetY,
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

const textureMaterialGraph = (): MaterialGraph => {
  const texCoord = { ...createMaterialNode('texCoord', [0, 0]), id: 'tex-coord' };
  const baseColor = { ...createMaterialNode('textureSample2D', [0, 0]), id: 'base-color' };
  const packed = { ...createMaterialNode('textureSample2D', [0, 0]), id: 'packed' };
  const normalTexture = { ...createMaterialNode('textureSample2D', [0, 0]), id: 'normal-texture' };
  const normalMap = { ...createMaterialNode('normalMap', [0, 0]), id: 'normal-map' };
  const output = { ...createMaterialNode('output', [0, 0]), id: 'material-output' };
  return {
    version: 1,
    nodes: [texCoord, baseColor, packed, normalTexture, normalMap, output],
    connections: [
      {
        id: 'tex-coord-base-color',
        from: { nodeId: texCoord.id, pin: 'uv' },
        to: { nodeId: baseColor.id, pin: 'uv' },
      },
      {
        id: 'tex-coord-packed',
        from: { nodeId: texCoord.id, pin: 'uv' },
        to: { nodeId: packed.id, pin: 'uv' },
      },
      {
        id: 'tex-coord-normal',
        from: { nodeId: texCoord.id, pin: 'uv' },
        to: { nodeId: normalTexture.id, pin: 'uv' },
      },
      {
        id: 'base-color-output',
        from: { nodeId: baseColor.id, pin: 'rgb' },
        to: { nodeId: output.id, pin: 'baseColor' },
      },
      {
        id: 'packed-metallic',
        from: { nodeId: packed.id, pin: 'r' },
        to: { nodeId: output.id, pin: 'metallic' },
      },
      {
        id: 'packed-roughness',
        from: { nodeId: packed.id, pin: 'g' },
        to: { nodeId: output.id, pin: 'roughness' },
      },
      {
        id: 'normal-texture-map',
        from: { nodeId: normalTexture.id, pin: 'rgb' },
        to: { nodeId: normalMap.id, pin: 'texture' },
      },
      {
        id: 'normal-map-output',
        from: { nodeId: normalMap.id, pin: 'normal' },
        to: { nodeId: output.id, pin: 'normal' },
      },
    ],
    viewport: { x: 0, y: 0, zoom: 1 },
  };
};

const singleNormalChainGraph = (): MaterialGraph => {
  const texture = { ...createMaterialNode('textureSample2D', [0, 0]), id: 'normal-texture' };
  const normalMap = { ...createMaterialNode('normalMap', [0, 0]), id: 'normal-map' };
  const output = { ...createMaterialNode('output', [0, 0]), id: 'material-output' };
  return {
    version: 1,
    nodes: [texture, normalMap, output],
    connections: [
      {
        id: 'texture-normal-map',
        from: { nodeId: texture.id, pin: 'rgb' },
        to: { nodeId: normalMap.id, pin: 'texture' },
      },
      {
        id: 'normal-map-output',
        from: { nodeId: normalMap.id, pin: 'normal' },
        to: { nodeId: output.id, pin: 'normal' },
      },
    ],
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
      expect(Number.isInteger(node.position[1])).toBe(true);
    }
  });

  it('keeps peer source node types aligned even when one branch has an extra processor', () => {
    const arranged = autoArrangeMaterialGraph(textureMaterialGraph());
    const byId = new Map(arranged.nodes.map((node) => [node.id, node]));
    const texCoord = byId.get('tex-coord')!;
    const baseColor = byId.get('base-color')!;
    const packed = byId.get('packed')!;
    const normalTexture = byId.get('normal-texture')!;
    const normalMap = byId.get('normal-map')!;
    const output = byId.get('material-output')!;

    expect(texCoord.position[0]).toBeLessThan(baseColor.position[0]);
    expect(baseColor.position[0]).toBe(packed.position[0]);
    expect(baseColor.position[0]).toBe(normalTexture.position[0]);
    expect(normalTexture.position[0]).toBeLessThan(normalMap.position[0]);
    expect(normalMap.position[0]).toBeLessThan(output.position[0]);
  });

  it('makes unconstrained single-input/output chains straight pin-to-pin', () => {
    const arranged = autoArrangeMaterialGraph(singleNormalChainGraph());
    const byId = new Map(arranged.nodes.map((node) => [node.id, node]));
    const normalTexture = byId.get('normal-texture')!;
    const normalMap = byId.get('normal-map')!;
    const output = byId.get('material-output')!;

    const normalTextureOut =
      normalTexture.position[1] + materialNodePinOffsetY(normalTexture, 'rgb', 'output');
    const normalMapIn = normalMap.position[1] + materialNodePinOffsetY(normalMap, 'texture', 'input');
    const normalMapOut = normalMap.position[1] + materialNodePinOffsetY(normalMap, 'normal', 'output');
    const outputNormal = output.position[1] + materialNodePinOffsetY(output, 'normal', 'input');

    expect(normalTextureOut).toBeCloseTo(normalMapIn, 6);
    expect(normalMapOut).toBeCloseTo(outputNormal, 6);
  });

  it('places an intermediate single-input/output node between crowded endpoints', () => {
    const arranged = autoArrangeMaterialGraph(textureMaterialGraph());
    const byId = new Map(arranged.nodes.map((node) => [node.id, node]));
    const normalTexture = byId.get('normal-texture')!;
    const normalMap = byId.get('normal-map')!;
    const output = byId.get('material-output')!;

    const sourceY = normalTexture.position[1] + materialNodePinOffsetY(normalTexture, 'rgb', 'output');
    const nodeY = normalMap.position[1] + materialNodePinOffsetY(normalMap, 'normal', 'output');
    const targetY = output.position[1] + materialNodePinOffsetY(output, 'normal', 'input');

    expect(nodeY).toBeGreaterThanOrEqual(Math.min(sourceY, targetY));
    expect(nodeY).toBeLessThanOrEqual(Math.max(sourceY, targetY));
  });

  it('centers a fan-out source between the input pins it feeds', () => {
    const arranged = autoArrangeMaterialGraph(textureMaterialGraph());
    const byId = new Map(arranged.nodes.map((node) => [node.id, node]));
    const texCoord = byId.get('tex-coord')!;
    const targets = ['base-color', 'packed', 'normal-texture'].map((id) => byId.get(id)!);

    const sourceY = texCoord.position[1] + materialNodePinOffsetY(texCoord, 'uv', 'output');
    const targetYs = targets.map(
      (target) => target.position[1] + materialNodePinOffsetY(target, 'uv', 'input'),
    );
    const averageTargetY = targetYs.reduce((sum, value) => sum + value, 0) / targetYs.length;

    expect(sourceY).toBeCloseTo(averageTargetY, 0);
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

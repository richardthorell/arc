import { describe, expect, it } from 'vitest';

import { compileMaterialGraph } from './materialCompiler';
import { createDefaultMaterialGraph, createMaterialNode, type MaterialGraph } from './materialGraphTypes';

describe('material graph compiler', () => {
  it('lowers the default legacy PBR graph to descriptor-compatible surface values', () => {
    const graph = createDefaultMaterialGraph({
      surface: {
        baseColor: { r: 0.2, g: 0.4, b: 0.7, a: 1 },
        metallic: 0.35,
        roughness: 0.72,
      },
    });

    const result = compileMaterialGraph(graph);

    expect(result.succeeded).toBe(true);
    expect(result.surface.baseColor).toEqual({ r: 0.2, g: 0.4, b: 0.7, a: 1 });
    expect(result.surface.metallic).toBeCloseTo(0.35);
    expect(result.surface.roughness).toBeCloseTo(0.72);
    expect(result.ir.parameters.map((parameter) => parameter.name)).toEqual(['Base Color', 'Metallic', 'Roughness']);
  });

  it('constant folds math expressions before lowering the graph', () => {
    const left = createMaterialNode('vector3', [0, 0], { value: [0.4, 0.5, 0.6] });
    const right = createMaterialNode('constant', [0, 160], { value: 0.5 });
    const multiply = createMaterialNode('multiply', [260, 70]);
    const output = createMaterialNode('output', [520, 70]);
    const graph: MaterialGraph = {
      version: 1,
      nodes: [left, right, multiply, output],
      connections: [
        { id: 'a', from: { nodeId: left.id, pin: 'value' }, to: { nodeId: multiply.id, pin: 'a' } },
        { id: 'b', from: { nodeId: right.id, pin: 'value' }, to: { nodeId: multiply.id, pin: 'b' } },
        { id: 'out', from: { nodeId: multiply.id, pin: 'result' }, to: { nodeId: output.id, pin: 'baseColor' } },
      ],
    };

    const result = compileMaterialGraph(graph);

    expect(result.succeeded).toBe(true);
    expect(result.surface.baseColor).toEqual({ r: 0.2, g: 0.25, b: 0.3, a: 1 });
    expect(result.ir.expressions.some((expression) => expression.operation === 'multiply')).toBe(true);
  });

  it('routes a direct texture sample into the existing material descriptor backend', () => {
    const texture = createMaterialNode('textureSample', [0, 0], { texture: 'Textures/rock_basecolor.png' });
    const output = createMaterialNode('output', [420, 0]);
    const graph: MaterialGraph = {
      version: 1,
      nodes: [texture, output],
      connections: [
        { id: 'texture-out', from: { nodeId: texture.id, pin: 'rgb' }, to: { nodeId: output.id, pin: 'baseColor' } },
      ],
    };

    const result = compileMaterialGraph(graph);

    expect(result.succeeded).toBe(true);
    expect(result.textures.baseColor).toBe('Textures/rock_basecolor.png');
    expect(result.surface.baseColor).toEqual({ r: 1, g: 1, b: 1, a: 1 });
  });

  it('reports cycles instead of recursively compiling forever', () => {
    const first = createMaterialNode('add', [0, 0]);
    const second = createMaterialNode('multiply', [240, 0]);
    const output = createMaterialNode('output', [500, 0]);
    const graph: MaterialGraph = {
      version: 1,
      nodes: [first, second, output],
      connections: [
        { id: 'one', from: { nodeId: first.id, pin: 'result' }, to: { nodeId: second.id, pin: 'a' } },
        { id: 'two', from: { nodeId: second.id, pin: 'result' }, to: { nodeId: first.id, pin: 'a' } },
        { id: 'out', from: { nodeId: first.id, pin: 'result' }, to: { nodeId: output.id, pin: 'baseColor' } },
      ],
    };

    const result = compileMaterialGraph(graph);

    expect(result.succeeded).toBe(false);
    expect(result.diagnostics.some((diagnostic) => diagnostic.message.includes('cycle'))).toBe(true);
  });
});

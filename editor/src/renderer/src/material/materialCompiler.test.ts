import { describe, expect, it } from 'vitest';

import { materialEditorParameters, materialGraphEditImpact, nativeMaterialCompileResult } from './materialCompiler';
import { createDefaultMaterialGraph, createMaterialNode } from './materialGraphTypes';

describe('native material compiler editor adapter', () => {
  it('maps native diagnostics without performing local graph validation', () => {
    const result = nativeMaterialCompileResult(true, {
      succeeded: false,
      message: 'Material graph validation failed',
      diagnostics: [
        {
          severity: 'error',
          code: 'material.cycle',
          message: 'Material graph contains a cycle',
          graphNode: 'multiply-1',
          line: 14,
        },
      ],
    });

    expect(result.succeeded).toBe(false);
    expect(result.status).toBe('failed');
    expect(result.diagnostics[0]).toMatchObject({
      severity: 'error',
      code: 'material.cycle',
      nodeId: 'multiply-1',
      line: 14,
    });
  });

  it('derives exposed parameter presentation metadata without creating editor IR', () => {
    const graph = createDefaultMaterialGraph();
    expect(materialEditorParameters(graph).map((parameter) => parameter.name)).toEqual([
      'Base Color',
      'Metallic',
      'Roughness',
    ]);
  });

  it('exposes Texture Sample parameters as texture2d values', () => {
    const graph = createDefaultMaterialGraph();
    const texture = createMaterialNode('textureSample', [160, 160], { texture: 'Content/Textures/albedo.png' });
    texture.parameter = { exposed: true, name: 'Albedo' };
    graph.nodes.push(texture);

    expect(materialEditorParameters(graph)).toContainEqual(
      expect.objectContaining({
        nodeId: texture.id,
        name: 'Albedo',
        type: 'texture2d',
      }),
    );
  });

  it('classifies value-only edits to existing exposed parameters without requiring a shader rebuild', () => {
    const before = createDefaultMaterialGraph();
    const after = structuredClone(before);
    after.nodes[1].values = { ...after.nodes[1].values, value: [0.2, 0.3, 0.4] };

    expect(materialGraphEditImpact(before, after)).toBe('parameter-values');
  });

  it('keeps topology and parameter metadata changes on the shader compile path', () => {
    const before = createDefaultMaterialGraph();
    const topology = structuredClone(before);
    topology.connections = topology.connections.slice(1);
    expect(materialGraphEditImpact(before, topology)).toBe('shader');

    const metadata = structuredClone(before);
    metadata.nodes[1].parameter = { ...metadata.nodes[1].parameter!, name: 'Tint' };
    expect(materialGraphEditImpact(before, metadata)).toBe('shader');
  });

  it('does not classify values on unexposed nodes as runtime parameter edits', () => {
    const before = createDefaultMaterialGraph();
    const node = createMaterialNode('constant', [160, 160], { value: 0.5 });
    before.nodes.push(node);
    const after = structuredClone(before);
    after.nodes.at(-1)!.values.value = 0.75;

    expect(materialGraphEditImpact(before, after)).toBe('shader');
  });
});

import { describe, expect, it } from 'vitest';

import { materialEditorParameters, nativeMaterialCompileResult } from './materialCompiler';
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
});

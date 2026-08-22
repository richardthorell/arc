import { describe, expect, it } from 'vitest';

import {
  materialEditorParameters,
  nativeMaterialCompileResult,
  projectLegacyMaterialPreview,
} from './materialCompiler';
import { createDefaultMaterialGraph, createMaterialNode, type MaterialGraph } from './materialGraphTypes';

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

  it('keeps a legacy preview projection until compiled materials become the renderer default', () => {
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

    const projection = projectLegacyMaterialPreview(graph);
    expect(projection.surface.baseColor).toEqual({ r: 0.2, g: 0.25, b: 0.3, a: 1 });
  });
});

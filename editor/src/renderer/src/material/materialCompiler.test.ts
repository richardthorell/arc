import { describe, expect, it } from 'vitest';

import { materialEditorParameters, nativeMaterialCompileResult } from './materialCompiler';
import { createDefaultMaterialGraph } from './materialGraphTypes';

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
});

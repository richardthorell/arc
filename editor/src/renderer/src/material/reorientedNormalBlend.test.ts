import fs from 'node:fs';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

type ShaderFunction = {
  kind: string;
  inputs: Array<{ id: string; type: string; default?: number }>;
  outputs: Array<{ id: string; type: string }>;
  shader: { entryPoint: string; source: string };
};

const read = (): ShaderFunction =>
  JSON.parse(
    fs.readFileSync(
      path.resolve(process.cwd(), '..', 'assets', 'material_functions', 'reoriented_normal_blend.arcmatfn'),
      'utf8',
    ),
  ) as ShaderFunction;

describe('reoriented tangent-space normal blending', () => {
  it('keeps signed tangent-normal inputs distinct from world-space normal functions', () => {
    const fn = read();
    expect(fn.kind).toBe('materialFunction');
    expect(fn.inputs).toEqual([
      { id: 'base', name: 'Base Tangent Normal', type: 'vec3' },
      { id: 'detail', name: 'Detail Tangent Normal', type: 'vec3' },
      { id: 'weight', name: 'Detail Weight', type: 'float', default: 0 },
    ]);
    expect(fn.outputs).toEqual([{ id: 'normal', name: 'Tangent Normal', type: 'vec3' }]);
    expect(fn.shader.source).toContain('saturate(arc_input_weight)');
    expect(fn.shader.source).toContain('dot(t, u)');
    expect(fn.shader.source).toContain('arc_output_normal = normalize(combined)');
  });
});

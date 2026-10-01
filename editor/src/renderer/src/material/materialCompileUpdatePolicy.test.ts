import { describe, expect, it } from 'vitest';

import { materialCompileUpdateAction, materialEditRequiresShaderCompile } from './materialCompileUpdatePolicy';
import { cloneMaterialGraph, createDefaultMaterialGraph } from './materialGraphTypes';

describe('material compile update policy', () => {
  it('keeps exposed parameter value edits off the shader compile path', () => {
    const before = createDefaultMaterialGraph();
    const after = cloneMaterialGraph(before);
    const parameter = after.nodes.find((node) => node.parameter?.exposed);
    expect(parameter).toBeDefined();
    parameter!.values = { ...parameter!.values, value: [0.2, 0.3, 0.4] };

    expect(materialCompileUpdateAction(before, after)).toBe('parameter-update');
    expect(materialEditRequiresShaderCompile(before, after)).toBe(false);
  });

  it('keeps topology edits on the authoritative native shader compile path', () => {
    const before = createDefaultMaterialGraph();
    const after = cloneMaterialGraph(before);
    after.connections = after.connections.slice(1);

    expect(materialCompileUpdateAction(before, after)).toBe('shader-compile');
    expect(materialEditRequiresShaderCompile(before, after)).toBe(true);
  });

  it('does nothing for layout-only edits', () => {
    const before = createDefaultMaterialGraph();
    const after = cloneMaterialGraph(before);
    after.nodes[0].position = [after.nodes[0].position[0] + 64, after.nodes[0].position[1] + 32];

    expect(materialCompileUpdateAction(before, after)).toBe('none');
    expect(materialEditRequiresShaderCompile(before, after)).toBe(false);
  });
});

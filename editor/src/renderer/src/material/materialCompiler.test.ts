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
      'Base Color Tint',
      'Base Color Texture',
      'Metallic',
      'Roughness',
      'Metallic Roughness Texture',
      'Ambient Occlusion Texture',
      'Normal Texture',
      'Clear Coat',
      'Clear Coat Roughness',
      'Clear Coat Texture',
      'Emissive Color',
      'Emissive Texture',
      'Emissive Strength',
    ]);
  });

  it('exposes the default Base Color Texture as a texture2d parameter', () => {
    const graph = createDefaultMaterialGraph();
    const texture = graph.nodes.find((node) => node.parameter?.name === 'Base Color Texture');

    expect(texture).toBeDefined();
    expect(materialEditorParameters(graph)).toContainEqual(
      expect.objectContaining({
        nodeId: texture!.id,
        name: 'Base Color Texture',
        type: 'texture2d',
        editorKind: 'texture',
      }),
    );
  });

  it('projects authored Scalar ranges into exposed parameter metadata', () => {
    const graph = createDefaultMaterialGraph();
    const roughness = graph.nodes.find((node) => node.parameter?.name === 'Roughness');
    expect(roughness).toBeDefined();
    roughness!.values = { ...roughness!.values, min: 0, max: 1 };

    expect(materialEditorParameters(graph)).toContainEqual(
      expect.objectContaining({
        nodeId: roughness!.id,
        name: 'Roughness',
        type: 'float',
        editorKind: 'scalar',
        range: { min: 0, max: 1 },
      }),
    );
  });

  it('omits exposed nodes that cannot affect Material Output', () => {
    const graph = createDefaultMaterialGraph();
    const disconnected = createMaterialNode('constant', [160, 160], { value: 0.5 });
    disconnected.parameter = { exposed: true, name: 'Disconnected' };
    const multiply = createMaterialNode('multiply', [260, 160]);
    multiply.parameter = { exposed: true, name: 'Operation' };
    graph.nodes.push(disconnected, multiply);

    expect(materialEditorParameters(graph).map((parameter) => parameter.name)).not.toContain('Disconnected');
    expect(materialEditorParameters(graph).map((parameter) => parameter.name)).not.toContain('Operation');
  });

  it('supports the modern Color node as an RGBA color parameter', () => {
    const graph = createDefaultMaterialGraph();
    expect(materialEditorParameters(graph)).toContainEqual(
      expect.objectContaining({
        name: 'Base Color Tint',
        nodeType: 'colorRgba',
        type: 'vec4',
        editorKind: 'color',
      }),
    );
  });

  it('classifies value-only edits to existing exposed parameters without requiring a shader rebuild', () => {
    const before = createDefaultMaterialGraph();
    const after = structuredClone(before);
    after.nodes[0].values = { ...after.nodes[0].values, value: [0.2, 0.3, 0.4, 1] };

    expect(materialGraphEditImpact(before, after)).toBe('parameter-values');
  });

  it('treats assigning the default Base Color Texture as a parameter-only edit', () => {
    const before = createDefaultMaterialGraph();
    const after = structuredClone(before);
    const texture = after.nodes.find((node) => node.parameter?.name === 'Base Color Texture');
    expect(texture).toBeDefined();
    texture!.values = { ...texture!.values, texture: 'Content/Textures/wall.png' };

    expect(materialGraphEditImpact(before, after)).toBe('parameter-values');
  });

  it('treats assigning the Standard Lit Normal Texture as a parameter-only edit', () => {
    const before = createDefaultMaterialGraph();
    const after = structuredClone(before);
    const texture = after.nodes.find((node) => node.parameter?.name === 'Normal Texture');
    expect(texture).toBeDefined();
    texture!.values = { ...texture!.values, texture: 'Content/Textures/wall_normal.png' };

    expect(materialGraphEditImpact(before, after)).toBe('parameter-values');
    expect(materialEditorParameters(after)).toContainEqual(
      expect.objectContaining({ name: 'Normal Texture', type: 'texture2d', editorKind: 'texture' }),
    );
  });

  it('treats Standard Lit clear coat controls as parameter-only edits', () => {
    const before = createDefaultMaterialGraph();
    const after = structuredClone(before);
    const factor = after.nodes.find((node) => node.parameter?.name === 'Clear Coat');
    const roughness = after.nodes.find((node) => node.parameter?.name === 'Clear Coat Roughness');
    const texture = after.nodes.find((node) => node.parameter?.name === 'Clear Coat Texture');
    expect(factor).toBeDefined();
    expect(roughness).toBeDefined();
    expect(texture).toBeDefined();

    factor!.values = { ...factor!.values, value: 0.8 };
    roughness!.values = { ...roughness!.values, value: 0.15 };
    texture!.values = { ...texture!.values, texture: 'Content/Textures/car_clearcoat.png' };

    expect(materialGraphEditImpact(before, after)).toBe('parameter-values');
  });

  it('treats Standard Lit emissive controls as parameter-only edits', () => {
    const before = createDefaultMaterialGraph();
    const after = structuredClone(before);
    const texture = after.nodes.find((node) => node.parameter?.name === 'Emissive Texture');
    const color = after.nodes.find((node) => node.parameter?.name === 'Emissive Color');
    const strength = after.nodes.find((node) => node.parameter?.name === 'Emissive Strength');
    expect(texture).toBeDefined();
    expect(color).toBeDefined();
    expect(strength).toBeDefined();

    texture!.values = { ...texture!.values, texture: 'Content/Textures/sign_emissive.png' };
    color!.values = { ...color!.values, value: [0.2, 0.8, 1, 1] };
    strength!.values = { ...strength!.values, value: 4 };

    expect(materialGraphEditImpact(before, after)).toBe('parameter-values');
    expect(materialEditorParameters(after)).toContainEqual(
      expect.objectContaining({ name: 'Emissive Texture', type: 'texture2d', editorKind: 'texture' }),
    );
    expect(materialEditorParameters(after)).toContainEqual(
      expect.objectContaining({ name: 'Emissive Color', type: 'vec4', editorKind: 'color' }),
    );
    expect(materialEditorParameters(after)).toContainEqual(
      expect.objectContaining({ name: 'Emissive Strength', type: 'float' }),
    );
  });

  it('recompiles range metadata edits but keeps in-range Scalar value edits parameter-only', () => {
    const before = createDefaultMaterialGraph();
    const roughness = before.nodes.find((node) => node.parameter?.name === 'Roughness');
    expect(roughness).toBeDefined();
    roughness!.values = { ...roughness!.values, min: 0, max: 1 };

    const valueEdit = structuredClone(before);
    valueEdit.nodes.find((node) => node.id === roughness!.id)!.values.value = 0.4;
    expect(materialGraphEditImpact(before, valueEdit)).toBe('parameter-values');

    const rangeEdit = structuredClone(before);
    rangeEdit.nodes.find((node) => node.id === roughness!.id)!.values.max = 2;
    expect(materialGraphEditImpact(before, rangeEdit)).toBe('shader');
  });

  it('keeps topology and parameter metadata changes on the shader compile path', () => {
    const before = createDefaultMaterialGraph();
    const topology = structuredClone(before);
    topology.connections = topology.connections.slice(1);
    expect(materialGraphEditImpact(before, topology)).toBe('shader');

    const metadata = structuredClone(before);
    metadata.nodes[0].parameter = { ...metadata.nodes[0].parameter!, name: 'Tint' };
    expect(materialGraphEditImpact(before, metadata)).toBe('shader');
  });

  it('does not classify values on ineffective exposed nodes as runtime parameter edits', () => {
    const before = createDefaultMaterialGraph();
    const node = createMaterialNode('constant', [160, 160], { value: 0.5 });
    node.parameter = { exposed: true, name: 'Disconnected' };
    before.nodes.push(node);
    const after = structuredClone(before);
    after.nodes.at(-1)!.values.value = 0.75;

    expect(materialGraphEditImpact(before, after)).toBe('shader');
  });
});

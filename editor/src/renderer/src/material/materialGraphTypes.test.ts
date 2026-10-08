import { describe, expect, it } from 'vitest';

import {
  createDefaultMaterialGraph,
  isMaterialGraph,
  materialGraphCompileFingerprint,
  materialGraphFromAsset,
  materialNodeDefinitions,
} from './materialGraphTypes';

describe('material graph schema', () => {
  it('creates an editable starter graph', () => {
    const graph = createDefaultMaterialGraph();

    expect(graph.version).toBe(1);
    expect(graph.nodes.find((node) => node.type === 'output')?.id).toBe('material-output');
    expect(graph.nodes.filter((node) => node.parameter?.exposed).map((node) => node.parameter?.name)).toEqual([
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
    const tint = graph.nodes.find((node) => node.parameter?.name === 'Base Color Tint');
    const texture = graph.nodes.find((node) => node.parameter?.name === 'Base Color Texture');
    const multiply = graph.nodes.find((node) => node.type === 'multiply');
    expect(tint?.type).toBe('colorRgba');
    expect(texture).toMatchObject({ type: 'textureSample2D', values: { texture: '', dimension: '2d' } });
    expect(multiply).toBeDefined();
    expect(
      graph.connections.some(
        (connection) => connection.from.nodeId === texture?.id && connection.to.nodeId === multiply?.id,
      ),
    ).toBe(true);
    expect(
      graph.connections.some(
        (connection) => connection.from.nodeId === tint?.id && connection.to.nodeId === multiply?.id,
      ),
    ).toBe(true);
    expect(
      graph.connections.some(
        (connection) => connection.from.nodeId === multiply?.id && connection.to.nodeId === 'material-output',
      ),
    ).toBe(true);
    const metallicRoughnessTexture = graph.nodes.find((node) => node.parameter?.name === 'Metallic Roughness Texture');
    const ambientOcclusionTexture = graph.nodes.find((node) => node.parameter?.name === 'Ambient Occlusion Texture');
    expect(metallicRoughnessTexture).toMatchObject({
      type: 'textureSample2D',
      values: { texture: '', dimension: '2d' },
    });
    expect(ambientOcclusionTexture).toMatchObject({
      type: 'textureSample2D',
      values: { texture: '', dimension: '2d' },
    });
    expect(
      graph.connections.some(
        (connection) =>
          connection.from.nodeId === metallicRoughnessTexture?.id &&
          connection.from.pin === 'b' &&
          connection.to.nodeId !== 'material-output',
      ),
    ).toBe(true);
    expect(
      graph.connections.some(
        (connection) =>
          connection.from.nodeId === metallicRoughnessTexture?.id &&
          connection.from.pin === 'g' &&
          connection.to.nodeId !== 'material-output',
      ),
    ).toBe(true);
    expect(
      graph.connections.some(
        (connection) =>
          connection.from.nodeId === ambientOcclusionTexture?.id &&
          connection.from.pin === 'r' &&
          connection.to.pin === 'ao',
      ),
    ).toBe(true);

    const normalTexture = graph.nodes.find((node) => node.parameter?.name === 'Normal Texture');
    expect(normalTexture).toMatchObject({
      type: 'textureSample2D',
      values: { texture: '', dimension: '2d', semantic: 'normal' },
    });
    expect(
      graph.connections.some(
        (connection) => connection.from.nodeId === normalTexture?.id && connection.to.nodeId !== 'material-output',
      ),
    ).toBe(true);
    expect(
      graph.connections.some(
        (connection) => connection.to.nodeId === 'material-output' && connection.to.pin === 'normal',
      ),
    ).toBe(true);

    const clearCoat = graph.nodes.find((node) => node.parameter?.name === 'Clear Coat');
    const clearCoatRoughness = graph.nodes.find((node) => node.parameter?.name === 'Clear Coat Roughness');
    const clearCoatTexture = graph.nodes.find((node) => node.parameter?.name === 'Clear Coat Texture');
    expect(clearCoat).toMatchObject({ type: 'constant', values: { value: 0 } });
    expect(clearCoatRoughness).toMatchObject({ type: 'constant', values: { value: 0.1 } });
    expect(clearCoatTexture).toMatchObject({
      type: 'textureSample2D',
      values: { texture: '', dimension: '2d', semantic: 'clear_coat' },
    });
    expect(
      graph.connections.some(
        (connection) => connection.from.nodeId === clearCoatTexture?.id && connection.from.pin === 'r',
      ),
    ).toBe(true);
    expect(
      graph.connections.some(
        (connection) => connection.from.nodeId === clearCoatTexture?.id && connection.from.pin === 'g',
      ),
    ).toBe(true);
    expect(graph.connections.some((connection) => connection.to.pin === 'clearCoat')).toBe(true);
    expect(graph.connections.some((connection) => connection.to.pin === 'clearCoatRoughness')).toBe(true);

    const emissiveColor = graph.nodes.find((node) => node.parameter?.name === 'Emissive Color');
    const emissiveTexture = graph.nodes.find((node) => node.parameter?.name === 'Emissive Texture');
    const emissiveStrength = graph.nodes.find((node) => node.parameter?.name === 'Emissive Strength');
    expect(emissiveColor).toMatchObject({ type: 'colorRgba', values: { value: [1, 1, 1, 1] } });
    expect(emissiveTexture).toMatchObject({ type: 'textureSample2D', values: { texture: '', dimension: '2d' } });
    expect(emissiveStrength).toMatchObject({ type: 'constant', values: { value: 0 } });
    expect(
      graph.connections.some(
        (connection) => connection.from.nodeId === emissiveStrength?.id && connection.to.nodeId !== 'material-output',
      ),
    ).toBe(true);
    expect(
      graph.connections.some(
        (connection) => connection.to.nodeId === 'material-output' && connection.to.pin === 'emissive',
      ),
    ).toBe(true);
    expect(graph.connections).toHaveLength(23);
  });

  it('presents scalar constants as Scalar while keeping the stable serialized type', () => {
    expect(materialNodeDefinitions.constant.title).toBe('Scalar');
  });

  it('accepts valid Scalar ranges and rejects malformed persisted ranges', () => {
    const graph = createDefaultMaterialGraph();
    const scalar = graph.nodes.find((node) => node.type === 'constant');
    expect(scalar).toBeDefined();

    scalar!.values = { ...scalar!.values, value: 0.5, min: 0, max: 1 };
    expect(isMaterialGraph(graph)).toBe(true);

    const inverted = JSON.parse(JSON.stringify(graph));
    inverted.nodes.find((node: { type: string }) => node.type === 'constant').values = {
      value: 0.5,
      min: 1,
      max: 0,
    };
    expect(isMaterialGraph(inverted)).toBe(false);

    const outside = JSON.parse(JSON.stringify(graph));
    outside.nodes.find((node: { type: string }) => node.type === 'constant').values = {
      value: 2,
      min: 0,
      max: 1,
    };
    expect(isMaterialGraph(outside)).toBe(false);
  });

  it('exposes only the modern Color node for authoring', () => {
    expect(materialNodeDefinitions).toHaveProperty('colorRgba');
    expect(materialNodeDefinitions).not.toHaveProperty('colorRgb');
  });

  it('defines normalized semantic ranges on bounded Material Output inputs', () => {
    const output = materialNodeDefinitions.output;
    const rangeFor = (pin: string) => output.inputs.find((candidate) => candidate.id === pin)?.semanticRange;

    for (const pin of [
      'metallic',
      'roughness',
      'ao',
      'opacity',
      'alphaClip',
      'clearCoat',
      'clearCoatRoughness',
      'sheen',
      'sheenRoughness',
      'transmission',
      'subsurface',
    ])
      expect(rangeFor(pin)).toEqual({ min: 0, max: 1 });

    expect(rangeFor('indexOfRefraction')).toBeUndefined();
    expect(rangeFor('thickness')).toBeUndefined();
    expect(rangeFor('attenuationDistance')).toBeUndefined();
  });

  it('defines backend-neutral material input intrinsics', () => {
    expect(materialNodeDefinitions.worldPosition).toMatchObject({
      title: 'World Position',
      category: 'Utility',
      subcategory: 'Coordinates',
      outputs: [{ id: 'position', label: 'Position', type: 'vec3' }],
    });
    expect(materialNodeDefinitions.worldNormal).toMatchObject({
      title: 'World Normal',
      category: 'Utility',
      subcategory: 'Coordinates',
      outputs: [{ id: 'normal', label: 'Normal', type: 'vec3' }],
    });
    expect(materialNodeDefinitions.vertexColor.outputs.map((pin) => [pin.id, pin.type])).toEqual([
      ['rgb', 'vec3'],
      ['r', 'float'],
      ['g', 'float'],
      ['b', 'float'],
      ['a', 'float'],
      ['rgba', 'vec4'],
    ]);
    expect(materialNodeDefinitions.texCoord).toMatchObject({
      title: 'Texture Coordinate',
      outputs: [{ id: 'uv', label: 'UV0', type: 'vec2' }],
      defaultValues: { channel: 0 },
    });
  });

  it('defines texture sample UV input and channel outputs', () => {
    expect(materialNodeDefinitions.textureSample.inputs.map((pin) => [pin.id, pin.type])).toEqual([['uv', 'vec2']]);
    expect(materialNodeDefinitions.textureSample.outputs.map((pin) => [pin.id, pin.type])).toEqual([
      ['rgb', 'vec3'],
      ['r', 'float'],
      ['g', 'float'],
      ['b', 'float'],
      ['a', 'float'],
      ['rgba', 'vec4'],
    ]);
  });

  it('recognizes and reuses the stable graph stored in a material asset', () => {
    const stored = createDefaultMaterialGraph();
    stored.viewport = { x: 120, y: 80, zoom: 0.8 };

    expect(isMaterialGraph(stored)).toBe(true);
    expect(materialGraphFromAsset({ graph: stored })).toEqual(stored);
    expect(materialGraphFromAsset({ graph: stored })).not.toBe(stored);
  });

  it('rejects material assets without a native graph', () => {
    expect(() => materialGraphFromAsset({ graph: null })).toThrow('valid native material graph');
  });
});

describe('material graph compile fingerprint', () => {
  it('ignores node layout and viewport-only edits', () => {
    const graph = createDefaultMaterialGraph();
    const moved = JSON.parse(JSON.stringify(graph));
    moved.nodes[0].position = [900, 700];
    moved.viewport = { x: -250, y: 120, zoom: 1.4 };

    expect(materialGraphCompileFingerprint(moved)).toBe(materialGraphCompileFingerprint(graph));
  });

  it('changes for values, parameters, and graph connections', () => {
    const graph = createDefaultMaterialGraph();

    const valueEdit = JSON.parse(JSON.stringify(graph));
    valueEdit.nodes.find((node: { type: string }) => node.type === 'constant').values.value = 0.25;
    expect(materialGraphCompileFingerprint(valueEdit)).not.toBe(materialGraphCompileFingerprint(graph));

    const parameterEdit = JSON.parse(JSON.stringify(graph));
    parameterEdit.nodes[0].parameter.name = 'Tint';
    expect(materialGraphCompileFingerprint(parameterEdit)).not.toBe(materialGraphCompileFingerprint(graph));

    const connectionEdit = JSON.parse(JSON.stringify(graph));
    connectionEdit.connections.pop();
    expect(materialGraphCompileFingerprint(connectionEdit)).not.toBe(materialGraphCompileFingerprint(graph));
  });
});

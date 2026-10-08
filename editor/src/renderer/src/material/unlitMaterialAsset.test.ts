import fs from 'node:fs';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { materialEditorParameters } from './materialCompiler';
import { materialGraphFromAsset, type MaterialAssetJson } from './materialGraphTypes';
import { materialGraphOutputConnected, materialRenderPathLabel } from './materialSettingsPresentation';

const unlitAssetPath = path.resolve(process.cwd(), '..', 'assets', 'materials', 'unlit.arcmat');

describe('built-in Unlit material', () => {
  it('ships a texture-ready lighting-independent surface', () => {
    const asset = JSON.parse(fs.readFileSync(unlitAssetPath, 'utf8')) as MaterialAssetJson;
    expect(asset).toMatchObject({
      version: 4,
      name: 'Unlit',
      domain: 'surface',
      blendMode: 'opaque',
      shadingModel: 'unlit',
    });

    const graph = materialGraphFromAsset(asset);
    expect(materialEditorParameters(graph).map((parameter) => parameter.name)).toEqual([
      'Color Tint',
      'Color Texture',
      'Alpha Clip',
    ]);
    expect(materialEditorParameters(graph)).toContainEqual(
      expect.objectContaining({ name: 'Color Tint', type: 'vec4', editorKind: 'color' }),
    );
    expect(materialEditorParameters(graph)).toContainEqual(
      expect.objectContaining({ name: 'Color Texture', type: 'texture2d', editorKind: 'texture' }),
    );
    expect(materialEditorParameters(graph)).toContainEqual(
      expect.objectContaining({ name: 'Alpha Clip', type: 'float', range: { min: 0, max: 1 } }),
    );

    const tint = graph.nodes.find((node) => node.parameter?.name === 'Color Tint');
    const texture = graph.nodes.find((node) => node.parameter?.name === 'Color Texture');
    const output = graph.nodes.find((node) => node.type === 'output');
    expect(tint).toMatchObject({ values: { value: [1, 1, 1, 1] } });
    expect(texture).toMatchObject({ values: { texture: '', dimension: '2d' } });
    expect(output).toBeDefined();

    expect(materialGraphOutputConnected(graph, 'emissive')).toBe(true);
    expect(materialGraphOutputConnected(graph, 'opacity')).toBe(true);
    expect(materialGraphOutputConnected(graph, 'alphaClip')).toBe(true);
    expect(
      graph.connections.some(
        (connection) =>
          connection.from.nodeId === texture?.id && connection.from.pin === 'a' && connection.to.pin === 'b',
      ),
    ).toBe(true);

    expect(
      materialRenderPathLabel({
        domain: 'surface',
        blendMode: 'opaque',
        shadingModel: 'unlit',
        graph,
        customShader: false,
      }),
    ).toBe('Clustered Forward');
  });
});

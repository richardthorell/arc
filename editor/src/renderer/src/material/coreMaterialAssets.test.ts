import fs from 'node:fs';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { materialEditorParameters } from './materialCompiler';
import { materialGraphFromAsset, type MaterialAssetJson } from './materialGraphTypes';
import { materialGraphOutputConnected, materialRenderPathLabel } from './materialSettingsPresentation';

const readBuiltIn = (name: string) =>
  JSON.parse(
    fs.readFileSync(path.resolve(process.cwd(), '..', 'assets', 'materials', name), 'utf8'),
  ) as MaterialAssetJson;

describe('core built-in material families', () => {
  it('ships Standard Lit with neutral optional surface maps', () => {
    const asset = readBuiltIn('standard_lit.arcmat');
    expect(asset).toMatchObject({
      version: 4,
      name: 'Standard Lit',
      domain: 'surface',
      blendMode: 'opaque',
      shadingModel: 'standard',
      doubleSided: false,
    });

    const graph = materialGraphFromAsset(asset);
    expect(materialEditorParameters(graph).map((parameter) => parameter.name)).toEqual([
      'Base Color Tint',
      'Base Color Texture',
      'Metallic',
      'Roughness',
      'Metallic Roughness Texture',
      'Ambient Occlusion Texture',
      'Normal Texture',
      'Emissive Color',
      'Emissive Texture',
      'Emissive Strength',
    ]);
    expect(materialGraphOutputConnected(graph, 'metallic')).toBe(true);
    expect(materialGraphOutputConnected(graph, 'roughness')).toBe(true);
    expect(materialGraphOutputConnected(graph, 'ao')).toBe(true);
    expect(materialGraphOutputConnected(graph, 'normal')).toBe(true);

    const packed = graph.nodes.find((node) => node.parameter?.name === 'Metallic Roughness Texture');
    const ao = graph.nodes.find((node) => node.parameter?.name === 'Ambient Occlusion Texture');
    const normal = graph.nodes.find((node) => node.parameter?.name === 'Normal Texture');
    expect(packed).toMatchObject({ type: 'textureSample2D', values: { texture: '', dimension: '2d' } });
    expect(ao).toMatchObject({ type: 'textureSample2D', values: { texture: '', dimension: '2d' } });
    expect(normal).toMatchObject({
      type: 'textureSample2D',
      values: { texture: '', dimension: '2d', semantic: 'normal' },
    });
    expect(
      graph.connections.some((connection) => connection.from.nodeId === packed?.id && connection.from.pin === 'b'),
    ).toBe(true);
    expect(
      graph.connections.some((connection) => connection.from.nodeId === packed?.id && connection.from.pin === 'g'),
    ).toBe(true);
    expect(
      graph.connections.some(
        (connection) => connection.from.nodeId === ao?.id && connection.from.pin === 'r' && connection.to.pin === 'ao',
      ),
    ).toBe(true);
  });

  it('ships a texture-ready Glass transmission material', () => {
    const asset = readBuiltIn('glass.arcmat');
    expect(asset).toMatchObject({
      version: 4,
      name: 'Glass',
      domain: 'surface',
      blendMode: 'blend',
      shadingModel: 'transmission',
      doubleSided: true,
    });

    const graph = materialGraphFromAsset(asset);
    const parameters = materialEditorParameters(graph).map((parameter) => parameter.name);
    expect(parameters).toEqual([
      'Base Color Tint',
      'Base Color Texture',
      'Roughness',
      'Transmission',
      'Index of Refraction',
      'Thickness',
      'Attenuation Color',
      'Attenuation Distance',
      'Opacity',
    ]);
    expect(materialGraphOutputConnected(graph, 'transmission')).toBe(true);
    expect(materialGraphOutputConnected(graph, 'indexOfRefraction')).toBe(true);
    expect(materialGraphOutputConnected(graph, 'thickness')).toBe(true);
    expect(materialGraphOutputConnected(graph, 'attenuationColor')).toBe(true);
    expect(materialGraphOutputConnected(graph, 'attenuationDistance')).toBe(true);
    expect(
      materialRenderPathLabel({
        domain: 'surface',
        blendMode: 'blend',
        shadingModel: 'transmission',
        graph,
        customShader: false,
      }),
    ).toBe('Clustered Forward');
  });

  it('ships a texture-ready Subsurface material', () => {
    const asset = readBuiltIn('subsurface.arcmat');
    expect(asset).toMatchObject({
      version: 4,
      name: 'Subsurface',
      domain: 'surface',
      blendMode: 'opaque',
      shadingModel: 'skin',
      doubleSided: false,
    });

    const graph = materialGraphFromAsset(asset);
    expect(materialEditorParameters(graph).map((parameter) => parameter.name)).toEqual([
      'Base Color Tint',
      'Base Color Texture',
      'Roughness',
      'Subsurface Color',
      'Subsurface',
      'Thickness',
    ]);
    expect(materialGraphOutputConnected(graph, 'subsurfaceColor')).toBe(true);
    expect(materialGraphOutputConnected(graph, 'subsurface')).toBe(true);
    expect(materialGraphOutputConnected(graph, 'thickness')).toBe(true);
    expect(
      materialRenderPathLabel({
        domain: 'surface',
        blendMode: 'opaque',
        shadingModel: 'skin',
        graph,
        customShader: false,
      }),
    ).toBe('Clustered Forward');
  });
});

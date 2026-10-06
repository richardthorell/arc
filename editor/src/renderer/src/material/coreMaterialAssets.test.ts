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

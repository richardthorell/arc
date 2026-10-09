import fs from 'node:fs';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { materialEditorParameters } from './materialCompiler';
import { materialFunctionCompatibleWithSlot } from './materialInstanceAuthoring';
import { materialGraphFromAsset, type MaterialAssetJson, type MaterialFunctionAssetJson } from './materialGraphTypes';
import { materialGraphOutputConnected, materialRenderPathLabel } from './materialSettingsPresentation';

const readBuiltIn = (name: string) =>
  JSON.parse(
    fs.readFileSync(path.resolve(process.cwd(), '..', 'assets', 'materials', name), 'utf8'),
  ) as MaterialAssetJson;

const readBuiltInFunction = (name: string) =>
  JSON.parse(
    fs.readFileSync(path.resolve(process.cwd(), '..', 'assets', 'material_functions', name), 'utf8'),
  ) as MaterialFunctionAssetJson;

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
    const editorParameters = materialEditorParameters(graph);
    expect(editorParameters.map((parameter) => parameter.name)).toEqual([
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
    expect(materialGraphOutputConnected(graph, 'metallic')).toBe(true);
    expect(materialGraphOutputConnected(graph, 'roughness')).toBe(true);
    expect(materialGraphOutputConnected(graph, 'ao')).toBe(true);
    expect(editorParameters.find((parameter) => parameter.name === 'Metallic')?.range).toEqual({ min: 0, max: 1 });
    expect(editorParameters.find((parameter) => parameter.name === 'Roughness')?.range).toEqual({ min: 0, max: 1 });
    expect(editorParameters.find((parameter) => parameter.name === 'Clear Coat')?.range).toEqual({ min: 0, max: 1 });
    expect(editorParameters.find((parameter) => parameter.name === 'Clear Coat Roughness')?.range).toEqual({
      min: 0,
      max: 1,
    });
    expect(editorParameters.find((parameter) => parameter.name === 'Emissive Strength')?.range).toBeUndefined();
    expect(materialGraphOutputConnected(graph, 'normal')).toBe(true);
    expect(materialGraphOutputConnected(graph, 'clearCoat')).toBe(true);
    expect(materialGraphOutputConnected(graph, 'clearCoatRoughness')).toBe(true);
    expect(
      materialRenderPathLabel({
        domain: 'surface',
        blendMode: 'opaque',
        shadingModel: 'standard',
        graph,
        customShader: false,
      }),
    ).toBe('Deferred');

    const baseColorSource = graph.nodes.find((node) => node.id === 'base-color-source');
    expect(baseColorSource).toMatchObject({
      type: 'functionCall',
      values: {
        slotId: 'base-color-source',
        name: 'Base Color Source',
        path: 'assets/material_functions/default_base_color.arcmatfn',
        functions: [
          { path: 'assets/material_functions/default_base_color.arcmatfn' },
          { path: 'assets/material_functions/checker.arcmatfn' },
          { path: 'assets/material_functions/gradient.arcmatfn' },
          { path: 'assets/material_functions/noise.arcmatfn' },
        ],
        inputPins: [],
        outputPins: [{ id: 'color', name: 'Color', type: 'vec3' }],
      },
    });
    expect(graph.nodes.some((node) => node.id === 'base-color-multiply')).toBe(false);
    expect(graph.nodes.some((node) => node.id === 'base-color-tint')).toBe(false);
    expect(graph.nodes.some((node) => node.id === 'base-color-texture')).toBe(false);
    expect(
      graph.connections.some(
        (connection) =>
          connection.from.nodeId === 'base-color-source' &&
          connection.from.pin === 'color' &&
          connection.to.nodeId === 'material-output' &&
          connection.to.pin === 'baseColor',
      ),
    ).toBe(true);
    expect(
      graph.connections.some(
        (connection) =>
          connection.from.nodeId === 'base-color-multiply' &&
          connection.to.nodeId === 'material-output' &&
          connection.to.pin === 'baseColor',
      ),
    ).toBe(false);

    const defaultBaseColor = readBuiltInFunction('default_base_color.arcmatfn');
    expect(defaultBaseColor.inputs).toEqual([]);
    expect(defaultBaseColor.outputs).toEqual([{ id: 'color', name: 'Color', type: 'vec3' }]);
    expect(materialEditorParameters(defaultBaseColor.graph).map((parameter) => parameter.name)).toEqual([
      'Base Color Tint',
      'Base Color Texture',
    ]);

    const slotInputs = baseColorSource?.values.inputPins as MaterialFunctionAssetJson['inputs'];
    const slotOutputs = baseColorSource?.values.outputPins as MaterialFunctionAssetJson['outputs'];
    for (const functionName of [
      'default_base_color.arcmatfn',
      'checker.arcmatfn',
      'gradient.arcmatfn',
      'noise.arcmatfn',
    ]) {
      expect(
        materialFunctionCompatibleWithSlot(slotInputs, slotOutputs, readBuiltInFunction(functionName)),
        functionName,
      ).toBe(true);
    }

    const packed = graph.nodes.find((node) => node.parameter?.name === 'Metallic Roughness Texture');
    const ao = graph.nodes.find((node) => node.parameter?.name === 'Ambient Occlusion Texture');
    const normal = graph.nodes.find((node) => node.parameter?.name === 'Normal Texture');
    const clearCoat = graph.nodes.find((node) => node.parameter?.name === 'Clear Coat');
    const clearCoatRoughness = graph.nodes.find((node) => node.parameter?.name === 'Clear Coat Roughness');
    const clearCoatTexture = graph.nodes.find((node) => node.parameter?.name === 'Clear Coat Texture');
    expect(packed).toMatchObject({ type: 'textureSample2D', values: { texture: '', dimension: '2d' } });
    expect(ao).toMatchObject({ type: 'textureSample2D', values: { texture: '', dimension: '2d' } });
    expect(normal).toMatchObject({
      type: 'textureSample2D',
      values: { texture: '', dimension: '2d', semantic: 'normal' },
    });
    expect(clearCoat).toMatchObject({ type: 'constant', values: { value: 0, min: 0, max: 1 } });
    expect(clearCoatRoughness).toMatchObject({ type: 'constant', values: { value: 0.1, min: 0, max: 1 } });
    expect(clearCoatTexture).toMatchObject({
      type: 'textureSample2D',
      values: { texture: '', dimension: '2d', semantic: 'clear_coat' },
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
    const normalTexture = graph.nodes.find((node) => node.parameter?.name === 'Normal Texture');
    expect(normalTexture).toMatchObject({
      type: 'textureSample2D',
      values: { texture: '', dimension: '2d', semantic: 'normal' },
    });
    const editorParameters = materialEditorParameters(graph);
    const parameters = editorParameters.map((parameter) => parameter.name);
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
      'Normal Texture',
    ]);
    expect(editorParameters.find((parameter) => parameter.name === 'Roughness')?.range).toEqual({ min: 0, max: 1 });
    expect(editorParameters.find((parameter) => parameter.name === 'Transmission')?.range).toEqual({ min: 0, max: 1 });
    expect(editorParameters.find((parameter) => parameter.name === 'Opacity')?.range).toEqual({ min: 0, max: 1 });
    expect(editorParameters.find((parameter) => parameter.name === 'Index of Refraction')?.range).toBeUndefined();
    expect(editorParameters.find((parameter) => parameter.name === 'Thickness')?.range).toBeUndefined();
    expect(editorParameters.find((parameter) => parameter.name === 'Attenuation Distance')?.range).toBeUndefined();
    expect(materialGraphOutputConnected(graph, 'normal')).toBe(true);
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
    const normalTexture = graph.nodes.find((node) => node.parameter?.name === 'Normal Texture');
    expect(normalTexture).toMatchObject({
      type: 'textureSample2D',
      values: { texture: '', dimension: '2d', semantic: 'normal' },
    });
    const editorParameters = materialEditorParameters(graph);
    expect(editorParameters.map((parameter) => parameter.name)).toEqual([
      'Base Color Tint',
      'Base Color Texture',
      'Roughness',
      'Subsurface Color',
      'Subsurface',
      'Thickness',
      'Normal Texture',
    ]);
    expect(editorParameters.find((parameter) => parameter.name === 'Roughness')?.range).toEqual({ min: 0, max: 1 });
    expect(editorParameters.find((parameter) => parameter.name === 'Subsurface')?.range).toEqual({ min: 0, max: 1 });
    expect(editorParameters.find((parameter) => parameter.name === 'Thickness')?.range).toBeUndefined();
    expect(materialGraphOutputConnected(graph, 'normal')).toBe(true);
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

import { describe, expect, it } from 'vitest';

import { materialGraphDomain, materialGraphPinTypesCompatible } from './materialGraphDomain';
import { createMaterialNode, type MaterialGraphNode } from './materialGraphTypes';

const textureNode = (dimension: '2d' | 'cube' | '3d'): MaterialGraphNode => ({
  id: `texture-${dimension}`,
  type: 'textureSample',
  position: [0, 0],
  values: { texture: '', dimension },
});

describe('materialGraphDomain', () => {
  it('adapts material definitions to the shared graph domain contract', () => {
    const node = createMaterialNode('multiply', [10, 20]);
    expect(materialGraphDomain.getNodeDefinition(node).title).toBe('Multiply');
    expect(materialGraphDomain.getNodeDefinitions().some((definition) => definition.type === 'textureSample2D')).toBe(
      true,
    );
  });

  it('keeps material-specific protection and connection rules behind the domain boundary', () => {
    const source = createMaterialNode('constant', [0, 0]);
    const target = createMaterialNode('multiply', [100, 0]);
    const sourcePin = materialGraphDomain.getNodeDefinition(source).outputs[0]!;
    const targetPin = materialGraphDomain.getNodeDefinition(target).inputs[0]!;

    expect(
      materialGraphDomain.canConnect(
        { node: source, pin: sourcePin, direction: 'output' },
        { node: target, pin: targetPin, direction: 'input' },
      ),
    ).toEqual({ allowed: true });
    expect(
      materialGraphDomain.canConnect(
        { node: source, pin: sourcePin, direction: 'input' },
        { node: target, pin: targetPin, direction: 'input' },
      ).allowed,
    ).toBe(false);
    expect(materialGraphDomain.canDeleteNode(createMaterialNode('output', [0, 0]))).toBe(false);
  });

  it('accepts generic numeric pins but rejects incompatible concrete value types', () => {
    expect(materialGraphPinTypesCompatible('float', 'numeric')).toBe(true);
    expect(materialGraphPinTypesCompatible('numeric', 'vec3')).toBe(true);
    expect(materialGraphPinTypesCompatible('vec3', 'vec3')).toBe(true);
    expect(materialGraphPinTypesCompatible('vec4', 'vec3')).toBe(false);
    expect(materialGraphPinTypesCompatible('texture2d', 'vec4')).toBe(false);

    const color = createMaterialNode('colorRgba', [0, 0]);
    const output = createMaterialNode('output', [100, 0]);
    const rgba = materialGraphDomain.getNodeDefinition(color).outputs.find((pin) => pin.id === 'rgba')!;
    const baseColor = materialGraphDomain.getNodeDefinition(output).inputs.find((pin) => pin.id === 'baseColor')!;
    expect(
      materialGraphDomain.canConnect(
        { node: color, pin: rgba, direction: 'output' },
        { node: output, pin: baseColor, direction: 'input' },
      ).allowed,
    ).toBe(false);
  });
});

describe('material texture sample dimensions', () => {
  it('uses vec2 UVs for Texture2D', () => {
    const definition = materialGraphDomain.getNodeDefinition(textureNode('2d'));
    expect(definition.inputs[0]).toMatchObject({ id: 'uv', label: 'UV', type: 'vec2' });
  });

  it('uses vec3 directions for TextureCube', () => {
    const definition = materialGraphDomain.getNodeDefinition(textureNode('cube'));
    expect(definition.inputs[0]).toMatchObject({ id: 'uv', label: 'Direction', type: 'vec3' });
  });

  it('uses vec3 UVW coordinates for Texture3D', () => {
    const definition = materialGraphDomain.getNodeDefinition(textureNode('3d'));
    expect(definition.inputs[0]).toMatchObject({ id: 'uv', label: 'UVW', type: 'vec3' });
  });

  it('exposes fixed texture sample dimensions and hides the legacy generic sample from the add menu', () => {
    const definitions = materialGraphDomain.getNodeDefinitions();
    expect(definitions.some((definition) => definition.type === 'textureSample')).toBe(false);
    expect(definitions.some((definition) => definition.type === 'textureSample2D')).toBe(true);
    expect(definitions.some((definition) => definition.type === 'textureSampleCube')).toBe(true);
    expect(definitions.some((definition) => definition.type === 'textureSample3D')).toBe(true);

    const sample2D = createMaterialNode('textureSample2D', [0, 0]);
    const sampleCube = createMaterialNode('textureSampleCube', [0, 0]);
    const sample3D = createMaterialNode('textureSample3D', [0, 0]);
    expect(materialGraphDomain.getNodeDefinition(sample2D).inputs[0]).toMatchObject({ label: 'UV', type: 'vec2' });
    expect(materialGraphDomain.getNodeDefinition(sampleCube).inputs[0]).toMatchObject({
      label: 'Direction',
      type: 'vec3',
    });
    expect(materialGraphDomain.getNodeDefinition(sample3D).inputs[0]).toMatchObject({ label: 'UVW', type: 'vec3' });
  });
});

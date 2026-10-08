import { describe, expect, it } from 'vitest';

import type { MaterialAssetJson } from '../material/materialGraphTypes';
import { buildAssetCreation } from './assetCreation';

const project = {
  root: 'D:/Project',
  assetRoot: 'D:/Project/Content',
};

describe('Material asset creation', () => {
  it('creates a texture-ready Standard Lit graph without requiring manual texture nodes', () => {
    const definition = buildAssetCreation(project, {
      kind: 'material',
      name: 'Wall',
      folder: 'Content/Materials',
    });

    expect(definition.asset).toMatchObject({
      name: 'Wall.arcmat',
      path: 'Content/Materials/Wall.arcmat',
      kind: 'material',
      scope: 'project',
      status: 'ready',
    });

    const asset = JSON.parse(definition.contents) as MaterialAssetJson;
    expect(asset).toMatchObject({
      version: 4,
      name: 'Wall',
      domain: 'surface',
      blendMode: 'opaque',
      shadingModel: 'standard',
    });

    const graph = asset.graph!;
    const tint = graph.nodes.find((node) => node.parameter?.name === 'Base Color Tint');
    const texture = graph.nodes.find((node) => node.parameter?.name === 'Base Color Texture');
    const multiply = graph.nodes.find((node) => node.type === 'multiply');
    const emissiveColor = graph.nodes.find((node) => node.parameter?.name === 'Emissive Color');
    const emissiveTexture = graph.nodes.find((node) => node.parameter?.name === 'Emissive Texture');
    const emissiveStrength = graph.nodes.find((node) => node.parameter?.name === 'Emissive Strength');
    const output = graph.nodes.find((node) => node.type === 'output');

    expect(tint).toMatchObject({
      type: 'colorRgba',
      values: { value: [0.78, 0.8, 0.84, 1] },
    });
    expect(texture).toMatchObject({
      type: 'textureSample2D',
      values: { texture: '', dimension: '2d' },
    });
    expect(emissiveColor).toMatchObject({
      type: 'colorRgba',
      values: { value: [1, 1, 1, 1] },
    });
    expect(emissiveTexture).toMatchObject({
      type: 'textureSample2D',
      values: { texture: '', dimension: '2d' },
    });
    expect(emissiveStrength).toMatchObject({
      type: 'constant',
      values: { value: 0 },
    });
    expect(multiply).toBeDefined();
    expect(output).toBeDefined();
    expect(
      graph.connections.some((connection) => connection.to.nodeId === texture?.id && connection.to.pin === 'uv'),
    ).toBe(false);
    expect(
      graph.connections.some(
        (connection) =>
          connection.from.nodeId === texture?.id &&
          connection.from.pin === 'rgb' &&
          connection.to.nodeId === multiply?.id,
      ),
    ).toBe(true);
    expect(
      graph.connections.some(
        (connection) =>
          connection.from.nodeId === tint?.id && connection.from.pin === 'rgb' && connection.to.nodeId === multiply?.id,
      ),
    ).toBe(true);
    expect(
      graph.connections.some(
        (connection) =>
          connection.from.nodeId === multiply?.id &&
          connection.from.pin === 'result' &&
          connection.to.nodeId === output?.id &&
          connection.to.pin === 'baseColor',
      ),
    ).toBe(true);
    expect(
      graph.connections.some((connection) => connection.to.nodeId === output?.id && connection.to.pin === 'emissive'),
    ).toBe(true);
  });
});

describe('Material Function asset creation', () => {
  it('creates an editable typed function graph with stable boundary nodes', () => {
    const definition = buildAssetCreation(project, {
      kind: 'materialFunction',
      name: 'Checker',
      folder: 'Content/MaterialFunctions',
    });

    expect(definition.asset).toMatchObject({
      name: 'Checker.arcmatfn',
      path: 'Content/MaterialFunctions/Checker.arcmatfn',
      kind: 'materialFunction',
      scope: 'project',
      status: 'ready',
    });

    const asset = JSON.parse(definition.contents);
    expect(asset).toMatchObject({
      kind: 'materialFunction',
      version: 1,
      name: 'Checker',
      inputs: [{ id: 'value', name: 'Value', type: 'vec3' }],
      outputs: [{ id: 'result', name: 'Result', type: 'vec3' }],
    });
    expect(asset.graph.nodes.some((node: { type: string }) => node.type === 'functionInput')).toBe(true);
    expect(asset.graph.nodes.some((node: { type: string }) => node.type === 'functionOutput')).toBe(true);
    expect(
      asset.graph.connections.some(
        (connection: { from: { pin: string }; to: { pin: string } }) =>
          connection.from.pin === 'value' && connection.to.pin === 'result',
      ),
    ).toBe(true);
  });
});

import fs from 'node:fs';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { isMaterialGraph, type MaterialFunctionAssetJson } from './materialGraphTypes';

const readFunction = (name: string) =>
  JSON.parse(
    fs.readFileSync(path.resolve(process.cwd(), '..', 'assets', 'material_functions', `${name}.arcmatfn`), 'utf8'),
  ) as MaterialFunctionAssetJson;

describe('parameter-stable tint function composition', () => {
  it('keeps Base Color parameter node identities while reusing the RGB tint helper', () => {
    const color = readFunction('default_base_color');
    const helper = readFunction('tint_rgb');
    expect(isMaterialGraph(color.graph)).toBe(true);
    expect(isMaterialGraph(helper.graph)).toBe(true);
    expect(helper.inputs.map((pin) => pin.id)).toEqual(['color', 'tint']);
    expect(helper.outputs).toEqual([{ id: 'color', name: 'Color', type: 'vec3' }]);
    expect(helper.graph.nodes.some((node) => node.type === 'multiply')).toBe(true);

    const byId = new Map(color.graph.nodes.map((node) => [node.id, node]));
    expect(byId.get('base-color-tint')?.parameter).toEqual({ exposed: true, name: 'Base Color Tint' });
    expect(byId.get('base-color-texture')?.parameter).toEqual({ exposed: true, name: 'Base Color Texture' });
    const call = byId.get('base-color-multiply');
    expect(call?.type).toBe('functionCall');
    expect(call?.values.path).toBe('material_functions/tint_rgb.arcmatfn');
    const edge = (from: string, output: string, to: string, input: string) =>
      color.graph.connections.some(
        (connection) =>
          connection.from.nodeId === from &&
          connection.from.pin === output &&
          connection.to.nodeId === to &&
          connection.to.pin === input,
      );
    expect(edge('base-color-texture', 'rgb', 'base-color-multiply', 'color')).toBe(true);
    expect(edge('base-color-tint', 'rgb', 'base-color-multiply', 'tint')).toBe(true);
    expect(edge('base-color-multiply', 'color', 'function-output', 'color')).toBe(true);
    expect(edge('input-uv', 'value', 'base-color-texture', 'uv')).toBe(true);
  });

  it('keeps the public Texture Tint asset parameters and UV sampling', () => {
    const textureTint = readFunction('texture_tint');
    expect(textureTint.graph.nodes.filter((node) => node.parameter?.exposed).map((node) => node.id)).toEqual([
      'texture',
      'tint',
    ]);
    expect(textureTint.graph.nodes.find((node) => node.id === 'tinted')?.values.path).toBe(
      'material_functions/tint_rgb.arcmatfn',
    );
  });
});

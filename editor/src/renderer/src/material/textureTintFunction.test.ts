import fs from 'node:fs';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { isMaterialGraph, type MaterialFunctionAssetJson } from './materialGraphTypes';

const functionPath = path.resolve(process.cwd(), '..', 'assets', 'material_functions', 'texture_tint.arcmatfn');

describe('Texture Tint built-in Material Function', () => {
  it('samples the supplied UV and multiplies RGB by an editable tint', () => {
    const asset = JSON.parse(fs.readFileSync(functionPath, 'utf8')) as MaterialFunctionAssetJson;
    expect(asset.kind).toBe('materialFunction');
    expect(isMaterialGraph(asset.graph)).toBe(true);
    expect(asset.inputs).toEqual([{ id: 'uv', name: 'UV', type: 'vec2' }]);
    expect(asset.outputs).toEqual([{ id: 'color', name: 'Color', type: 'vec3' }]);

    const nodes = new Map(asset.graph.nodes.map((node) => [node.id, node]));
    const connected = (from: string, pin: string, to: string, targetPin: string) =>
      asset.graph.connections.some(
        (connection) =>
          connection.from.nodeId === from &&
          connection.from.pin === pin &&
          connection.to.nodeId === to &&
          connection.to.pin === targetPin,
      );

    expect(nodes.get('tint')?.parameter).toEqual({ exposed: true, name: 'Tint' });
    expect(nodes.get('tint')?.values.value).toEqual([1, 1, 1, 1]);
    expect(nodes.get('texture')?.parameter).toEqual({ exposed: true, name: 'Texture' });
    expect(nodes.get('texture')?.values.texture).toBe('');
    expect(connected('uv-input', 'value', 'texture', 'uv')).toBe(true);
    expect(connected('texture', 'rgb', 'tinted', 'color')).toBe(true);
    expect(connected('tint', 'rgb', 'tinted', 'tint')).toBe(true);
    expect(connected('tinted', 'color', 'out', 'color')).toBe(true);
  });
});

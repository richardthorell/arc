import fs from 'node:fs';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { isMaterialGraph, type MaterialFunctionAssetJson } from './materialGraphTypes';

const root = path.resolve(process.cwd(), '..', 'assets', 'material_functions');
const read = (file: string) =>
  JSON.parse(fs.readFileSync(path.join(root, `${file}.arcmatfn`), 'utf8')) as MaterialFunctionAssetJson;

const wiredSource = (asset: MaterialFunctionAssetJson, target: string, pin: string) =>
  asset.graph.connections.find((connection) => connection.to.nodeId === target && connection.to.pin === pin)?.from
    .nodeId;

describe('packed channel material functions', () => {
  it('extracts a selectable RGBA channel with a default R mask', () => {
    const asset = read('channel_mask');
    expect(isMaterialGraph(asset.graph)).toBe(true);
    expect(asset.inputs.map((input) => input.id)).toEqual(['rgba', 'mask']);
    expect(asset.inputs[1]?.default).toEqual([1, 0, 0, 0]);
    expect(asset.outputs).toEqual([{ id: 'value', name: 'Value', type: 'float' }]);
    expect(wiredSource(asset, 'dot', 'a')).toBe('in-rgba');
    expect(wiredSource(asset, 'dot', 'b')).toBe('in-mask');
    expect(wiredSource(asset, 'out', 'value')).toBe('dot');
  });

  it('unpacks AO from R, roughness from G, and metallic from B', () => {
    const asset = read('unpack_orm');
    expect(isMaterialGraph(asset.graph)).toBe(true);
    expect(asset.inputs).toEqual([
      { id: 'rgba', name: 'Packed ORM RGBA', type: 'vec4', default: [1, 1, 0, 1] },
    ]);
    for (const [channel, output, weights] of [
      ['r', 'ao', [1, 0, 0, 0]],
      ['g', 'roughness', [0, 1, 0, 0]],
      ['b', 'metallic', [0, 0, 1, 0]],
    ] as const) {
      expect(asset.graph.nodes.find((node) => node.id === `mask-${channel}`)?.values.value).toEqual(weights);
      expect(wiredSource(asset, `dot-${channel}`, 'a')).toBe('in-rgba');
      expect(wiredSource(asset, `dot-${channel}`, 'b')).toBe(`mask-${channel}`);
      expect(wiredSource(asset, 'out', output)).toBe(`dot-${channel}`);
    }
  });
});

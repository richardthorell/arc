import fs from 'node:fs';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { isMaterialGraph, type MaterialFunctionAssetJson } from './materialGraphTypes';

const read = (name: string) =>
  JSON.parse(
    fs.readFileSync(path.resolve(process.cwd(), '..', 'assets', 'material_functions', `${name}.arcmatfn`), 'utf8'),
  ) as MaterialFunctionAssetJson;

const connected = (asset: MaterialFunctionAssetJson, from: string, fromPin: string, to: string, toPin: string) =>
  asset.graph.connections.some(
    (edge) => edge.from.nodeId === from && edge.from.pin === fromPin && edge.to.nodeId === to && edge.to.pin === toPin,
  );

describe('reusable PBR roughness utilities', () => {
  it('remaps a bounded roughness through a safe power curve and specified output range', () => {
    const fn = read('roughness_remap');
    expect(isMaterialGraph(fn.graph)).toBe(true);
    expect(fn.inputs.map((pin) => [pin.id, pin.default])).toEqual([
      ['roughness', undefined],
      ['min', 0],
      ['max', 1],
      ['power', 1],
    ]);
    expect(fn.graph.nodes.find((node) => node.id === 'epsilon')?.values.value).toBe(0.0001);
    expect(connected(fn, 'input-roughness', 'value', 'clamp-input', 'value')).toBe(true);
    expect(connected(fn, 'clamp-input', 'result', 'curve', 'base')).toBe(true);
    expect(connected(fn, 'safe-exponent', 'result', 'curve', 'exponent')).toBe(true);
    expect(connected(fn, 'curve', 'result', 'remap', 't')).toBe(true);
    expect(connected(fn, 'remap', 'result', 'out', 'roughness')).toBe(true);
  });

  it('converts glossiness into clamped inverse roughness', () => {
    const fn = read('glossiness_to_roughness');
    expect(isMaterialGraph(fn.graph)).toBe(true);
    expect(fn.inputs.map((pin) => pin.id)).toEqual(['glossiness']);
    expect(connected(fn, 'input-glossiness', 'value', 'clamp', 'value')).toBe(true);
    expect(connected(fn, 'clamp', 'result', 'invert', 'value')).toBe(true);
    expect(connected(fn, 'invert', 'result', 'out', 'roughness')).toBe(true);
  });
});

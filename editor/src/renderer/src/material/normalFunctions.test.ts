import fs from 'node:fs';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { isMaterialGraph, type MaterialFunctionAssetJson } from './materialGraphTypes';

const read = (name: string) =>
  JSON.parse(
    fs.readFileSync(path.resolve(process.cwd(), '..', 'assets', 'material_functions', `${name}.arcmatfn`), 'utf8'),
  ) as MaterialFunctionAssetJson;

const source = (asset: MaterialFunctionAssetJson, target: string, pin: string) =>
  asset.graph.connections.find((c) => c.to.nodeId === target && c.to.pin === pin)?.from.nodeId;

describe('reusable normal functions', () => {
  it('converts sampled RGB using the native tangent-space normal conversion', () => {
    const fn = read('normal_map');
    expect(isMaterialGraph(fn.graph)).toBe(true);
    expect(fn.inputs).toEqual([{ id: 'rgb', name: 'Normal RGB', type: 'vec3' }]);
    expect(fn.graph.nodes.find((n) => n.id === 'normal-map')?.type).toBe('normalMap');
    expect(source(fn, 'normal-map', 'texture')).toBe('input-rgb');
    expect(source(fn, 'out', 'normal')).toBe('normal-map');
  });

  it('blends world-space normals with clamped weight and safe normalization', () => {
    const fn = read('normal_blend');
    expect(isMaterialGraph(fn.graph)).toBe(true);
    expect(fn.inputs.find((pin) => pin.id === 'weight')?.default).toBe(0);
    expect(source(fn, 'clamped-weight', 'value')).toBe('input-weight');
    expect(source(fn, 'blend', 't')).toBe('clamped-weight');
    expect(source(fn, 'normal-length', 'value')).toBe('blend');
    expect(source(fn, 'safe-length', 'a')).toBe('normal-length');
    expect(source(fn, 'safe-length', 'b')).toBe('epsilon');
    expect(source(fn, 'normalized', 'a')).toBe('blend');
    expect(source(fn, 'normalized', 'b')).toBe('safe-length');
    expect(source(fn, 'out', 'normal')).toBe('normalized');
  });
});

import fs from 'node:fs';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { isMaterialGraph, type MaterialFunctionAssetJson } from './materialGraphTypes';

const read = (name: string) =>
  JSON.parse(
    fs.readFileSync(path.resolve(process.cwd(), '..', 'assets', 'material_functions', `${name}.arcmatfn`), 'utf8'),
  ) as MaterialFunctionAssetJson;

const connected = (fn: MaterialFunctionAssetJson, a: string, ap: string, b: string, bp: string) =>
  fn.graph.connections.some(
    (edge) => edge.from.nodeId === a && edge.from.pin === ap && edge.to.nodeId === b && edge.to.pin === bp,
  );

describe('dielectric PBR utility functions', () => {
  it('computes IOR-derived reflectance with a safe minimum index', () => {
    const fn = read('ior_to_f0');
    expect(isMaterialGraph(fn.graph)).toBe(true);
    expect(fn.inputs).toEqual([{ id: 'ior', name: 'Index of Refraction', type: 'float', default: 1.5 }]);
    expect(fn.outputs).toEqual([{ id: 'f0', name: 'F0', type: 'float' }]);
    expect(connected(fn, 'input-ior', 'value', 'safe-ior', 'a')).toBe(true);
    expect(connected(fn, 'one', 'value', 'safe-ior', 'b')).toBe(true);
    expect(connected(fn, 'numerator', 'result', 'ratio', 'a')).toBe(true);
    expect(connected(fn, 'denominator', 'result', 'ratio', 'b')).toBe(true);
    expect(connected(fn, 'ratio', 'result', 'square', 'a')).toBe(true);
    expect(connected(fn, 'ratio', 'result', 'square', 'b')).toBe(true);
    expect(connected(fn, 'square', 'result', 'out', 'f0')).toBe(true);
  });

  it('uses clamped Schlick Fresnel with neutral head-on reflectance', () => {
    const fn = read('fresnel_schlick');
    expect(isMaterialGraph(fn.graph)).toBe(true);
    expect(fn.inputs.map((pin) => [pin.id, pin.default])).toEqual([
      ['f0', 0.04],
      ['cosTheta', 1],
    ]);
    expect(connected(fn, 'clamp-cosine', 'result', 'one-minus-cosine', 'value')).toBe(true);
    expect(connected(fn, 'one-minus-cosine', 'result', 'pow5', 'base')).toBe(true);
    expect(fn.graph.nodes.find((node) => node.id === 'five')?.values.value).toBe(5);
    expect(connected(fn, 'pow5', 'result', 'scaled', 'b')).toBe(true);
    expect(connected(fn, 'combined', 'result', 'out', 'fresnel')).toBe(true);
  });
});

import fs from 'node:fs';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { isMaterialGraph, type MaterialAssetJson } from './materialGraphTypes';

describe('Standard Lit physical IOR', () => {
  it('exposes the renderer-native IOR input with its existing 1.5 default', () => {
    const asset = JSON.parse(
      fs.readFileSync(path.resolve(process.cwd(), '..', 'assets', 'materials', 'standard_lit.arcmat'), 'utf8'),
    ) as MaterialAssetJson;
    expect(isMaterialGraph(asset.graph)).toBe(true);
    const graph = asset.graph!;
    const ior = graph.nodes.find((node) => node.id === 'index-of-refraction');
    expect(ior?.type).toBe('constant');
    expect(ior?.values).toEqual({ value: 1.5, min: 1.0001, max: 3 });
    expect(ior?.parameter).toEqual({
      exposed: true,
      name: 'Index of Refraction',
      group: 'surface',
      order: 25,
    });
    expect(
      graph.connections.some(
        (edge) =>
          edge.from.nodeId === 'index-of-refraction' &&
          edge.from.pin === 'value' &&
          edge.to.nodeId === 'material-output' &&
          edge.to.pin === 'indexOfRefraction',
      ),
    ).toBe(true);
    expect(graph.nodes.some((node) => node.values.path === 'material_functions/fresnel_schlick.arcmatfn')).toBe(false);
    expect(graph.nodes.some((node) => node.values.path === 'material_functions/ior_to_f0.arcmatfn')).toBe(false);
  });
});

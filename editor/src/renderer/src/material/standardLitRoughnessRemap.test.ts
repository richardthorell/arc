import fs from 'node:fs';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { isMaterialGraph, type MaterialAssetJson } from './materialGraphTypes';

describe('Standard Lit roughness remapping', () => {
  it('applies a neutral remap after existing roughness and ORM multiplication', () => {
    const asset = JSON.parse(
      fs.readFileSync(path.resolve(process.cwd(), '..', 'assets', 'materials', 'standard_lit.arcmat'), 'utf8'),
    ) as MaterialAssetJson;
    expect(isMaterialGraph(asset.graph)).toBe(true);
    const graph = asset.graph!;
    const nodes = new Map(graph.nodes.map((node) => [node.id, node]));
    const wired = (a: string, ap: string, b: string, bp: string) =>
      graph.connections.some(
        (edge) => edge.from.nodeId === a && edge.from.pin === ap && edge.to.nodeId === b && edge.to.pin === bp,
      );

    expect(nodes.get('roughness')?.parameter?.name).toBe('Roughness');
    expect(nodes.get('metallic-roughness-texture')?.parameter?.name).toBe('Metallic Roughness Texture');
    expect(nodes.get('roughness-remap')?.values.path).toBe('material_functions/roughness_remap.arcmatfn');
    expect(nodes.get('roughness-remap')?.parameter?.exposed).toBe(false);
    for (const [node, name, value, pin] of [
      ['roughness-remap-min', 'Roughness Minimum', 0, 'min'],
      ['roughness-remap-max', 'Roughness Maximum', 1, 'max'],
      ['roughness-remap-power', 'Roughness Curve', 1, 'power'],
    ] as const) {
      expect(nodes.get(node)?.parameter?.name).toBe(name);
      expect(nodes.get(node)?.values.value).toBe(value);
      expect(wired(node, 'value', 'roughness-remap', pin)).toBe(true);
    }
    expect(wired('roughness-multiply', 'result', 'roughness-remap', 'roughness')).toBe(true);
    expect(wired('roughness-remap', 'roughness', 'material-output', 'roughness')).toBe(true);
    expect(wired('orm-unpack', 'roughness', 'roughness-multiply', 'b')).toBe(true);
  });
});

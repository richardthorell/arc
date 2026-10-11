import fs from 'node:fs';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { isMaterialGraph, type MaterialAssetJson } from './materialGraphTypes';

const read = () =>
  JSON.parse(
    fs.readFileSync(path.resolve(process.cwd(), '..', 'assets', 'materials', 'standard_lit.arcmat'), 'utf8'),
  ) as MaterialAssetJson;

describe('Standard Lit detail normal integration', () => {
  it('preserves the original normal input and routes optional detail through tangent-space RNM', () => {
    const asset = read();
    expect(isMaterialGraph(asset.graph)).toBe(true);
    const graph = asset.graph!;
    const nodes = new Map(graph.nodes.map((node) => [node.id, node]));
    const connected = (a: string, ap: string, b: string, bp: string) =>
      graph.connections.some(
        (edge) => edge.from.nodeId === a && edge.from.pin === ap && edge.to.nodeId === b && edge.to.pin === bp,
      );

    expect(nodes.get('normal-texture')?.parameter?.name).toBe('Normal Texture');
    expect(nodes.get('normal-map')?.type).toBe('normalMap');
    expect(nodes.get('detail-normal-texture')?.parameter?.name).toBe('Detail Normal Texture');
    expect(nodes.get('detail-normal-texture')?.values.semantic).toBe('normal');
    expect(nodes.get('detail-normal-strength')?.parameter?.name).toBe('Detail Normal Strength');
    expect(nodes.get('detail-normal-strength')?.values.value).toBe(0);
    expect(nodes.get('detail-normal-rnm')?.values.path).toBe('material_functions/reoriented_normal_blend.arcmatfn');
    expect(nodes.get('detail-normal-rnm')?.parameter?.exposed).toBe(false);
    expect(connected('normal-texture', 'rgb', 'base-normal-double', 'a')).toBe(true);
    expect(connected('base-normal-signed', 'result', 'detail-normal-rnm', 'base')).toBe(true);
    expect(connected('detail-normal-texture', 'rgb', 'detail-normal-double', 'a')).toBe(true);
    expect(connected('detail-normal-signed', 'result', 'detail-normal-rnm', 'detail')).toBe(true);
    expect(connected('detail-normal-strength', 'value', 'detail-normal-rnm', 'weight')).toBe(true);
    expect(connected('combined-normal-encoded', 'result', 'normal-map', 'texture')).toBe(true);
    expect(connected('normal-map', 'normal', 'material-output', 'normal')).toBe(true);
    expect(connected('shared-uv-transform', 'uv', 'detail-normal-texture', 'uv')).toBe(true);
  });
});

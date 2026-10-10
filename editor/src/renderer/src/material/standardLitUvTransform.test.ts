import fs from 'node:fs';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { isMaterialGraph, type MaterialAssetJson } from './materialGraphTypes';

const materialPath = path.resolve(process.cwd(), '..', 'assets', 'materials', 'standard_lit.arcmat');
const material = () => JSON.parse(fs.readFileSync(materialPath, 'utf8')) as MaterialAssetJson;

describe('Standard Lit shared UV Transform', () => {
  it('shares one UV function across existing surface texture samples', () => {
    const asset = material();
    expect(isMaterialGraph(asset.graph)).toBe(true);
    const graph = asset.graph!;
    const nodes = new Map(graph.nodes.map((node) => [node.id, node]));
    const connected = (from: string, output: string, to: string, input: string) =>
      graph.connections.some(
        (edge) =>
          edge.from.nodeId === from && edge.from.pin === output && edge.to.nodeId === to && edge.to.pin === input,
      );

    expect(nodes.get('shared-uv-transform')?.values.path).toBe('material_functions/uv_transform.arcmatfn');
    expect(connected('shared-uv-transform', 'uv', 'base-color-source', 'uv')).toBe(true);
    expect(nodes.get('base-color-source')?.values.inputPins).toEqual([{ id: 'uv', name: 'UV', type: 'vec2' }]);
    expect(nodes.get('shared-uv-transform')?.parameter?.exposed).toBe(false);
    for (const [node, pin] of [
      ['uv-tiling', 'tiling'],
      ['uv-offset', 'offset'],
      ['uv-pivot', 'pivot'],
      ['uv-rotation', 'rotation'],
    ]) {
      expect(connected(node, 'value', 'shared-uv-transform', pin)).toBe(true);
    }
    for (const texture of [
      'metallic-roughness-texture',
      'ambient-occlusion-texture',
      'normal-texture',
      'clear-coat-texture',
      'emissive-texture',
    ]) {
      expect(connected('shared-uv-transform', 'uv', texture, 'uv')).toBe(true);
    }
    expect(nodes.get('uv-tiling')?.values.value).toEqual([1, 1]);
    expect(nodes.get('uv-offset')?.values.value).toEqual([0, 0]);
    expect(nodes.get('uv-pivot')?.values.value).toEqual([0.5, 0.5]);
    expect(nodes.get('uv-rotation')?.values.value).toBe(0);
    expect(graph.groups?.find((group) => group.id === 'surface')?.nodeIds).toEqual(
      expect.arrayContaining(['shared-uv-transform', 'uv-tiling', 'uv-offset', 'uv-pivot', 'uv-rotation']),
    );
  });
});

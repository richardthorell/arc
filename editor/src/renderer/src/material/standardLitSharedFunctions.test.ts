import fs from 'node:fs';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { isMaterialGraph, type MaterialAssetJson } from './materialGraphTypes';

const materialPath = path.resolve(process.cwd(), '..', 'assets', 'materials', 'standard_lit.arcmat');
const read = () => JSON.parse(fs.readFileSync(materialPath, 'utf8')) as MaterialAssetJson;

describe('Standard Lit shared channel functions', () => {
  it('routes existing texture samples through the reusable ORM and Channel Mask functions', () => {
    const material = read();
    expect(isMaterialGraph(material.graph)).toBe(true);
    const graph = material.graph!;
    const byId = new Map(graph.nodes.map((node) => [node.id, node]));
    const connected = (source: string, pin: string, target: string, targetPin: string) =>
      graph.connections.some(
        (edge) =>
          edge.from.nodeId === source &&
          edge.from.pin === pin &&
          edge.to.nodeId === target &&
          edge.to.pin === targetPin,
      );

    expect(byId.get('orm-unpack')?.values.path).toBe('material_functions/unpack_orm.arcmatfn');
    expect(byId.get('ao-channel-mask')?.values.path).toBe('material_functions/channel_mask.arcmatfn');
    expect(connected('metallic-roughness-texture', 'rgba', 'orm-unpack', 'rgba')).toBe(true);
    expect(connected('orm-unpack', 'metallic', 'metallic-multiply', 'b')).toBe(true);
    expect(connected('orm-unpack', 'roughness', 'roughness-multiply', 'b')).toBe(true);
    expect(connected('ambient-occlusion-texture', 'rgba', 'ao-channel-mask', 'rgba')).toBe(true);
    expect(connected('ao-channel-mask', 'value', 'material-output', 'ao')).toBe(true);
    expect(byId.get('base-color-source')?.parameter?.exposed).toBe(true);
    expect(byId.get('orm-unpack')?.parameter?.exposed).toBe(false);
    expect(byId.get('ao-channel-mask')?.parameter?.exposed).toBe(false);
    expect(
      graph.nodes
        .filter((node) => node.parameter?.exposed === true && node.type !== 'functionCall')
        .map((node) => node.parameter?.name),
    ).toEqual([
      'Metallic',
      'Roughness',
      'Metallic Roughness Texture',
      'Ambient Occlusion Texture',
      'Normal Texture',
      'Clear Coat',
      'Clear Coat Roughness',
      'Clear Coat Texture',
      'Emissive Color',
      'Emissive Texture',
      'Emissive Strength',
      'UV Tiling',
      'UV Offset',
      'UV Pivot',
      'UV Rotation (radians)',
    ]);
  });
});

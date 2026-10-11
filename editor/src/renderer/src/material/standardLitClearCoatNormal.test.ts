import fs from 'node:fs';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { isMaterialGraph, type MaterialAssetJson } from './materialGraphTypes';

describe('Standard Lit clear-coat normal', () => {
  it('uses a separate tangent-space normal texture without changing base normal controls', () => {
    const asset = JSON.parse(
      fs.readFileSync(path.resolve(process.cwd(), '..', 'assets', 'materials', 'standard_lit.arcmat'), 'utf8'),
    ) as MaterialAssetJson;
    expect(isMaterialGraph(asset.graph)).toBe(true);
    const graph = asset.graph!;
    const nodes = new Map(graph.nodes.map((node) => [node.id, node]));
    const edge = (from: string, fromPin: string, to: string, toPin: string) =>
      graph.connections.some(
        (connection) =>
          connection.from.nodeId === from &&
          connection.from.pin === fromPin &&
          connection.to.nodeId === to &&
          connection.to.pin === toPin,
      );

    expect(nodes.get('normal-texture')?.parameter?.name).toBe('Normal Texture');
    expect(edge('normal-map', 'normal', 'material-output', 'normal')).toBe(true);
    expect(nodes.get('clear-coat-normal-texture')?.parameter).toEqual({
      exposed: true,
      name: 'Clear Coat Normal Texture',
      group: 'clear-coat',
      order: 40,
    });
    expect(nodes.get('clear-coat-normal-texture')?.values.semantic).toBe('normal');
    expect(nodes.get('clear-coat-normal-texture')?.values.texture).toBe('');
    expect(nodes.get('clear-coat-normal-map')?.type).toBe('normalMap');
    expect(edge('clear-coat-normal-texture', 'rgb', 'clear-coat-normal-map', 'texture')).toBe(true);
    expect(edge('clear-coat-normal-map', 'normal', 'material-output', 'clearCoatNormal')).toBe(true);
    expect(edge('shared-uv-transform', 'uv', 'clear-coat-normal-texture', 'uv')).toBe(true);
    expect(graph.groups?.find((group) => group.id === 'clear-coat')?.nodeIds).toEqual(
      expect.arrayContaining(['clear-coat-normal-texture', 'clear-coat-normal-map']),
    );
  });
});

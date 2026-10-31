import fs from 'node:fs';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { clearCoatNormalTextureActive } from './clearCoatNormalActivity';
import { cloneMaterialGraph, isMaterialGraph, type MaterialAssetJson } from './materialGraphTypes';

const standardLit = () => {
  const asset = JSON.parse(
    fs.readFileSync(path.resolve(process.cwd(), '..', 'assets', 'materials', 'standard_lit.arcmat'), 'utf8'),
  ) as MaterialAssetJson;
  if (!isMaterialGraph(asset.graph)) throw new Error('Standard Lit has invalid graph');
  return asset.graph;
};

describe('clear-coat normal activity contract', () => {
  it('distinguishes neutral authored defaults from active instance texture assignments', () => {
    const graph = standardLit();
    expect(clearCoatNormalTextureActive(graph)).toBe(false);
    expect(clearCoatNormalTextureActive(graph, { 'clear-coat-normal-texture': 'textures/coat.png' })).toBe(true);
    expect(clearCoatNormalTextureActive(graph, { 'clear-coat-normal-texture': '' })).toBe(false);
  });

  it('conservatively treats nonstandard connected normal graphs as active', () => {
    const graph = cloneMaterialGraph(standardLit());
    const edge = graph.connections.find((connection) => connection.to.pin === 'clearCoatNormal');
    if (!edge) throw new Error('missing clear coat normal output');
    edge.from.nodeId = 'normal-one';
    expect(clearCoatNormalTextureActive(graph)).toBe(true);
  });
});

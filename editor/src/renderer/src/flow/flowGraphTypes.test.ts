import { describe, expect, it } from 'vitest';

import {
  createDefaultFlowGraph,
  createFlowAsset,
  createFlowNode,
  flowGraphFromAsset,
  isFlowAssetJson,
  isFlowGraph,
} from './flowGraphTypes';

describe('Flow graph authoring schema', () => {
  it('creates a v1 Flow asset with a Begin Play entry point', () => {
    const asset = createFlowAsset('PlayerController');

    expect(asset.assetType).toBe('flow');
    expect(asset.version).toBe(1);
    expect(asset.name).toBe('PlayerController');
    expect(asset.graph.version).toBe(1);
    expect(asset.graph.nodes).toHaveLength(1);
    expect(asset.graph.nodes[0].type).toBe('beginPlay');
    expect(isFlowAssetJson(asset)).toBe(true);
  });

  it('round-trips the authored graph without sharing mutable state', () => {
    const asset = createFlowAsset('Gameplay');
    asset.graph.nodes.push(createFlowNode('branch', [360, 140]));

    const graph = flowGraphFromAsset(asset);
    graph.nodes[0].position[0] = 999;

    expect(asset.graph.nodes[0].position[0]).not.toBe(999);
    expect(isFlowGraph(JSON.parse(JSON.stringify(graph)))).toBe(true);
  });

  it('rejects connections that reference missing nodes', () => {
    const graph = createDefaultFlowGraph();
    graph.connections.push({
      id: 'broken',
      kind: 'execution',
      from: { nodeId: graph.nodes[0].id, pin: 'exec' },
      to: { nodeId: 'missing', pin: 'exec' },
    });

    expect(isFlowGraph(graph)).toBe(false);
  });
});

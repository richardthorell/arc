import { describe, expect, it } from 'vitest';

import { cloneMaterialGraph, createDefaultMaterialGraph } from './materialGraphTypes';
import {
  materialGraphOutputConnected,
  materialGraphOutputSource,
  materialRenderPathLabel,
} from './materialSettingsPresentation';

describe('material settings presentation', () => {
  it('reports graph-driven outputs and stable defaults', () => {
    const graph = createDefaultMaterialGraph();

    expect(materialGraphOutputConnected(graph, 'opacity')).toBe(false);
    expect(materialGraphOutputSource(graph, 'opacity', '1.0')).toBe('Default 1.0');

    const connected = cloneMaterialGraph(graph);
    const output = connected.nodes.find((node) => node.type === 'output');
    const source = connected.nodes.find((node) => node.type === 'constant');
    expect(output).toBeDefined();
    expect(source).toBeDefined();
    connected.connections.push({
      id: 'test-opacity',
      from: { nodeId: source!.id, pin: 'value' },
      to: { nodeId: output!.id, pin: 'opacity' },
    });

    expect(materialGraphOutputConnected(connected, 'opacity')).toBe(true);
    expect(materialGraphOutputSource(connected, 'opacity', '1.0')).toBe('Graph');
  });

  it('derives the same high-level render routing rules as ARC materials', () => {
    const graph = createDefaultMaterialGraph();

    expect(
      materialRenderPathLabel({
        domain: 'surface',
        blendMode: 'opaque',
        shadingModel: 'standard',
        graph,
        customShader: false,
      }),
    ).toBe('Deferred');

    expect(
      materialRenderPathLabel({
        domain: 'surface',
        blendMode: 'blend',
        shadingModel: 'standard',
        graph,
        customShader: false,
      }),
    ).toBe('Clustered Forward');

    expect(
      materialRenderPathLabel({
        domain: 'surface',
        blendMode: 'opaque',
        shadingModel: 'unlit',
        graph,
        customShader: false,
      }),
    ).toBe('Clustered Forward');

    const featureGraph = cloneMaterialGraph(graph);
    const output = featureGraph.nodes.find((node) => node.type === 'output');
    const source = featureGraph.nodes.find((node) => node.type === 'constant');
    featureGraph.connections.push({
      id: 'test-clear-coat',
      from: { nodeId: source!.id, pin: 'value' },
      to: { nodeId: output!.id, pin: 'clearCoat' },
    });
    expect(
      materialRenderPathLabel({
        domain: 'surface',
        blendMode: 'opaque',
        shadingModel: 'standard',
        graph: featureGraph,
        customShader: false,
      }),
    ).toBe('Clustered Forward');

    expect(
      materialRenderPathLabel({
        domain: 'surface',
        blendMode: 'opaque',
        shadingModel: 'standard',
        graph,
        customShader: true,
      }),
    ).toBe('Clustered Forward');

    expect(
      materialRenderPathLabel({
        domain: 'terrain',
        blendMode: 'opaque',
        shadingModel: 'standard',
        graph,
        customShader: false,
      }),
    ).toBe('Terrain Renderer');
  });
});

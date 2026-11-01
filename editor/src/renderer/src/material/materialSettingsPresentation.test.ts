import { describe, expect, it } from 'vitest';

import { emptyMaterialCompileResult, compilingMaterialResult, nativeMaterialCompileResult } from './materialCompiler';
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
    const output = connected.nodes.find((node) => node.type === 'output')!;
    const source = connected.nodes.find((node) => node.type === 'constant')!;
    connected.connections.push({
      id: 'test-opacity',
      from: { nodeId: source.id, pin: 'value' },
      to: { nodeId: output.id, pin: 'opacity' },
    });
    expect(materialGraphOutputConnected(connected, 'opacity')).toBe(true);
    expect(materialGraphOutputSource(connected, 'opacity', '1.0')).toBe('Graph');
  });

  it.each([
    ['deferred', 'Deferred'],
    ['clusteredForward', 'Clustered Forward'],
  ])('presents native %s routing without interpreting the graph', (materialRenderPath, label) => {
    expect(materialRenderPathLabel(nativeMaterialCompileResult(true, { succeeded: true, materialRenderPath }))).toBe(
      label,
    );
  });

  it('does not guess when the native host omits routing or returns an unknown path', () => {
    for (const materialRenderPath of [undefined, 'future-path']) {
      expect(materialRenderPathLabel(nativeMaterialCompileResult(true, { succeeded: true, materialRenderPath }))).toBe(
        'Unavailable',
      );
    }
  });

  it('hides old routing during compilation and after failure', () => {
    const previous = nativeMaterialCompileResult(true, { succeeded: true, materialRenderPath: 'deferred' });
    expect(materialRenderPathLabel(emptyMaterialCompileResult())).toBe('Pending compilation');
    expect(materialRenderPathLabel(compilingMaterialResult(previous))).toBe('Pending compilation');
    expect(compilingMaterialResult(previous).materialRenderPath).toBeUndefined();
    for (const responseSucceeded of [true, false]) {
      const failed = nativeMaterialCompileResult(responseSucceeded, {
        succeeded: false,
        materialRenderPath: 'deferred',
      });
      expect(failed.materialRenderPath).toBeUndefined();
      expect(materialRenderPathLabel(failed)).toBe('Unavailable');
    }
  });
});

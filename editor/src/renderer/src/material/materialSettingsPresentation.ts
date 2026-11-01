import type { MaterialCompileResult } from './materialCompiler';
import type { MaterialGraph } from './materialGraphTypes';

export type MaterialRenderPathLabel = 'Deferred' | 'Clustered Forward' | 'Pending compilation' | 'Unavailable';

export const materialGraphOutputConnected = (graph: MaterialGraph, pin: string): boolean => {
  const output = graph.nodes.find((node) => node.type === 'output');
  if (!output) return false;
  return graph.connections.some((connection) => connection.to.nodeId === output.id && connection.to.pin === pin);
};

export const materialGraphOutputSource = (graph: MaterialGraph, pin: string, fallback: string): string =>
  materialGraphOutputConnected(graph, pin) ? 'Graph' : `Default ${fallback}`;

export const materialRenderPathLabel = (compilation: MaterialCompileResult): MaterialRenderPathLabel => {
  if (compilation.status === 'idle' || compilation.status === 'compiling') return 'Pending compilation';
  if (!compilation.succeeded) return 'Unavailable';
  if (compilation.materialRenderPath === 'deferred') return 'Deferred';
  if (compilation.materialRenderPath === 'clusteredForward') return 'Clustered Forward';
  return 'Unavailable';
};

import type {
  MaterialBlendMode,
  MaterialDomain,
  MaterialGraph,
  MaterialShadingModel,
} from './materialGraphTypes';

export type MaterialRenderPathLabel = 'Deferred' | 'Clustered Forward' | 'Terrain Renderer';

const forwardFeaturePins = ['clearCoat', 'sheen', 'transmission', 'subsurface', 'anisotropy'] as const;

export const materialGraphOutputConnected = (graph: MaterialGraph, pin: string): boolean => {
  const output = graph.nodes.find((node) => node.type === 'output');
  if (!output) return false;
  return graph.connections.some((connection) => connection.to.nodeId === output.id && connection.to.pin === pin);
};

export const materialGraphOutputSource = (graph: MaterialGraph, pin: string, fallback: string): string =>
  materialGraphOutputConnected(graph, pin) ? 'Graph' : `Default ${fallback}`;

export const materialRenderPathLabel = ({
  domain,
  blendMode,
  shadingModel,
  graph,
  customShader,
}: {
  domain: MaterialDomain;
  blendMode: MaterialBlendMode;
  shadingModel: MaterialShadingModel;
  graph: MaterialGraph;
  customShader: boolean;
}): MaterialRenderPathLabel => {
  if (domain === 'terrain') return 'Terrain Renderer';
  if (
    customShader ||
    blendMode === 'blend' ||
    shadingModel !== 'standard' ||
    forwardFeaturePins.some((pin) => materialGraphOutputConnected(graph, pin))
  )
    return 'Clustered Forward';
  return 'Deferred';
};

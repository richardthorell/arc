import type { MaterialGraph } from './materialGraphTypes';

/**
 * Material-instance-aware authoring signal for separate clear-coat normals.
 *
 * This is an input to future native material variant specialization, NOT a render
 * path decision. The renderer must continue to report Forward for graph-connected
 * clearCoatNormal until a neutral shader variant is compiled and selected.
 */
export const clearCoatNormalTextureActive = (
  graph: MaterialGraph,
  resolvedTextures: Readonly<Record<string, string | null | undefined>> = {},
): boolean => {
  const output = graph.nodes.find((node) => node.type === 'output');
  if (!output) return false;
  const coatNormal = graph.connections.find(
    (connection) => connection.to.nodeId === output.id && connection.to.pin === 'clearCoatNormal',
  );
  if (!coatNormal) return false;

  // General authored custom normal networks cannot safely be considered neutral.
  const normalMap = graph.nodes.find((node) => node.id === coatNormal.from.nodeId);
  if (!normalMap || normalMap.type !== 'normalMap') return true;

  const sample = graph.connections.find(
    (connection) => connection.to.nodeId === normalMap.id && connection.to.pin === 'texture',
  );
  const textureNode = graph.nodes.find((node) => node.id === sample?.from.nodeId);
  if (!textureNode || (textureNode.type !== 'textureSample2D' && textureNode.type !== 'textureSample')) {
    return true;
  }

  // Resolve Material Instance overrides before testing for an assigned texture.
  // Unknown/indirect sources are conservatively treated as active.
  if (Object.prototype.hasOwnProperty.call(resolvedTextures, textureNode.id)) {
    return Boolean(resolvedTextures[textureNode.id]?.trim());
  }
  return typeof textureNode.values.texture === 'string'
    ? textureNode.values.texture.trim().length > 0
    : true;
};

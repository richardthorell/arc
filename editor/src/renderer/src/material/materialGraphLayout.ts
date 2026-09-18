import { clampGraphZoom, type GraphPoint } from '../graph';
import { materialGraphDomain } from './materialGraphDomain';
import {
  cloneMaterialGraph,
  isMaterialTextureSampleNodeType,
  type MaterialGraph,
  type MaterialGraphNode,
  type MaterialGraphNodeType,
  type MaterialGraphViewport,
} from './materialGraphTypes';

const defaultNodeWidth = 214;
const headerHeight = 34;
const pinRowHeight = 25;
const nodePaddingTop = 9;
const nodePaddingBottom = 8;
const parameterEditorHeight = 38;
const valueEditorHeight = 38;
const textureEditorHeight = 62;

const framePadding = {
  left: 56,
  right: 56,
  top: 56,
  bottom: 88,
};

export const materialGraphSnapSize = 20;

export const materialNodeWidth = (type: MaterialGraphNodeType) => {
  switch (type) {
    case 'vector2':
      return 232;
    case 'vector3':
      return 252;
    case 'vector4':
      return 286;
    case 'colorRgb':
    case 'colorRgba':
      return 300;
    case 'textureSample':
    case 'textureSample2D':
    case 'textureSampleCube':
    case 'textureSample3D':
      return 286;
    case 'output':
      return 236;
    case 'normalMap':
    case 'clamp':
      return 232;
    default:
      return defaultNodeWidth;
  }
};

const editableValueNode = (node: MaterialGraphNode) =>
  node.type === 'constant' ||
  node.type === 'vector2' ||
  node.type === 'vector3' ||
  node.type === 'vector4' ||
  node.type === 'colorRgb' ||
  node.type === 'colorRgba';

export const materialNodeHeight = (node: MaterialGraphNode) => {
  const definition = materialGraphDomain.getNodeDefinition(node);
  const pinRows = Math.max(definition.inputs.length, definition.outputs.length);
  let height = headerHeight + nodePaddingTop + pinRows * pinRowHeight + nodePaddingBottom;

  if (isMaterialTextureSampleNodeType(node.type)) height += textureEditorHeight;
  else if (editableValueNode(node) || node.type === 'normalMap' || node.type === 'clamp')
    height += valueEditorHeight;

  if (editableValueNode(node)) height += parameterEditorHeight;
  return Math.max(88, height);
};

export type MaterialGraphBounds = {
  left: number;
  top: number;
  right: number;
  bottom: number;
  width: number;
  height: number;
};

export const materialGraphBounds = (graph: MaterialGraph): MaterialGraphBounds | null => {
  if (graph.nodes.length === 0) return null;

  let left = Number.POSITIVE_INFINITY;
  let top = Number.POSITIVE_INFINITY;
  let right = Number.NEGATIVE_INFINITY;
  let bottom = Number.NEGATIVE_INFINITY;

  for (const node of graph.nodes) {
    left = Math.min(left, node.position[0]);
    top = Math.min(top, node.position[1]);
    right = Math.max(right, node.position[0] + materialNodeWidth(node.type));
    bottom = Math.max(bottom, node.position[1] + materialNodeHeight(node));
  }

  return {
    left,
    top,
    right,
    bottom,
    width: Math.max(1, right - left),
    height: Math.max(1, bottom - top),
  };
};

export const frameMaterialGraphViewport = (
  graph: MaterialGraph,
  canvasWidth: number,
  canvasHeight: number,
): MaterialGraphViewport => {
  const bounds = materialGraphBounds(graph);
  if (!bounds || canvasWidth <= 0 || canvasHeight <= 0) return { x: 40, y: 40, zoom: 1 };

  const availableWidth = Math.max(1, canvasWidth - framePadding.left - framePadding.right);
  const availableHeight = Math.max(1, canvasHeight - framePadding.top - framePadding.bottom);
  const fittedZoom = Math.min(1, availableWidth / bounds.width, availableHeight / bounds.height);
  const zoom = clampGraphZoom(fittedZoom);

  return {
    x: framePadding.left + (availableWidth - bounds.width * zoom) / 2 - bounds.left * zoom,
    y: framePadding.top + (availableHeight - bounds.height * zoom) / 2 - bounds.top * zoom,
    zoom,
  };
};

export const snapMaterialGraphPoint = (point: GraphPoint, size = materialGraphSnapSize): GraphPoint => [
  Math.round(point[0] / size) * size,
  Math.round(point[1] / size) * size,
];

const average = (values: number[]) =>
  values.length > 0 ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;

export const autoArrangeMaterialGraph = (graph: MaterialGraph): MaterialGraph => {
  if (graph.nodes.length <= 1) return cloneMaterialGraph(graph);

  const next = cloneMaterialGraph(graph);
  const nodeById = new Map(next.nodes.map((node) => [node.id, node]));
  const outgoing = new Map<string, string[]>();
  const connected = new Set<string>();

  for (const connection of next.connections) {
    if (!nodeById.has(connection.from.nodeId) || !nodeById.has(connection.to.nodeId)) continue;
    const targets = outgoing.get(connection.from.nodeId) ?? [];
    targets.push(connection.to.nodeId);
    outgoing.set(connection.from.nodeId, targets);
    connected.add(connection.from.nodeId);
    connected.add(connection.to.nodeId);
  }

  const depthMemo = new Map<string, number>();
  const depthFor = (nodeId: string, visiting = new Set<string>()): number => {
    const cached = depthMemo.get(nodeId);
    if (cached !== undefined) return cached;
    if (visiting.has(nodeId)) return 0;

    const targets = outgoing.get(nodeId) ?? [];
    if (targets.length === 0) {
      depthMemo.set(nodeId, 0);
      return 0;
    }

    const branch = new Set(visiting);
    branch.add(nodeId);
    const depth = 1 + Math.max(...targets.map((target) => depthFor(target, branch)));
    depthMemo.set(nodeId, depth);
    return depth;
  };

  const depths = new Map(next.nodes.map((node) => [node.id, depthFor(node.id)]));
  const connectedDepths = next.nodes
    .filter((node) => connected.has(node.id))
    .map((node) => depths.get(node.id) ?? 0);
  const connectedMaximumDepth = connectedDepths.length > 0 ? Math.max(...connectedDepths) : 0;

  for (const node of next.nodes)
    if (!connected.has(node.id) && node.type !== 'output')
      depths.set(node.id, connectedMaximumDepth + 1);

  const maximumDepth = Math.max(...depths.values());
  const columns = new Map<number, MaterialGraphNode[]>();
  for (const node of next.nodes) {
    const depth = depths.get(node.id) ?? 0;
    const column = columns.get(depth) ?? [];
    column.push(node);
    columns.set(depth, column);
  }

  const columnSpacing = 350;
  const rowGap = 34;
  const centers = new Map<string, number>();

  for (let depth = 0; depth <= maximumDepth; ++depth) {
    const column = columns.get(depth);
    if (!column?.length) continue;

    const layout = column.map((node) => {
      const downstreamCenters = (outgoing.get(node.id) ?? []).flatMap((target) => {
        const center = centers.get(target);
        return center === undefined ? [] : [center];
      });
      return {
        node,
        height: materialNodeHeight(node),
        idealCenter: downstreamCenters.length
          ? average(downstreamCenters)
          : node.position[1] + materialNodeHeight(node) / 2,
      };
    });

    layout.sort(
      (left, right) =>
        left.idealCenter - right.idealCenter ||
        left.node.position[1] - right.node.position[1] ||
        left.node.id.localeCompare(right.node.id),
    );

    const totalHeight =
      layout.reduce((sum, item) => sum + item.height, 0) + Math.max(0, layout.length - 1) * rowGap;
    const targetCenter = average(layout.map((item) => item.idealCenter));
    let y = targetCenter - totalHeight / 2;

    for (const item of layout) {
      item.node.position = [(maximumDepth - depth) * columnSpacing, y];
      centers.set(item.node.id, y + item.height / 2);
      y += item.height + rowGap;
    }
  }

  const bounds = materialGraphBounds(next);
  if (bounds) {
    const offsetX = 80 - bounds.left;
    const offsetY = 80 - bounds.top;
    for (const node of next.nodes)
      node.position = snapMaterialGraphPoint([
        node.position[0] + offsetX,
        node.position[1] + offsetY,
      ]);
  }

  return next;
};

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
  else if (editableValueNode(node) || node.type === 'normalMap' || node.type === 'clamp') height += valueEditorHeight;

  if (editableValueNode(node)) height += parameterEditorHeight;
  return Math.max(88, height);
};

export const materialNodePinOffsetY = (
  node: MaterialGraphNode,
  pinId: string,
  direction: 'input' | 'output',
) => {
  const definition = materialGraphDomain.getNodeDefinition(node);
  const pins = direction === 'input' ? definition.inputs : definition.outputs;
  const index = pins.findIndex((pin) => pin.id === pinId);
  if (index < 0) return materialNodeHeight(node) / 2;
  return headerHeight + nodePaddingTop + pinRowHeight * (index + 0.5);
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
  const outgoingConnections = new Map<string, typeof next.connections>();
  const incoming = new Map<string, string[]>();
  const connectionCounts = new Map<string, number>();
  const connected = new Set<string>();

  for (const connection of next.connections) {
    if (!nodeById.has(connection.from.nodeId) || !nodeById.has(connection.to.nodeId)) continue;

    const targets = outgoing.get(connection.from.nodeId) ?? [];
    targets.push(connection.to.nodeId);
    outgoing.set(connection.from.nodeId, targets);

    const sourceConnections = outgoingConnections.get(connection.from.nodeId) ?? [];
    sourceConnections.push(connection);
    outgoingConnections.set(connection.from.nodeId, sourceConnections);

    const sources = incoming.get(connection.to.nodeId) ?? [];
    sources.push(connection.from.nodeId);
    incoming.set(connection.to.nodeId, sources);

    connectionCounts.set(connection.from.nodeId, (connectionCounts.get(connection.from.nodeId) ?? 0) + 1);
    connectionCounts.set(connection.to.nodeId, (connectionCounts.get(connection.to.nodeId) ?? 0) + 1);
    connected.add(connection.from.nodeId);
    connected.add(connection.to.nodeId);
  }

  // Stage the graph forward from its sources instead of backward from Material
  // Output. This keeps peer source nodes such as Texture Samples in one visual
  // column even when one branch contains an extra processor (for example a
  // Normal Map node) before reaching the output.
  const stageMemo = new Map<string, number>();
  const stageFor = (nodeId: string, visiting = new Set<string>()): number => {
    const cached = stageMemo.get(nodeId);
    if (cached !== undefined) return cached;
    if (visiting.has(nodeId)) return 0;

    const sources = incoming.get(nodeId) ?? [];
    if (sources.length === 0) {
      stageMemo.set(nodeId, 0);
      return 0;
    }

    const branch = new Set(visiting);
    branch.add(nodeId);
    const stage = 1 + Math.max(...sources.map((source) => stageFor(source, branch)));
    stageMemo.set(nodeId, stage);
    return stage;
  };

  const depths = new Map(next.nodes.map((node) => [node.id, stageFor(node.id)]));
  const maximumDepth = Math.max(...depths.values());
  const columns = new Map<number, MaterialGraphNode[]>();
  for (const node of next.nodes) {
    const depth = depths.get(node.id) ?? 0;
    const column = columns.get(depth) ?? [];
    column.push(node);
    columns.set(depth, column);
  }

  const columnWidths = new Map<number, number>();
  for (const [depth, column] of columns)
    columnWidths.set(depth, Math.max(...column.map((node) => materialNodeWidth(node.type))));

  const columnX = new Map<number, number>([[0, 0]]);
  for (let depth = 1; depth <= maximumDepth; ++depth) {
    const previousDepth = depth - 1;
    const previousX = columnX.get(previousDepth) ?? 0;
    const previousWidth = columnWidths.get(previousDepth) ?? defaultNodeWidth;

    let crossingConnections = 0;
    let maximumConnectionCount = 0;
    for (const connection of next.connections) {
      const fromDepth = depths.get(connection.from.nodeId);
      const toDepth = depths.get(connection.to.nodeId);
      if (fromDepth === undefined || toDepth === undefined) continue;
      if (fromDepth >= depth || toDepth < depth) continue;
      crossingConnections += 1;
      maximumConnectionCount = Math.max(
        maximumConnectionCount,
        connectionCounts.get(connection.from.nodeId) ?? 0,
        connectionCounts.get(connection.to.nodeId) ?? 0,
      );
    }

    const pressure = Math.max(crossingConnections, maximumConnectionCount);
    const baseGap = 150;
    const pressureGap = Math.min(220, Math.max(0, pressure - 2) * 22);
    columnX.set(depth, previousX + previousWidth + baseGap + pressureGap);
  }

  const rowGap = 34;

  // Place columns from the sinks back toward the sources. The ideal Y for an
  // upstream node is derived from the actual connected pin locations, not just
  // the downstream node center. Single-link chains therefore become straight,
  // while fan-in/fan-out nodes are centered across all of their connections.
  for (let depth = maximumDepth; depth >= 0; --depth) {
    const column = columns.get(depth);
    if (!column?.length) continue;

    const layout = column.map((node) => {
      const connectionTargets = (outgoingConnections.get(node.id) ?? []).flatMap((connection) => {
        const target = nodeById.get(connection.to.nodeId);
        if (!target || (depths.get(target.id) ?? 0) <= depth) return [];

        const targetPinY =
          target.position[1] + materialNodePinOffsetY(target, connection.to.pin, 'input');
        const sourcePinOffset = materialNodePinOffsetY(node, connection.from.pin, 'output');
        return [targetPinY - sourcePinOffset];
      });
      const height = materialNodeHeight(node);
      const idealTop = connectionTargets.length ? average(connectionTargets) : node.position[1];
      return {
        node,
        height,
        idealTop,
        idealCenter: idealTop + height / 2,
      };
    });

    layout.sort(
      (left, right) =>
        left.idealCenter - right.idealCenter ||
        left.node.position[1] - right.node.position[1] ||
        left.node.id.localeCompare(right.node.id),
    );

    const tops = layout.map((item) => item.idealTop);
    for (let index = 1; index < layout.length; ++index) {
      const minimumTop = tops[index - 1] + layout[index - 1].height + rowGap;
      tops[index] = Math.max(tops[index], minimumTop);
    }

    // Forward packing can push a whole column down. Translate it back toward
    // the connection-derived ideals without changing the non-overlap spacing.
    const packedOffset = average(tops.map((top, index) => top - layout[index].idealTop));
    for (let index = 0; index < layout.length; ++index) {
      const item = layout[index];
      item.node.position = [columnX.get(depth) ?? 0, tops[index] - packedOffset];
    }
  }

  const bounds = materialGraphBounds(next);
  if (bounds) {
    const offsetX = 80 - bounds.left;
    const offsetY = 80 - bounds.top;
    for (const node of next.nodes) {
      const x = Math.round((node.position[0] + offsetX) / materialGraphSnapSize) * materialGraphSnapSize;
      // Preserve relative vertical pin alignment. Independent Y snapping would
      // turn a perfectly straight pin-to-pin connection back into a curve.
      node.position = [x, Math.round(node.position[1] + offsetY)];
    }
  }

  return next;
};

import type { GraphPoint, GraphViewport } from './graphTypes';

export type GraphSelection = {
  start: GraphPoint;
  current: GraphPoint;
};

export type GraphSelectionBounds = {
  left: number;
  top: number;
  right: number;
  bottom: number;
};

type GraphViewportSize = {
  width: number;
  height: number;
};

export const graphPinKey = (nodeId: string, pin: string, output: boolean) =>
  `${nodeId}:${output ? 'output' : 'input'}:${pin}`;

export const sameGraphPointMaps = (left: Map<string, GraphPoint>, right: Map<string, GraphPoint>) => {
  if (left.size !== right.size) return false;
  for (const [key, point] of left) {
    const candidate = right.get(key);
    if (!candidate || Math.abs(candidate[0] - point[0]) > 0.01 || Math.abs(candidate[1] - point[1]) > 0.01)
      return false;
  }
  return true;
};

export const clientToGraphPoint = (
  rect: Pick<DOMRect, 'left' | 'top'>,
  viewport: GraphViewport,
  clientX: number,
  clientY: number,
): GraphPoint => [
  (clientX - rect.left - viewport.x) / viewport.zoom,
  (clientY - rect.top - viewport.y) / viewport.zoom,
];

export const graphConnectionPath = (from: GraphPoint, to: GraphPoint, minimumHandleDistance = 55) => {
  const distance = Math.max(minimumHandleDistance, Math.abs(to[0] - from[0]) * 0.45);
  return `M ${from[0]} ${from[1]} C ${from[0] + distance} ${from[1]}, ${to[0] - distance} ${to[1]}, ${to[0]} ${to[1]}`;
};

export const clampGraphZoom = (zoom: number, minimum = 0.35, maximum = 1.8) =>
  Math.min(maximum, Math.max(minimum, zoom));

export const graphViewportZoomAt = (
  viewport: GraphViewport,
  screenPoint: GraphPoint,
  zoom: number,
  minimum = 0.35,
  maximum = 1.8,
): GraphViewport => {
  const nextZoom = clampGraphZoom(zoom, minimum, maximum);
  const graphX = (screenPoint[0] - viewport.x) / viewport.zoom;
  const graphY = (screenPoint[1] - viewport.y) / viewport.zoom;
  return {
    x: screenPoint[0] - graphX * nextZoom,
    y: screenPoint[1] - graphY * nextZoom,
    zoom: nextZoom,
  };
};

export const graphViewportFitBounds = (
  bounds: GraphSelectionBounds,
  viewportSize: GraphViewportSize,
  padding = 48,
  minimumZoom = 0.35,
  maximumZoom = 1.8,
): GraphViewport => {
  const width = Math.max(0, bounds.right - bounds.left);
  const height = Math.max(0, bounds.bottom - bounds.top);
  const availableWidth = Math.max(1, viewportSize.width - padding * 2);
  const availableHeight = Math.max(1, viewportSize.height - padding * 2);
  const widthZoom = width > 0 ? availableWidth / width : maximumZoom;
  const heightZoom = height > 0 ? availableHeight / height : maximumZoom;
  const zoom = clampGraphZoom(Math.min(widthZoom, heightZoom), minimumZoom, maximumZoom);
  const centerX = (bounds.left + bounds.right) * 0.5;
  const centerY = (bounds.top + bounds.bottom) * 0.5;
  return {
    x: viewportSize.width * 0.5 - centerX * zoom,
    y: viewportSize.height * 0.5 - centerY * zoom,
    zoom,
  };
};

export const graphSelectionBounds = ({ start, current }: GraphSelection): GraphSelectionBounds => ({
  left: Math.min(start[0], current[0]),
  top: Math.min(start[1], current[1]),
  right: Math.max(start[0], current[0]),
  bottom: Math.max(start[1], current[1]),
});

export const graphSelectionScreenRect = (selection: GraphSelection, viewport: GraphViewport) => {
  const bounds = graphSelectionBounds(selection);
  return {
    left: viewport.x + bounds.left * viewport.zoom,
    top: viewport.y + bounds.top * viewport.zoom,
    width: (bounds.right - bounds.left) * viewport.zoom,
    height: (bounds.bottom - bounds.top) * viewport.zoom,
  };
};

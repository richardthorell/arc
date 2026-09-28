export interface GraphViewportState {
  x: number;
  y: number;
  zoom: number;
}

export interface GraphViewportLimits {
  minZoom: number;
  maxZoom: number;
}

export interface GraphBounds {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface GraphViewportSize {
  width: number;
  height: number;
}

export const DEFAULT_GRAPH_VIEWPORT: GraphViewportState = {
  x: 0,
  y: 0,
  zoom: 1,
};

export const DEFAULT_GRAPH_VIEWPORT_LIMITS: GraphViewportLimits = {
  minZoom: 0.1,
  maxZoom: 4,
};

const finiteOr = (value: unknown, fallback: number): number =>
  typeof value === 'number' && Number.isFinite(value) ? value : fallback;

export function clampGraphViewportZoom(
  zoom: number,
  limits: GraphViewportLimits = DEFAULT_GRAPH_VIEWPORT_LIMITS,
): number {
  const minZoom = Math.min(limits.minZoom, limits.maxZoom);
  const maxZoom = Math.max(limits.minZoom, limits.maxZoom);
  return Math.min(maxZoom, Math.max(minZoom, finiteOr(zoom, DEFAULT_GRAPH_VIEWPORT.zoom)));
}

/**
 * Normalizes persisted/domain-provided viewport data before it reaches a graph renderer.
 * Keeping this policy in the shared graph layer prevents Material, Flow, and future graph
 * editors from growing subtly different validation and zoom rules.
 */
export function normalizeGraphViewport(
  value: Partial<GraphViewportState> | null | undefined,
  fallback: GraphViewportState = DEFAULT_GRAPH_VIEWPORT,
  limits: GraphViewportLimits = DEFAULT_GRAPH_VIEWPORT_LIMITS,
): GraphViewportState {
  return {
    x: finiteOr(value?.x, fallback.x),
    y: finiteOr(value?.y, fallback.y),
    zoom: clampGraphViewportZoom(finiteOr(value?.zoom, fallback.zoom), limits),
  };
}

/** Returns a deterministic JSON-safe snapshot suitable for document/workspace persistence. */
export function serializeGraphViewport(
  value: GraphViewportState,
  limits: GraphViewportLimits = DEFAULT_GRAPH_VIEWPORT_LIMITS,
): GraphViewportState {
  return normalizeGraphViewport(value, DEFAULT_GRAPH_VIEWPORT, limits);
}

/**
 * Computes the shared viewport transform used by Fit Selection and Fit All.
 * Bounds are graph-space values and viewport dimensions/padding are screen-space pixels.
 */
export function fitGraphViewport(
  bounds: GraphBounds,
  viewport: GraphViewportSize,
  padding = 32,
  limits: GraphViewportLimits = DEFAULT_GRAPH_VIEWPORT_LIMITS,
): GraphViewportState {
  const viewportWidth = Math.max(0, finiteOr(viewport.width, 0));
  const viewportHeight = Math.max(0, finiteOr(viewport.height, 0));
  const width = Math.max(0, finiteOr(bounds.width, 0));
  const height = Math.max(0, finiteOr(bounds.height, 0));
  const safePadding = Math.max(0, finiteOr(padding, 0));
  const availableWidth = Math.max(0, viewportWidth - safePadding * 2);
  const availableHeight = Math.max(0, viewportHeight - safePadding * 2);

  const widthZoom = width > 0 ? availableWidth / width : Number.POSITIVE_INFINITY;
  const heightZoom = height > 0 ? availableHeight / height : Number.POSITIVE_INFINITY;
  const requestedZoom = Math.min(widthZoom, heightZoom);
  const zoom = clampGraphViewportZoom(
    Number.isFinite(requestedZoom) ? requestedZoom : limits.maxZoom,
    limits,
  );
  const centerX = finiteOr(bounds.x, 0) + width * 0.5;
  const centerY = finiteOr(bounds.y, 0) + height * 0.5;

  return {
    x: viewportWidth * 0.5 - centerX * zoom,
    y: viewportHeight * 0.5 - centerY * zoom,
    zoom,
  };
}

/**
 * Zooms around a screen-space anchor while preserving the graph point beneath that anchor.
 * The transform convention matches React Flow: screen = graph * zoom + translation.
 */
export function zoomGraphViewportAt(
  viewport: GraphViewportState,
  requestedZoom: number,
  anchor: { x: number; y: number },
  limits: GraphViewportLimits = DEFAULT_GRAPH_VIEWPORT_LIMITS,
): GraphViewportState {
  const current = normalizeGraphViewport(viewport, DEFAULT_GRAPH_VIEWPORT, limits);
  const zoom = clampGraphViewportZoom(requestedZoom, limits);
  if (zoom === current.zoom) return current;

  const graphX = (anchor.x - current.x) / current.zoom;
  const graphY = (anchor.y - current.y) / current.zoom;
  return {
    x: anchor.x - graphX * zoom,
    y: anchor.y - graphY * zoom,
    zoom,
  };
}

export function panGraphViewport(
  viewport: GraphViewportState,
  delta: { x: number; y: number },
  limits: GraphViewportLimits = DEFAULT_GRAPH_VIEWPORT_LIMITS,
): GraphViewportState {
  const current = normalizeGraphViewport(viewport, DEFAULT_GRAPH_VIEWPORT, limits);
  return {
    x: current.x + finiteOr(delta.x, 0),
    y: current.y + finiteOr(delta.y, 0),
    zoom: current.zoom,
  };
}

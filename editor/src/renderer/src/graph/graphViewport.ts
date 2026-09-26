export interface GraphViewportState {
  x: number;
  y: number;
  zoom: number;
}

export interface GraphViewportLimits {
  minZoom: number;
  maxZoom: number;
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

export function clampGraphZoom(zoom: number, limits: GraphViewportLimits = DEFAULT_GRAPH_VIEWPORT_LIMITS): number {
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
    zoom: clampGraphZoom(finiteOr(value?.zoom, fallback.zoom), limits),
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
  const zoom = clampGraphZoom(requestedZoom, limits);
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

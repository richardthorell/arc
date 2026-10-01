import type { GraphViewport } from './graphTypes';
import { clampGraphZoom } from './graphGeometry';

export const DEFAULT_GRAPH_VIEWPORT: GraphViewport = { x: 0, y: 0, zoom: 1 };

export type PersistedGraphViewport = {
  version: 1;
  x: number;
  y: number;
  zoom: number;
};

const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

export const persistGraphViewport = (viewport: GraphViewport): PersistedGraphViewport => ({
  version: 1,
  x: finite(viewport.x) ? viewport.x : DEFAULT_GRAPH_VIEWPORT.x,
  y: finite(viewport.y) ? viewport.y : DEFAULT_GRAPH_VIEWPORT.y,
  zoom: clampGraphZoom(finite(viewport.zoom) ? viewport.zoom : DEFAULT_GRAPH_VIEWPORT.zoom),
});

export const restoreGraphViewport = (value: unknown): GraphViewport => {
  if (!value || typeof value !== 'object') return { ...DEFAULT_GRAPH_VIEWPORT };

  const candidate = value as Partial<PersistedGraphViewport>;
  if (candidate.version !== 1 || !finite(candidate.x) || !finite(candidate.y) || !finite(candidate.zoom)) {
    return { ...DEFAULT_GRAPH_VIEWPORT };
  }

  return {
    x: candidate.x,
    y: candidate.y,
    zoom: clampGraphZoom(candidate.zoom),
  };
};

export const sameGraphViewport = (left: GraphViewport, right: GraphViewport, epsilon = 0.001) =>
  Math.abs(left.x - right.x) <= epsilon &&
  Math.abs(left.y - right.y) <= epsilon &&
  Math.abs(left.zoom - right.zoom) <= epsilon;

import { clampGraphZoom } from './graphGeometry';
import type { GraphViewport } from './graphTypes';

export const GRAPH_VIEWPORT_STATE_VERSION = 1 as const;

export type PersistedGraphViewport = {
  version: typeof GRAPH_VIEWPORT_STATE_VERSION;
  x: number;
  y: number;
  zoom: number;
};

const isFiniteNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

export const serializeGraphViewport = (viewport: GraphViewport): PersistedGraphViewport => ({
  version: GRAPH_VIEWPORT_STATE_VERSION,
  x: viewport.x,
  y: viewport.y,
  zoom: viewport.zoom,
});

export const parseGraphViewport = (value: unknown, minimumZoom = 0.35, maximumZoom = 1.8): GraphViewport | null => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;

  const candidate = value as Partial<PersistedGraphViewport>;
  if (candidate.version !== GRAPH_VIEWPORT_STATE_VERSION) return null;
  if (!isFiniteNumber(candidate.x) || !isFiniteNumber(candidate.y) || !isFiniteNumber(candidate.zoom)) return null;
  if (candidate.zoom <= 0) return null;

  return {
    x: candidate.x,
    y: candidate.y,
    zoom: clampGraphZoom(candidate.zoom, minimumZoom, maximumZoom),
  };
};

export const restoreGraphViewport = (
  value: unknown,
  fallback: GraphViewport,
  minimumZoom = 0.35,
  maximumZoom = 1.8,
): GraphViewport => parseGraphViewport(value, minimumZoom, maximumZoom) ?? { ...fallback };

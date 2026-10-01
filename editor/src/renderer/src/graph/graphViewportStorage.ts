import type { GraphViewport } from './graphTypes';
import {
  DEFAULT_GRAPH_VIEWPORT,
  persistGraphViewport,
  restoreGraphViewport,
} from './graphViewportState';

export interface GraphViewportStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

const KEY_PREFIX = 'arc.graph.viewport.v1';

export const graphViewportStorageKey = (domain: string, documentId: string) =>
  `${KEY_PREFIX}:${encodeURIComponent(domain)}:${encodeURIComponent(documentId)}`;

export const loadGraphViewport = (
  storage: GraphViewportStorage,
  domain: string,
  documentId: string,
): GraphViewport => {
  try {
    const serialized = storage.getItem(graphViewportStorageKey(domain, documentId));
    if (serialized === null) return { ...DEFAULT_GRAPH_VIEWPORT };
    return restoreGraphViewport(JSON.parse(serialized));
  } catch {
    return { ...DEFAULT_GRAPH_VIEWPORT };
  }
};

export const saveGraphViewport = (
  storage: GraphViewportStorage,
  domain: string,
  documentId: string,
  viewport: GraphViewport,
) => {
  storage.setItem(
    graphViewportStorageKey(domain, documentId),
    JSON.stringify(persistGraphViewport(viewport)),
  );
};

export const clearGraphViewport = (
  storage: GraphViewportStorage,
  domain: string,
  documentId: string,
) => storage.removeItem(graphViewportStorageKey(domain, documentId));

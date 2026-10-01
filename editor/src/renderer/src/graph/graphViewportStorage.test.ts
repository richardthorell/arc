import { describe, expect, it } from 'vitest';
import { DEFAULT_GRAPH_VIEWPORT } from './graphViewportState';
import {
  clearGraphViewport,
  graphViewportStorageKey,
  loadGraphViewport,
  saveGraphViewport,
  type GraphViewportStorage,
} from './graphViewportStorage';

const createStorage = (): GraphViewportStorage & { values: Map<string, string> } => {
  const values = new Map<string, string>();
  return {
    values,
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => { values.set(key, value); },
    removeItem: (key) => { values.delete(key); },
  };
};

describe('graphViewportStorage', () => {
  it('isolates persisted viewports by graph domain and document', () => {
    const storage = createStorage();
    saveGraphViewport(storage, 'material', 'asset/a', { x: 10, y: 20, zoom: 1.25 });
    saveGraphViewport(storage, 'flow', 'asset/a', { x: -5, y: 4, zoom: 0.75 });

    expect(loadGraphViewport(storage, 'material', 'asset/a')).toEqual({ x: 10, y: 20, zoom: 1.25 });
    expect(loadGraphViewport(storage, 'flow', 'asset/a')).toEqual({ x: -5, y: 4, zoom: 0.75 });
    expect(loadGraphViewport(storage, 'material', 'asset/b')).toEqual(DEFAULT_GRAPH_VIEWPORT);
  });

  it('uses stable escaped keys for arbitrary document identities', () => {
    expect(graphViewportStorageKey('material graph', 'folder/a:b')).toBe(
      'arc.graph.viewport.v1:material%20graph:folder%2Fa%3Ab',
    );
  });

  it('falls back safely when persisted JSON is malformed', () => {
    const storage = createStorage();
    storage.values.set(graphViewportStorageKey('material', 'broken'), '{not json');
    expect(loadGraphViewport(storage, 'material', 'broken')).toEqual(DEFAULT_GRAPH_VIEWPORT);
  });

  it('clears only the requested graph viewport', () => {
    const storage = createStorage();
    saveGraphViewport(storage, 'material', 'a', { x: 1, y: 2, zoom: 1 });
    saveGraphViewport(storage, 'material', 'b', { x: 3, y: 4, zoom: 1 });

    clearGraphViewport(storage, 'material', 'a');
    expect(loadGraphViewport(storage, 'material', 'a')).toEqual(DEFAULT_GRAPH_VIEWPORT);
    expect(loadGraphViewport(storage, 'material', 'b')).toEqual({ x: 3, y: 4, zoom: 1 });
  });
});

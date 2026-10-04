import { describe, expect, it } from 'vitest';

import {
  GRAPH_VIEWPORT_STATE_VERSION,
  parseGraphViewport,
  restoreGraphViewport,
  serializeGraphViewport,
} from './graphViewportPersistence';

describe('graph viewport persistence', () => {
  it('serializes a versioned domain-neutral viewport', () => {
    expect(serializeGraphViewport({ x: 120, y: -40, zoom: 1.25 })).toEqual({
      version: GRAPH_VIEWPORT_STATE_VERSION,
      x: 120,
      y: -40,
      zoom: 1.25,
    });
  });

  it('restores valid persisted viewport state', () => {
    expect(
      parseGraphViewport({
        version: GRAPH_VIEWPORT_STATE_VERSION,
        x: -32,
        y: 48,
        zoom: 0.8,
      }),
    ).toEqual({ x: -32, y: 48, zoom: 0.8 });
  });

  it('clamps persisted zoom to the shared navigation range', () => {
    expect(parseGraphViewport({ version: GRAPH_VIEWPORT_STATE_VERSION, x: 0, y: 0, zoom: 9 })).toEqual({
      x: 0,
      y: 0,
      zoom: 1.8,
    });
  });

  it.each([
    null,
    [],
    {},
    { version: 2, x: 0, y: 0, zoom: 1 },
    { version: GRAPH_VIEWPORT_STATE_VERSION, x: Number.NaN, y: 0, zoom: 1 },
    { version: GRAPH_VIEWPORT_STATE_VERSION, x: 0, y: Number.POSITIVE_INFINITY, zoom: 1 },
    { version: GRAPH_VIEWPORT_STATE_VERSION, x: 0, y: 0, zoom: 0 },
  ])('rejects malformed or unsupported state %#', (value) => {
    expect(parseGraphViewport(value)).toBeNull();
  });

  it('returns an isolated fallback when persisted state cannot be restored', () => {
    const fallback = { x: 10, y: 20, zoom: 1 };
    const restored = restoreGraphViewport({ version: 99 }, fallback);

    expect(restored).toEqual(fallback);
    expect(restored).not.toBe(fallback);
  });
});

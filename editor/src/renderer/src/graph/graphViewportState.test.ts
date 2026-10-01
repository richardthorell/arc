import { describe, expect, it } from 'vitest';
import {
  DEFAULT_GRAPH_VIEWPORT,
  persistGraphViewport,
  restoreGraphViewport,
  sameGraphViewport,
} from './graphViewportState';

describe('graphViewportState', () => {
  it('round trips a valid viewport', () => {
    const viewport = { x: -125.5, y: 42, zoom: 1.25 };
    expect(restoreGraphViewport(persistGraphViewport(viewport))).toEqual(viewport);
  });

  it('clamps persisted and restored zoom to shared navigation limits', () => {
    expect(persistGraphViewport({ x: 0, y: 0, zoom: 10 }).zoom).toBe(1.8);
    expect(restoreGraphViewport({ version: 1, x: 0, y: 0, zoom: 0.01 }).zoom).toBe(0.35);
  });

  it('falls back safely for stale or malformed persisted state', () => {
    expect(restoreGraphViewport(undefined)).toEqual(DEFAULT_GRAPH_VIEWPORT);
    expect(restoreGraphViewport({ version: 2, x: 1, y: 2, zoom: 1 })).toEqual(DEFAULT_GRAPH_VIEWPORT);
    expect(restoreGraphViewport({ version: 1, x: Number.NaN, y: 2, zoom: 1 })).toEqual(DEFAULT_GRAPH_VIEWPORT);
  });

  it('normalizes invalid live values before persistence', () => {
    expect(persistGraphViewport({ x: Number.POSITIVE_INFINITY, y: Number.NaN, zoom: Number.NaN })).toEqual({
      version: 1,
      ...DEFAULT_GRAPH_VIEWPORT,
    });
  });

  it('compares viewports with a small tolerance', () => {
    expect(sameGraphViewport({ x: 1, y: 2, zoom: 1 }, { x: 1.0005, y: 2, zoom: 1 })).toBe(true);
    expect(sameGraphViewport({ x: 1, y: 2, zoom: 1 }, { x: 1.01, y: 2, zoom: 1 })).toBe(false);
  });
});

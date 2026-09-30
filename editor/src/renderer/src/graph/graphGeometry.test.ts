import { describe, expect, it } from 'vitest';

import {
  clampGraphZoom,
  clientToGraphPoint,
  graphConnectionPath,
  graphPinKey,
  graphSelectionBounds,
  graphSelectionScreenRect,
  graphViewportFitBounds,
  graphViewportZoomAt,
  sameGraphPointMaps,
} from './graphGeometry';

describe('graph geometry', () => {
  it('maps client coordinates into graph space for any graph domain', () => {
    expect(clientToGraphPoint({ left: 100, top: 50 }, { x: 20, y: 10, zoom: 2 }, 160, 100)).toEqual([20, 20]);
  });

  it('builds stable pin keys and bezier paths', () => {
    expect(graphPinKey('event', 'then', true)).toBe('event:output:then');
    expect(graphConnectionPath([10, 20], [110, 60])).toBe('M 10 20 C 65 20, 55 60, 110 60');
  });

  it('scales the minimum bezier handle for screen-space interaction overlays', () => {
    expect(graphConnectionPath([0, 0], [10, 20], 27.5)).toBe('M 0 0 C 27.5 0, -17.5 20, 10 20');
  });

  it('clamps zoom and converts box selection to graph and screen bounds', () => {
    expect(clampGraphZoom(0.1)).toBe(0.35);
    expect(clampGraphZoom(3)).toBe(1.8);
    const selection = { start: [30, 50] as [number, number], current: [10, 20] as [number, number] };
    expect(graphSelectionBounds(selection)).toEqual({ left: 10, top: 20, right: 30, bottom: 50 });
    expect(graphSelectionScreenRect(selection, { x: 5, y: 7, zoom: 2 })).toEqual({
      left: 25,
      top: 47,
      width: 40,
      height: 60,
    });
  });

  it('zooms around a stable screen-space anchor', () => {
    const next = graphViewportZoomAt({ x: 20, y: 10, zoom: 1 }, [120, 60], 2);
    expect(next).toEqual({ x: -80, y: -40, zoom: 2 });
    expect([(120 - next.x) / next.zoom, (60 - next.y) / next.zoom]).toEqual([100, 50]);
  });

  it('clamps anchored zoom without shifting the graph point under the cursor', () => {
    const next = graphViewportZoomAt({ x: 0, y: 0, zoom: 1 }, [100, 50], 10);
    expect(next.zoom).toBe(1.8);
    expect([(100 - next.x) / next.zoom, (50 - next.y) / next.zoom]).toEqual([100, 50]);
  });

  it('fits graph bounds into the viewport with shared padding and zoom limits', () => {
    expect(
      graphViewportFitBounds({ left: 0, top: 0, right: 400, bottom: 200 }, { width: 1000, height: 600 }, 50),
    ).toEqual({
      x: 140,
      y: 120,
      zoom: 1.8,
    });

    expect(
      graphViewportFitBounds({ left: 0, top: 0, right: 2000, bottom: 1000 }, { width: 1000, height: 600 }, 50),
    ).toEqual({
      x: 50,
      y: 75,
      zoom: 0.45,
    });
  });

  it('centers degenerate fit bounds deterministically', () => {
    expect(graphViewportFitBounds({ left: 20, top: 30, right: 20, bottom: 30 }, { width: 800, height: 600 })).toEqual({
      x: 364,
      y: 246,
      zoom: 1.8,
    });
  });

  it('compares measured point maps with sub-pixel tolerance', () => {
    expect(sameGraphPointMaps(new Map([['a', [1, 2]]]), new Map([['a', [1.005, 2.005]]]))).toBe(true);
    expect(sameGraphPointMaps(new Map([['a', [1, 2]]]), new Map([['a', [2, 2]]]))).toBe(false);
  });
});

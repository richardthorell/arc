import { describe, expect, it } from 'vitest';

import {
  clampGraphZoom,
  clientToGraphPoint,
  graphConnectionPath,
  graphPinKey,
  graphSelectionBounds,
  graphSelectionScreenRect,
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

  it('compares measured point maps with sub-pixel tolerance', () => {
    expect(sameGraphPointMaps(new Map([['a', [1, 2]]]), new Map([['a', [1.005, 2.005]]]))).toBe(true);
    expect(sameGraphPointMaps(new Map([['a', [1, 2]]]), new Map([['a', [2, 2]]]))).toBe(false);
  });
});

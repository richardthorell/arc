import { describe, expect, it } from 'vitest';

import {
  normalizeGraphViewport,
  panGraphViewport,
  serializeGraphViewport,
  zoomGraphViewportAt,
} from './graphViewport';

describe('graphViewport', () => {
  it('normalizes invalid persisted values and clamps zoom', () => {
    expect(normalizeGraphViewport({ x: Number.NaN, y: 12, zoom: 99 })).toEqual({
      x: 0,
      y: 12,
      zoom: 4,
    });
  });

  it('serializes a deterministic JSON-safe snapshot', () => {
    const snapshot = serializeGraphViewport({ x: 18.5, y: -4, zoom: 0.01 });
    expect(snapshot).toEqual({ x: 18.5, y: -4, zoom: 0.1 });
    expect(JSON.parse(JSON.stringify(snapshot))).toEqual(snapshot);
  });

  it('keeps the graph point under the zoom anchor stable', () => {
    const before = { x: 20, y: 30, zoom: 1 };
    const anchor = { x: 120, y: 80 };
    const after = zoomGraphViewportAt(before, 2, anchor);

    const beforePoint = {
      x: (anchor.x - before.x) / before.zoom,
      y: (anchor.y - before.y) / before.zoom,
    };
    const afterPoint = {
      x: (anchor.x - after.x) / after.zoom,
      y: (anchor.y - after.y) / after.zoom,
    };

    expect(afterPoint).toEqual(beforePoint);
    expect(after.zoom).toBe(2);
  });

  it('applies screen-space pan without changing zoom', () => {
    expect(panGraphViewport({ x: 10, y: -5, zoom: 2 }, { x: 7, y: 3 })).toEqual({
      x: 17,
      y: -2,
      zoom: 2,
    });
  });

  it('honors domain-provided zoom limits', () => {
    expect(
      normalizeGraphViewport({ x: 0, y: 0, zoom: 3 }, undefined, {
        minZoom: 0.5,
        maxZoom: 2,
      }),
    ).toEqual({ x: 0, y: 0, zoom: 2 });
  });
});

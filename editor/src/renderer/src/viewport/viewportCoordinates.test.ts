import { describe, expect, it } from 'vitest';
import { toViewportPixels } from './viewportCoordinates';

describe('viewport coordinate mapping', () => {
  it('maps CSS pixels to the physical render extent', () => {
    expect(
      toViewportPixels({
        clientX: 250,
        clientY: 125,
        left: 50,
        top: 25,
        cssWidth: 400,
        cssHeight: 200,
        renderWidth: 800,
        renderHeight: 400,
        devicePixelRatio: 2,
      }),
    ).toEqual({ x: 400, y: 200 });
  });

  it('clamps pointer coordinates to the rendered image', () => {
    expect(
      toViewportPixels({
        clientX: 900,
        clientY: -20,
        left: 100,
        top: 100,
        cssWidth: 300,
        cssHeight: 200,
        renderWidth: 600,
        renderHeight: 400,
        devicePixelRatio: 2,
      }),
    ).toEqual({ x: 599, y: 0 });
  });
});

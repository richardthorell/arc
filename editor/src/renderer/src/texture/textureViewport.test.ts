import { describe, expect, it } from 'vitest';

import { calculateTextureFitZoom } from './textureViewport';

const fit = (viewportWidth: number, viewportHeight: number, textureWidth: number, textureHeight: number) =>
  calculateTextureFitZoom({
    viewport: { width: viewportWidth, height: viewportHeight },
    texture: { width: textureWidth, height: textureHeight },
    padding: 28,
    minZoom: 0.05,
    maxZoom: 16,
  });

describe('texture viewport fit zoom', () => {
  it('fits wide textures by width while preserving padding', () => {
    expect(fit(1000, 800, 2000, 1000)).toBeCloseTo(0.472);
  });

  it('fits tall textures by height while preserving padding', () => {
    expect(fit(1000, 800, 500, 2000)).toBeCloseTo(0.372);
  });

  it('allows small textures to scale up to the available viewport', () => {
    expect(fit(1000, 800, 100, 100)).toBeCloseTo(7.44);
  });

  it('clamps extreme fits to the configured zoom range', () => {
    expect(fit(1000, 800, 1, 1)).toBe(16);
    expect(fit(100, 100, 100000, 100000)).toBe(0.05);
  });

  it('uses the actual displayed mip dimensions', () => {
    expect(fit(1056, 1056, 512, 512)).toBe(1.953125);
  });

  it('rejects invalid viewport or texture dimensions', () => {
    expect(fit(0, 800, 512, 512)).toBeNull();
    expect(fit(1000, 800, 0, 512)).toBeNull();
    expect(fit(1000, 800, Number.NaN, 512)).toBeNull();
  });
});

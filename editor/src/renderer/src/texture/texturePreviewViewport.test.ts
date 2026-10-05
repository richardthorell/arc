import { describe, expect, it } from 'vitest';

import {
  TEXTURE_PREVIEW_MAX_ZOOM,
  TEXTURE_PREVIEW_MIN_ZOOM,
  clampTexturePreviewZoom,
  fitTexturePreviewZoom,
  zoomTexturePreviewAroundPoint,
} from './texturePreviewViewport';

describe('texture preview viewport math', () => {
  it('fits wide and tall textures while preserving aspect ratio', () => {
    expect(fitTexturePreviewZoom({ width: 2000, height: 1000 }, { width: 1000, height: 800 })).toBe(0.5);
    expect(fitTexturePreviewZoom({ width: 1000, height: 2000 }, { width: 800, height: 1000 })).toBe(0.5);
  });

  it('accounts for preview padding', () => {
    expect(fitTexturePreviewZoom({ width: 100, height: 100 }, { width: 140, height: 140 }, 20)).toBe(1);
  });

  it('never persists invalid zoom from transient geometry', () => {
    expect(fitTexturePreviewZoom({ width: 0, height: 100 }, { width: 100, height: 100 })).toBe(1);
    expect(fitTexturePreviewZoom({ width: 100, height: 100 }, { width: Number.NaN, height: 100 })).toBe(1);
    expect(clampTexturePreviewZoom(Number.POSITIVE_INFINITY)).toBe(1);
  });

  it('clamps extreme fit results to supported zoom bounds', () => {
    expect(fitTexturePreviewZoom({ width: 1, height: 1 }, { width: 10000, height: 10000 })).toBe(
      TEXTURE_PREVIEW_MAX_ZOOM,
    );
    expect(fitTexturePreviewZoom({ width: 100000, height: 100000 }, { width: 1, height: 1 })).toBe(
      TEXTURE_PREVIEW_MIN_ZOOM,
    );
  });

  it('keeps the cursor anchor stable while zooming', () => {
    expect(zoomTexturePreviewAroundPoint(1, 2, { x: 10, y: 20 }, { x: 50, y: 60 })).toEqual({
      zoom: 2,
      pan: { x: -30, y: -20 },
    });
  });
});

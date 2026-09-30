import { describe, expect, it } from 'vitest';
import {
  TEXTURE_PREVIEW_MAX_ZOOM,
  TEXTURE_PREVIEW_MIN_ZOOM,
  fitTexturePreview,
  zoomTexturePreviewAroundPoint,
} from './texturePreviewGeometry';

describe('texture preview geometry', () => {
  it('fits large textures inside the viewport with padding', () => {
    expect(fitTexturePreview({ width: 4096, height: 2048 }, { width: 1024, height: 768 }, 24)).toEqual({
      zoom: 976 / 4096,
      pan: { x: 0, y: 0 },
    });
  });

  it('clamps tiny textures to the maximum zoom', () => {
    expect(fitTexturePreview({ width: 1, height: 1 }, { width: 4096, height: 4096 }).zoom).toBe(
      TEXTURE_PREVIEW_MAX_ZOOM,
    );
  });

  it('returns a safe default for invalid dimensions', () => {
    expect(fitTexturePreview({ width: 0, height: 512 }, { width: 800, height: 600 })).toEqual({
      zoom: 1,
      pan: { x: 0, y: 0 },
    });
  });

  it('keeps the preview point under the zoom anchor stable', () => {
    expect(zoomTexturePreviewAroundPoint(1, 2, { x: 10, y: 20 }, { x: 100, y: 80 })).toEqual({
      zoom: 2,
      pan: { x: -80, y: -40 },
    });
  });

  it('clamps anchored zoom to the shared limits', () => {
    expect(zoomTexturePreviewAroundPoint(1, 0, { x: 0, y: 0 }, { x: 0, y: 0 }).zoom).toBe(TEXTURE_PREVIEW_MIN_ZOOM);
    expect(zoomTexturePreviewAroundPoint(1, 100, { x: 0, y: 0 }, { x: 0, y: 0 }).zoom).toBe(TEXTURE_PREVIEW_MAX_ZOOM);
  });
});

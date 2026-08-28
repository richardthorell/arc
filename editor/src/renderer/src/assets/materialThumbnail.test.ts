import { describe, expect, it } from 'vitest';

import { materialThumbnailCacheKey, materialThumbnailViewportId, transparentPreviewPixels } from './materialThumbnail';

describe('material thumbnails', () => {
  it('uses the production material preview viewport identity with an isolated thumbnail instance', () => {
    expect(materialThumbnailViewportId('1234-abcd', 7)).toBe('asset-preview-material-1234-abcd~thumbnail-7');
    expect(materialThumbnailCacheKey('1234-abcd', 4, 128)).toBe('1234-abcd:4:128');
  });

  it('removes only clear-color pixels connected to the image edge', () => {
    const width = 5;
    const height = 5;
    const pixels = new Uint8ClampedArray(width * height * 4);
    for (let index = 0; index < width * height; index += 1) {
      const offset = index * 4;
      pixels[offset] = 20;
      pixels[offset + 1] = 24;
      pixels[offset + 2] = 28;
      pixels[offset + 3] = 255;
    }

    const center = (2 * width + 2) * 4;
    pixels[center] = 180;
    pixels[center + 1] = 80;
    pixels[center + 2] = 40;
    // Same clear color enclosed by the sphere must not be erased by the edge flood fill.
    const enclosed = (2 * width + 1) * 4;
    pixels[enclosed] = 20;
    pixels[enclosed + 1] = 24;
    pixels[enclosed + 2] = 28;
    // Surround the enclosed pixel with material-colored pixels.
    for (const index of [6, 7, 8, 11, 13, 16, 17, 18]) {
      const offset = index * 4;
      pixels[offset] = 180;
      pixels[offset + 1] = 80;
      pixels[offset + 2] = 40;
    }

    const result = transparentPreviewPixels(pixels, width, height);
    expect(result[3]).toBe(0);
    expect(result[(width - 1) * 4 + 3]).toBe(0);
    expect(result[center + 3]).toBe(255);
    expect(result[enclosed + 3]).toBe(255);
  });
});

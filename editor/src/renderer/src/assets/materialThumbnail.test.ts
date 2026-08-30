import { describe, expect, it } from 'vitest';

import {
  maskMaterialSpherePixels,
  materialThumbnailCacheKey,
  materialThumbnailViewportId,
  opaquePixelBounds,
  transparentPreviewPixels,
} from './materialThumbnail';

describe('material thumbnails', () => {
  it('keeps stable cache and diagnostic viewport identities', () => {
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

    const centerIndex = 12;
    const center = centerIndex * 4;
    pixels[center] = 180;
    pixels[center + 1] = 80;
    pixels[center + 2] = 40;

    // Same clear color enclosed by material-colored pixels must not be erased by the edge flood fill.
    const enclosedIndex = 11;
    const enclosed = enclosedIndex * 4;
    for (const index of [6, 7, 10, 12, 16, 17]) {
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

  it('uses the known material sphere geometry to remove a graded studio background', () => {
    const width = 10;
    const height = 10;
    const pixels = new Uint8ClampedArray(width * height * 4);
    for (let index = 0; index < width * height; index += 1) {
      const offset = index * 4;
      pixels[offset] = index;
      pixels[offset + 1] = index + 10;
      pixels[offset + 2] = index + 20;
      pixels[offset + 3] = 255;
    }

    const result = maskMaterialSpherePixels(pixels, width, height);
    expect(result[3]).toBe(0);
    expect(result[(width * height - 1) * 4 + 3]).toBe(0);
    expect(result[(5 * width + 5) * 4 + 3]).toBe(255);
  });

  it('finds the rendered object bounds independently of its source-frame position', () => {
    const width = 8;
    const height = 6;
    const pixels = new Uint8ClampedArray(width * height * 4);
    for (let y = 3; y <= 4; y += 1) {
      for (let x = 5; x <= 7; x += 1) pixels[(y * width + x) * 4 + 3] = 255;
    }

    expect(opaquePixelBounds(pixels, width, height)).toEqual({ x: 5, y: 3, width: 3, height: 2 });
  });
});

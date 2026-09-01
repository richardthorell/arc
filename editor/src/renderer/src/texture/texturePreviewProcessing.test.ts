import { describe, expect, it } from 'vitest';
import { processTexturePixel } from './texturePreviewProcessing';
import type { TextureSettingsSnapshot } from './textureSettings';

const settings: TextureSettingsSnapshot = {
  settingsVersion: 5,
  preset: 'color',
  semantic: 'base_color',
  colorSpace: 'linear',
  streamingMode: 'streamed_mips',
  compression: 'color',
  powerOfTwo: 'preserve',
  minFilter: 'linear',
  magFilter: 'linear',
  mipFilter: 'linear',
  wrapU: 'repeat',
  wrapV: 'repeat',
  mipGenerationFilter: 'box',
  maxSize: 8192,
  anisotropy: 8,
  lodBias: 0,
  minimumLod: 0,
  maximumLod: 1000,
  alphaCoverageThreshold: 0.5,
  brightness: 0,
  gamma: 1,
  contrast: 1,
  saturation: 1,
  vibrance: 0,
  tintR: 1,
  tintG: 1,
  tintB: 1,
  inputBlack: 0,
  inputWhite: 1,
  outputBlack: 0,
  outputWhite: 1,
  channelR: 'red',
  channelG: 'green',
  channelB: 'blue',
  channelA: 'alpha',
  invertR: false,
  invertG: false,
  invertB: false,
  invertA: false,
  generateMips: true,
  preserveAlphaCoverage: false,
};

describe('texture Stage 3 preview processing', () => {
  it('preserves pixels with neutral settings', () => {
    expect(processTexturePixel([0.2, 0.4, 0.6, 0.8], settings)).toEqual([0.2, 0.4, 0.6, 0.8]);
  });
  it('supports channel remapping and inversion', () => {
    const value = processTexturePixel([0.2, 0.4, 0.6, 0.8], {
      ...settings,
      channelR: 'blue',
      channelG: 'zero',
      channelB: 'one',
      invertA: true,
    });
    expect(value[0]).toBeCloseTo(0.6);
    expect(value[1]).toBe(0);
    expect(value[2]).toBe(1);
    expect(value[3]).toBeCloseTo(0.2);
  });
  it('applies levels and brightness deterministically', () => {
    const value = processTexturePixel([0.5, 0.5, 0.5, 1], {
      ...settings,
      brightness: 1,
      inputBlack: 0.25,
      inputWhite: 0.75,
    });
    expect(value[0]).toBe(1);
    expect(value[1]).toBe(1);
    expect(value[2]).toBe(1);
  });
});

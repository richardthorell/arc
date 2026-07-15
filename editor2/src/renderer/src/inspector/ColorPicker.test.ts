import { describe, expect, it } from 'vitest';

import {
  colorToHex,
  hexToLinearColor,
  hsvToLinearColor,
  linearColorToHsv,
  linearToSrgb,
  srgbToLinear,
} from './ColorPicker';

describe('ARC color picker conversions', () => {
  it('round trips scene-linear channels through sRGB', () => {
    for (const channel of [0, 0.003, 0.055, 0.18, 0.5, 1]) {
      expect(srgbToLinear(linearToSrgb(channel))).toBeCloseTo(channel, 6);
    }
  });

  it('round trips linear colors through artist-facing HSV', () => {
    const original = { x: 0.055, y: 0.22, z: 0.71, w: 0.42 };
    const hsv = linearColorToHsv(original);
    const restored = hsvToLinearColor(hsv, original.w);
    expect(restored.x).toBeCloseTo(original.x, 6);
    expect(restored.y).toBeCloseTo(original.y, 6);
    expect(restored.z).toBeCloseTo(original.z, 6);
    expect(restored.w).toBe(original.w);
  });

  it('parses and writes sRGB hexadecimal values with alpha', () => {
    const color = hexToLinearColor('#FF800080');
    expect(color).not.toBeNull();
    expect(color?.x).toBeCloseTo(1, 6);
    expect(color?.y).toBeCloseTo(srgbToLinear(128 / 255), 6);
    expect(color?.z).toBe(0);
    expect(color?.w).toBeCloseTo(128 / 255, 6);
    expect(colorToHex(color!)).toBe('#FF800080');
    expect(hexToLinearColor('not-a-color')).toBeNull();
  });
});

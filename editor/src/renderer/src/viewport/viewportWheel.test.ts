import { describe, expect, it } from 'vitest';

import { normalizeViewportWheel } from './viewportWheel';

describe('normalizeViewportWheel', () => {
  it('keeps wheel directions distinct', () => {
    expect(normalizeViewportWheel(-100)).toBeGreaterThan(0);
    expect(normalizeViewportWheel(100)).toBeLessThan(0);
  });

  it('normalizes common Chromium mouse-wheel deltas close to one camera step', () => {
    expect(normalizeViewportWheel(-100)).toBe(1);
    expect(normalizeViewportWheel(100)).toBe(-1);
    expect(normalizeViewportWheel(-120)).toBeCloseTo(1.2);
    expect(normalizeViewportWheel(120)).toBeCloseTo(-1.2);
  });

  it('keeps high-resolution wheel and trackpad events useful', () => {
    expect(normalizeViewportWheel(-1)).toBe(0.2);
    expect(normalizeViewportWheel(1)).toBe(-0.2);
  });

  it('clamps extreme events and ignores invalid input', () => {
    expect(normalizeViewportWheel(-100000)).toBe(4);
    expect(normalizeViewportWheel(100000)).toBe(-4);
    expect(normalizeViewportWheel(0)).toBe(0);
    expect(normalizeViewportWheel(Number.NaN)).toBe(0);
  });
});

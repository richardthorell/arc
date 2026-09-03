import { describe, expect, it } from 'vitest';
import { evaluateUiCurve, type UiCurvePoint } from './UiCurveEditor';
const identity: UiCurvePoint[] = [
  { x: 0, y: 0, inTangent: 1, outTangent: 1, interpolation: 'smooth' },
  { x: 1, y: 1, inTangent: 1, outTangent: 1, interpolation: 'smooth' },
];
describe('UiCurveEditor evaluation', () => {
  it('keeps identity values stable', () => {
    expect(evaluateUiCurve(identity, 0.25)).toBeCloseTo(0.25);
    expect(evaluateUiCurve(identity, 0.75)).toBeCloseTo(0.75);
  });
  it('supports constant and linear segments', () => {
    const points = [
      { ...identity[0], interpolation: 'constant' as const },
      { x: 0.5, y: 0.8, inTangent: 1, outTangent: 1, interpolation: 'linear' as const },
      identity[1],
    ];
    expect(evaluateUiCurve(points, 0.25)).toBe(0);
    expect(evaluateUiCurve(points, 0.75)).toBeCloseTo(0.9);
  });
});

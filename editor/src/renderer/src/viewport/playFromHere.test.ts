import { describe, expect, it } from 'vitest';

import { createPlayFromHereOverride } from './playFromHere';

describe('Play From Here override', () => {
  it('captures a deterministic transient camera transform', () => {
    const position: [number, number, number] = [1, 2, 3];
    const rotation: [number, number, number, number] = [0, 0, 0, 1];

    const override = createPlayFromHereOverride('camera', { position, rotation });
    position[0] = 99;
    rotation[3] = 0;

    expect(override).toEqual({
      version: 1,
      source: 'camera',
      transform: { position: [1, 2, 3], rotation: [0, 0, 0, 1] },
    });
  });

  it('preserves cursor provenance without changing transform semantics', () => {
    expect(
      createPlayFromHereOverride('cursor', {
        position: [-4, 0.5, 12],
        rotation: [0, 1, 0, 0],
      }),
    ).toMatchObject({ version: 1, source: 'cursor' });
  });

  it.each([
    { position: [Number.NaN, 0, 0] as [number, number, number], rotation: [0, 0, 0, 1] as [number, number, number, number] },
    { position: [0, 0, 0] as [number, number, number], rotation: [0, Number.POSITIVE_INFINITY, 0, 1] as [number, number, number, number] },
  ])('rejects non-finite spawn transforms', ({ position, rotation }) => {
    expect(() => createPlayFromHereOverride('camera', { position, rotation })).toThrow(/finite values/);
  });
});

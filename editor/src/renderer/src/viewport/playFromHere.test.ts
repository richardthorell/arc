import { describe, expect, it } from 'vitest';

import { createPlayFromHereOverride, PlayFromHereOverrideQueue } from './playFromHere';

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
    {
      position: [Number.NaN, 0, 0] as [number, number, number],
      rotation: [0, 0, 0, 1] as [number, number, number, number],
    },
    {
      position: [0, 0, 0] as [number, number, number],
      rotation: [0, Number.POSITIVE_INFINITY, 0, 1] as [number, number, number, number],
    },
  ])('rejects non-finite spawn transforms', ({ position, rotation }) => {
    expect(() => createPlayFromHereOverride('camera', { position, rotation })).toThrow(/finite values/);
  });

  it('consumes a staged override exactly once', () => {
    const queue = new PlayFromHereOverrideQueue();
    queue.stage(
      createPlayFromHereOverride('camera', {
        position: [3, 4, 5],
        rotation: [0, 0, 0, 1],
      }),
    );

    expect(queue.hasPendingOverride).toBe(true);
    expect(queue.consume(true)).toMatchObject({ source: 'camera', transform: { position: [3, 4, 5] } });
    expect(queue.hasPendingOverride).toBe(false);
    expect(queue.consume(true)).toBeUndefined();
  });

  it('clears the override when a project opts out', () => {
    const queue = new PlayFromHereOverrideQueue();
    queue.stage(
      createPlayFromHereOverride('cursor', {
        position: [8, 0, -2],
        rotation: [0, 0, 0, 1],
      }),
    );

    expect(queue.consume(false)).toBeUndefined();
    expect(queue.hasPendingOverride).toBe(false);
    expect(queue.consume(true)).toBeUndefined();
  });

  it('detaches staged and consumed values from caller mutation', () => {
    const queue = new PlayFromHereOverrideQueue();
    const override = createPlayFromHereOverride('camera', {
      position: [1, 2, 3],
      rotation: [0, 0, 0, 1],
    });
    queue.stage(override);
    (override.transform.position as [number, number, number])[0] = 99;

    const consumed = queue.consume(true)!;
    expect(consumed.transform.position).toEqual([1, 2, 3]);
    (consumed.transform.position as [number, number, number])[1] = 77;

    expect(queue.hasPendingOverride).toBe(false);
  });

  it('supports explicit cancellation before Play starts', () => {
    const queue = new PlayFromHereOverrideQueue();
    queue.stage(
      createPlayFromHereOverride('camera', {
        position: [0, 1, 2],
        rotation: [0, 0, 0, 1],
      }),
    );

    queue.clear();

    expect(queue.hasPendingOverride).toBe(false);
    expect(queue.consume(true)).toBeUndefined();
  });
});

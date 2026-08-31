import { describe, expect, it } from 'vitest';

import { viewportFlyMovement } from './viewportFlyNavigation';

describe('viewportFlyMovement', () => {
  it('maps WASD and QE to right, forward, and world-up movement', () => {
    expect(viewportFlyMovement(new Set(['KeyW']), 4)).toEqual({ moveRight: 0, moveUp: 0, moveForward: 4 });
    expect(viewportFlyMovement(new Set(['KeyS']), 4)).toEqual({ moveRight: 0, moveUp: 0, moveForward: -4 });
    expect(viewportFlyMovement(new Set(['KeyD']), 4)).toEqual({ moveRight: 4, moveUp: 0, moveForward: 0 });
    expect(viewportFlyMovement(new Set(['KeyA']), 4)).toEqual({ moveRight: -4, moveUp: 0, moveForward: 0 });
    expect(viewportFlyMovement(new Set(['KeyE']), 4)).toEqual({ moveRight: 0, moveUp: 4, moveForward: 0 });
    expect(viewportFlyMovement(new Set(['KeyQ']), 4)).toEqual({ moveRight: 0, moveUp: -4, moveForward: 0 });
  });

  it('normalizes diagonal movement so it does not move faster', () => {
    const movement = viewportFlyMovement(new Set(['KeyW', 'KeyD', 'KeyE']), 6);
    expect(movement).not.toBeNull();
    expect(Math.hypot(movement!.moveRight, movement!.moveUp, movement!.moveForward)).toBeCloseTo(6);
  });

  it('cancels opposing keys', () => {
    expect(viewportFlyMovement(new Set(['KeyW', 'KeyS']), 4)).toBeNull();
  });
});

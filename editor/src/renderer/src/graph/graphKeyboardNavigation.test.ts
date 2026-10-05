import { describe, expect, it } from 'vitest';

import { findGraphNavigationTarget, type GraphNavigationNode } from './graphKeyboardNavigation';

const nodes: GraphNavigationNode[] = [
  { id: 'center', x: 100, y: 100 },
  { id: 'left', x: 20, y: 100 },
  { id: 'right', x: 180, y: 100 },
  { id: 'up', x: 100, y: 20 },
  { id: 'down', x: 100, y: 180 },
  { id: 'diagonal', x: 140, y: 145 },
];

describe('findGraphNavigationTarget', () => {
  it('moves predictably in each visual direction', () => {
    expect(findGraphNavigationTarget(nodes, 'center', 'left')).toBe('left');
    expect(findGraphNavigationTarget(nodes, 'center', 'right')).toBe('right');
    expect(findGraphNavigationTarget(nodes, 'center', 'up')).toBe('up');
    expect(findGraphNavigationTarget(nodes, 'center', 'down')).toBe('down');
  });

  it('prefers aligned nodes over closer diagonal jumps', () => {
    const candidates: GraphNavigationNode[] = [
      { id: 'current', x: 0, y: 0 },
      { id: 'aligned', x: 60, y: 0 },
      { id: 'diagonal', x: 30, y: 20 },
    ];

    expect(findGraphNavigationTarget(candidates, 'current', 'right')).toBe('aligned');
  });

  it('uses stable ids to resolve equal-score ties', () => {
    const candidates: GraphNavigationNode[] = [
      { id: 'current', x: 0, y: 0 },
      { id: 'z-node', x: 20, y: -10 },
      { id: 'a-node', x: 20, y: 10 },
    ];

    expect(findGraphNavigationTarget(candidates, 'current', 'right')).toBe('a-node');
  });

  it('ignores invalid positions and returns null at an edge', () => {
    const candidates: GraphNavigationNode[] = [
      { id: 'current', x: 0, y: 0 },
      { id: 'invalid', x: Number.NaN, y: 0 },
      { id: 'left', x: -10, y: 0 },
    ];

    expect(findGraphNavigationTarget(candidates, 'current', 'right')).toBeNull();
    expect(findGraphNavigationTarget(candidates, 'missing', 'left')).toBeNull();
  });
});

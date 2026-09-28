import { describe, expect, it } from 'vitest';

import {
  moveGraphNodePaletteSelection,
  queryGraphNodePalette,
  resolveGraphNodePaletteSelection,
  type GraphNodePaletteDescriptor,
} from './graphNodePalette';

type Kind = 'add' | 'constant' | 'branch';

const nodes: GraphNodePaletteDescriptor<Kind>[] = [
  {
    kind: 'branch',
    name: 'Branch',
    category: 'Flow',
    keywords: ['if', 'condition'],
    isCompatible: ({ direction, type }) => direction === 'output' && type === 'exec',
  },
  {
    kind: 'constant',
    name: 'Constant',
    category: 'Values',
    keywords: ['number', 'float'],
    description: 'Create a scalar value',
    isCompatible: ({ direction, type }) => direction === 'input' && type === 'float',
  },
  {
    kind: 'add',
    name: 'Add',
    category: 'Math',
    keywords: ['sum', 'plus'],
    isCompatible: ({ type }) => type === 'float',
  },
];

describe('queryGraphNodePalette', () => {
  it('searches names, categories, keywords, and descriptions case-insensitively', () => {
    expect(queryGraphNodePalette(nodes, { search: 'PLUS' }).map((node) => node.kind)).toEqual(['add']);
    expect(queryGraphNodePalette(nodes, { search: 'scalar value' }).map((node) => node.kind)).toEqual(['constant']);
    expect(queryGraphNodePalette(nodes, { search: 'flow condition' }).map((node) => node.kind)).toEqual(['branch']);
  });

  it('filters through domain-provided pin compatibility', () => {
    expect(
      queryGraphNodePalette(nodes, { pin: { direction: 'input', type: 'float' } }).map((node) => node.kind),
    ).toEqual(['add', 'constant']);
    expect(
      queryGraphNodePalette(nodes, { pin: { direction: 'output', type: 'exec' } }).map((node) => node.kind),
    ).toEqual(['branch']);
  });

  it('returns deterministic category/name ordering', () => {
    expect(queryGraphNodePalette(nodes).map((node) => node.kind)).toEqual(['branch', 'add', 'constant']);
  });
});

describe('graph node palette keyboard selection', () => {
  const results = queryGraphNodePalette(nodes);

  it('moves next/previous with deterministic wrapping', () => {
    expect(moveGraphNodePaletteSelection(results, undefined, 'next')).toBe('branch');
    expect(moveGraphNodePaletteSelection(results, 'branch', 'next')).toBe('add');
    expect(moveGraphNodePaletteSelection(results, 'constant', 'next')).toBe('branch');
    expect(moveGraphNodePaletteSelection(results, 'branch', 'previous')).toBe('constant');
  });

  it('supports first/last navigation and empty result sets', () => {
    expect(moveGraphNodePaletteSelection(results, 'add', 'first')).toBe('branch');
    expect(moveGraphNodePaletteSelection(results, 'add', 'last')).toBe('constant');
    expect(moveGraphNodePaletteSelection([], 'add', 'next')).toBeUndefined();
  });

  it('recovers safely when filtering removes the selected kind', () => {
    const filtered = queryGraphNodePalette(nodes, { search: 'constant' });
    expect(moveGraphNodePaletteSelection(filtered, 'branch', 'next')).toBe('constant');
    expect(moveGraphNodePaletteSelection(filtered, 'branch', 'previous')).toBe('constant');
    expect(resolveGraphNodePaletteSelection(filtered, 'branch')).toBeUndefined();
    expect(resolveGraphNodePaletteSelection(filtered, 'constant')?.name).toBe('Constant');
  });
});

import { describe, expect, it } from 'vitest';

import type { GraphNodePaletteDescriptor } from './graphNodePalette';
import { buildGraphNodePalettePresentation } from './graphNodePalettePresentation';

type Kind = 'add' | 'branch' | 'comment';

const nodes: GraphNodePaletteDescriptor<Kind>[] = [
  { kind: 'comment', name: 'Comment', keywords: ['note'] },
  { kind: 'branch', name: 'Branch', category: 'Flow', isCompatible: ({ type }) => type === 'exec' },
  { kind: 'add', name: 'Add', category: 'Math', keywords: ['sum'], isCompatible: ({ type }) => type === 'float' },
];

describe('buildGraphNodePalettePresentation', () => {
  it('groups filtered results deterministically for a shared palette surface', () => {
    const presentation = buildGraphNodePalettePresentation(nodes);

    expect(presentation.resultCount).toBe(3);
    expect(presentation.groups.map(({ category }) => category)).toEqual(['Flow', 'Math', 'Other']);
    expect(presentation.groups.map(({ nodes: groupNodes }) => groupNodes.map(({ kind }) => kind))).toEqual([
      ['branch'],
      ['add'],
      ['comment'],
    ]);
    expect(presentation.selectedKind).toBe('branch');
  });

  it('uses the same search and pin-context filtering as palette activation', () => {
    const presentation = buildGraphNodePalettePresentation(nodes, {
      search: 'sum',
      pin: { direction: 'input', type: 'float' },
    });

    expect(presentation.resultCount).toBe(1);
    expect(presentation.groups[0].nodes[0].kind).toBe('add');
  });

  it('preserves a visible selection and safely falls back after filtering', () => {
    expect(buildGraphNodePalettePresentation(nodes, {}, 'add').selectedKind).toBe('add');
    expect(buildGraphNodePalettePresentation(nodes, { search: 'branch' }, 'add').selectedKind).toBe('branch');
    expect(buildGraphNodePalettePresentation(nodes, { search: 'missing' }, 'add').selectedKind).toBeUndefined();
  });
});

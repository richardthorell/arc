import { describe, expect, it } from 'vitest';

import { buildGraphNodePalettePresentation, queryGraphNodePalette } from './index';

describe('shared graph public API', () => {
  it('exports node palette query and presentation helpers together', () => {
    const descriptors = [
      { kind: 'branch', name: 'Branch', category: 'Flow' },
      { kind: 'add', name: 'Add', category: 'Math', keywords: ['sum'] },
    ] as const;

    expect(queryGraphNodePalette(descriptors, { search: 'sum' }).map(({ kind }) => kind)).toEqual(['add']);
    expect(buildGraphNodePalettePresentation(descriptors).groups.map(({ category }) => category)).toEqual([
      'Flow',
      'Math',
    ]);
  });
});

import { describe, expect, it } from 'vitest';

import { buildEditorToolbarLayout } from './editorToolbarLayout';

describe('buildEditorToolbarLayout', () => {
  it('groups actions into the shared left, center, and right sections', () => {
    const layout = buildEditorToolbarLayout([
      { id: 'save', section: 'left' as const, order: 0, value: 'Save' },
      { id: 'compile', section: 'center' as const, order: 10, value: 'Compile' },
      { id: 'view', section: 'right' as const, order: 0, value: 'View' },
      { id: 'reload', section: 'center' as const, order: 5, value: 'Reload' },
    ]);

    expect(layout.left.map((item) => item.id)).toEqual(['save']);
    expect(layout.center.map((item) => item.id)).toEqual(['reload', 'compile']);
    expect(layout.right.map((item) => item.id)).toEqual(['view']);
  });

  it('uses stable ids to make equal-order layouts deterministic', () => {
    const layout = buildEditorToolbarLayout([
      { id: 'zoom', section: 'right' as const, value: 1 },
      { id: 'fit', section: 'right' as const, value: 2 },
      { id: 'reset', section: 'right' as const, value: 3 },
    ]);

    expect(layout.right.map((item) => item.id)).toEqual(['fit', 'reset', 'zoom']);
  });

  it('rejects ambiguous action identities', () => {
    expect(() =>
      buildEditorToolbarLayout([
        { id: 'save', section: 'left' as const, value: 1 },
        { id: ' save ', section: 'right' as const, value: 2 },
      ]),
    ).toThrow('Duplicate editor toolbar item id: save');

    expect(() => buildEditorToolbarLayout([{ id: '  ', section: 'left' as const, value: 1 }])).toThrow(
      'Editor toolbar item ids must not be empty.',
    );
  });
});

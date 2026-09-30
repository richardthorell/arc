import { describe, expect, it } from 'vitest';

import { editorToolbarRegions, isEditorToolbarRegion } from './editorToolbarContract';

describe('editorToolbarContract', () => {
  it('keeps the shared editor toolbar region vocabulary deterministic', () => {
    expect(editorToolbarRegions).toEqual(['left', 'center', 'right']);
    expect(editorToolbarRegions).toEqual([...new Set(editorToolbarRegions)]);
  });

  it('recognizes only shared toolbar regions', () => {
    expect(isEditorToolbarRegion('left')).toBe(true);
    expect(isEditorToolbarRegion('center')).toBe(true);
    expect(isEditorToolbarRegion('right')).toBe(true);
    expect(isEditorToolbarRegion('scene')).toBe(false);
  });
});

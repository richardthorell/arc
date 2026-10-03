import { describe, expect, it } from 'vitest';

import { editorToolbarPlacementFor, editorToolbarPrimaryActions } from './editorToolbarActionContract';

describe('editor toolbar primary action contract', () => {
  it('keeps document-local Save and Compile actions together on the left', () => {
    expect(editorToolbarPlacementFor('save')).toEqual({ region: 'left', supportsMenu: true });
    expect(editorToolbarPlacementFor('compile')).toEqual({ region: 'left', supportsMenu: true });
  });

  it('keeps scene Build with target controls on the right', () => {
    expect(editorToolbarPlacementFor('build')).toEqual({ region: 'right', supportsMenu: true });
  });

  it('defines every primary action through the shared contract', () => {
    expect(Object.keys(editorToolbarPrimaryActions).sort()).toEqual(['build', 'compile', 'save']);
  });
});

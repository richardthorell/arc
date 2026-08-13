import { describe, expect, it } from 'vitest';

import { commandRegistry } from './commandRegistry';
import type { CommandContext } from './workbenchTypes';

const context = (overrides: Partial<CommandContext> = {}): CommandContext => ({
  editorFocused: true,
  viewportFocused: true,
  textInputFocused: false,
  modalOpen: false,
  playing: false,
  hasSelection: true,
  canUndo: false,
  canRedo: false,
  projectOpen: true,
  ...overrides,
});

describe('commandRegistry', () => {
  it('allows the terrain toolbar command after its click moves focus out of the viewport', () => {
    const enabled = commandRegistry['viewport.terrain'].enabled;

    expect(enabled?.(context({ viewportFocused: false }))).toBe(true);
    expect(enabled?.(context({ viewportFocused: false, hasSelection: false }))).toBe(false);
  });
});

// @vitest-environment jsdom

import { describe, expect, it } from 'vitest';

import { KeybindingService, normalizeBinding } from './keybindingService';
import type { CommandContext } from './workbenchTypes';

const context = (overrides: Partial<CommandContext> = {}): CommandContext => ({
  editorFocused: true,
  viewportFocused: true,
  textInputFocused: false,
  modalOpen: false,
  playing: false,
  hasSelection: true,
  canUndo: true,
  canRedo: true,
  projectOpen: true,
  ...overrides,
});

const keyboardEvent = (key: string, options: KeyboardEventInit = {}) =>
  new KeyboardEvent('keydown', { key, ...options });

describe('KeybindingService', () => {
  it('normalizes modifiers and letter casing', () => {
    expect(normalizeBinding('shift+ctrl+s')).toBe('Ctrl+Shift+S');
    expect(normalizeBinding('ctrl+k ctrl+p')).toBe('Ctrl+K Ctrl+P');
  });

  it('matches default bindings and respects command contexts', () => {
    const service = new KeybindingService();
    expect(service.match(keyboardEvent('w'), context())?.command).toBe('viewport.translate');
    expect(service.match(keyboardEvent('w'), context({ viewportFocused: false }))).toBeNull();
    expect(service.match(keyboardEvent('d', { ctrlKey: true }), context())?.command).toBe('entity.duplicate');
    expect(service.match(keyboardEvent('d', { ctrlKey: true }), context({ hasSelection: false }))).toBeNull();
  });

  it('supports overrides and multi-stroke chords', () => {
    const service = new KeybindingService({ 'settings.open': ['Ctrl+J Ctrl+S'] });
    const first = service.match(keyboardEvent('j', { ctrlKey: true }), context(), 100);
    expect(first?.chordPending).toBe(true);
    expect(service.match(keyboardEvent('s', { ctrlKey: true }), context(), 200)).toEqual({
      command: 'settings.open',
      chordPending: false,
    });
  });

  it('reports conflicting overrides', () => {
    const service = new KeybindingService({
      'file.new': ['Ctrl+G'],
      'file.open': ['Ctrl+G'],
    });
    expect(service.conflicts().get('Ctrl+G')).toEqual(['file.new', 'file.open']);
  });
});

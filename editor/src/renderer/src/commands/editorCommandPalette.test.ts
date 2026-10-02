import { describe, expect, it } from 'vitest';
import { EditorCommandRegistry } from './editorCommands';
import {
  deriveEditorCommandPaletteView,
  moveEditorCommandPaletteSelection,
  updateEditorCommandPaletteQuery,
} from './editorCommandPalette';

const registry = (): EditorCommandRegistry => {
  const commands = new EditorCommandRegistry();
  commands.register({ id: 'scene.save', title: 'Save Scene', category: 'Scene' });
  commands.register({ id: 'editor.settings', title: 'Open Settings', category: 'Editor', keywords: ['preferences'] });
  commands.register({ id: 'scene.focus', title: 'Focus Selection', category: 'Scene' });
  return commands;
};

describe('editor command palette state', () => {
  it('selects the first deterministic search result by default', () => {
    const view = deriveEditorCommandPaletteView(registry(), { query: 'scene' });
    expect(view.selectedIndex).toBe(0);
    expect(view.selectedCommand?.id).toBe(view.matches[0].command.id);
  });

  it('tracks selection by stable command id while navigating', () => {
    const commands = registry();
    const initial = deriveEditorCommandPaletteView(commands, { query: 'scene' });
    const moved = moveEditorCommandPaletteSelection(
      commands,
      { query: 'scene', selectedCommandId: initial.selectedCommand?.id },
      1,
    );
    expect(moved.selectedCommandId).toBe(initial.matches[1].command.id);
  });

  it('clamps keyboard navigation to available results', () => {
    const commands = registry();
    const state = moveEditorCommandPaletteSelection(commands, { query: '' }, 100);
    const view = deriveEditorCommandPaletteView(commands, state);
    expect(view.selectedIndex).toBe(view.matches.length - 1);
  });

  it('reselects a valid command when filtering removes the previous selection', () => {
    const commands = registry();
    const state = updateEditorCommandPaletteQuery(
      commands,
      { query: '', selectedCommandId: 'scene.save' },
      'preferences',
    );
    expect(state.selectedCommandId).toBe('editor.settings');
  });

  it('clears selection when there are no matches', () => {
    const commands = registry();
    const state = updateEditorCommandPaletteQuery(
      commands,
      { query: '', selectedCommandId: 'scene.save' },
      'does-not-exist',
    );
    expect(state.selectedCommandId).toBeUndefined();
    expect(deriveEditorCommandPaletteView(commands, state).selectedIndex).toBe(-1);
  });
});

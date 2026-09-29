import { describe, expect, it } from 'vitest';

import {
  EditorCommandRegistry,
  editorShortcutConflicts,
  resolveEditorKeybindings,
  validateEditorKeybindingOverrides,
} from './editorCommands';

describe('EditorCommandRegistry', () => {
  it('keeps stable command IDs and rejects duplicates', () => {
    const registry = new EditorCommandRegistry();
    registry.register({ id: 'scene.focus-selection', title: 'Focus Selection' });

    expect(registry.get('scene.focus-selection')?.title).toBe('Focus Selection');
    expect(() => registry.register({ id: 'scene.focus-selection', title: 'Focus' })).toThrow(/already registered/);
  });

  it('searches title, category, ID, and keywords deterministically', () => {
    const registry = new EditorCommandRegistry();
    registry.register({
      id: 'edit.undo',
      title: 'Undo',
      category: 'Edit',
      keywords: ['history'],
      defaultShortcut: 'Ctrl+Z',
    });
    registry.register({
      id: 'scene.focus-selection',
      title: 'Focus Selection',
      category: 'Viewport',
      keywords: ['frame'],
    });
    registry.register({
      id: 'edit.redo',
      title: 'Redo',
      category: 'Edit',
      keywords: ['history'],
      defaultShortcut: 'Ctrl+Shift+Z',
    });

    expect(registry.search('focus').map((match) => match.command.id)).toEqual(['scene.focus-selection']);
    expect(registry.search('history').map((match) => match.command.id)).toEqual(['edit.redo', 'edit.undo']);
    expect(registry.search('viewport frame').map((match) => match.command.id)).toEqual(['scene.focus-selection']);
    expect(registry.search('edit.undo').map((match) => match.command.id)).toEqual(['edit.undo']);
  });

  it('lists commands by stable ID rather than registration order', () => {
    const registry = new EditorCommandRegistry();
    registry.register({ id: 'view.reset', title: 'Reset View' });
    registry.register({ id: 'edit.undo', title: 'Undo' });

    expect(registry.list().map((command) => command.id)).toEqual(['edit.undo', 'view.reset']);
  });
});

const commands = [
  { id: 'edit.undo', title: 'Undo', defaultShortcut: 'Ctrl+Z' },
  { id: 'edit.redo', title: 'Redo', defaultShortcut: 'Ctrl+Shift+Z' },
  { id: 'view.frame', title: 'Frame Selection', defaultShortcut: 'F' },
];

describe('resolveEditorKeybindings', () => {
  it('resolves defaults, overrides, and explicit disabled bindings deterministically', () => {
    expect(resolveEditorKeybindings(commands, { 'edit.undo': ' Alt+Z ', 'view.frame': null })).toEqual([
      { commandId: 'edit.redo', shortcut: 'Ctrl+Shift+Z', source: 'default' },
      { commandId: 'edit.undo', shortcut: 'Alt+Z', source: 'override' },
      { commandId: 'view.frame', source: 'disabled' },
    ]);
  });

  it('reports stale and empty persisted overrides', () => {
    expect(validateEditorKeybindingOverrides(commands, { stale: 'Q', 'edit.undo': '   ', 'view.frame': null })).toEqual([
      { commandId: 'edit.undo', kind: 'empty-shortcut' },
      { commandId: 'stale', kind: 'unknown-command' },
    ]);
  });
});

describe('editorShortcutConflicts', () => {
  it('detects normalized conflicts from user overrides', () => {
    expect(editorShortcutConflicts(commands, { 'edit.redo': ' ctrl + z ' })).toEqual([
      { shortcut: 'ctrl+z', commandIds: ['edit.redo', 'edit.undo'] },
    ]);
  });

  it('does not report disabled bindings as conflicts', () => {
    expect(editorShortcutConflicts(commands, { 'edit.undo': null, 'edit.redo': 'Ctrl+Z' })).toEqual([]);
  });

  it('does not report distinct shortcuts', () => {
    expect(editorShortcutConflicts(commands)).toEqual([]);
  });
});

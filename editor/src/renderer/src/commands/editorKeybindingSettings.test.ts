import { describe, expect, it } from 'vitest';

import { editorKeybindingSettingsRows } from './editorKeybindingSettings';
import type { EditorCommand } from './editorCommands';

const commands: EditorCommand[] = [
  { id: 'edit.redo', title: 'Redo', category: 'Edit', defaultShortcut: 'Ctrl+Shift+Z' },
  { id: 'edit.undo', title: 'Undo', category: 'Edit', defaultShortcut: 'Ctrl+Z' },
  { id: 'file.save', title: 'Save', category: 'File', defaultShortcut: 'Ctrl+S' },
  { id: 'view.focus', title: 'Focus Selection', category: 'View' },
];

describe('editorKeybindingSettingsRows', () => {
  it('projects defaults in stable category/title order', () => {
    expect(editorKeybindingSettingsRows(commands)).toEqual([
      {
        commandId: 'edit.redo',
        title: 'Redo',
        category: 'Edit',
        shortcut: 'Ctrl+Shift+Z',
        source: 'default',
        conflictCommandIds: [],
      },
      {
        commandId: 'edit.undo',
        title: 'Undo',
        category: 'Edit',
        shortcut: 'Ctrl+Z',
        source: 'default',
        conflictCommandIds: [],
      },
      {
        commandId: 'file.save',
        title: 'Save',
        category: 'File',
        shortcut: 'Ctrl+S',
        source: 'default',
        conflictCommandIds: [],
      },
      {
        commandId: 'view.focus',
        title: 'Focus Selection',
        category: 'View',
        source: 'default',
        conflictCommandIds: [],
      },
    ]);
  });

  it('surfaces effective overrides, disabled bindings, and conflicts', () => {
    const rows = editorKeybindingSettingsRows(commands, {
      'edit.redo': ' ctrl + s ',
      'edit.undo': null,
    });

    expect(rows.find((row) => row.commandId === 'edit.redo')).toMatchObject({
      shortcut: 'ctrl + s',
      source: 'override',
      conflictCommandIds: ['file.save'],
    });
    expect(rows.find((row) => row.commandId === 'file.save')).toMatchObject({
      shortcut: 'Ctrl+S',
      source: 'default',
      conflictCommandIds: ['edit.redo'],
    });
    expect(rows.find((row) => row.commandId === 'edit.undo')).toMatchObject({
      source: 'disabled',
      conflictCommandIds: [],
    });
  });

  it('does not mutate commands or overrides', () => {
    const inputCommands = commands.map((command) => ({ ...command }));
    const overrides = { 'view.focus': 'F' } as const;

    editorKeybindingSettingsRows(inputCommands, overrides);

    expect(inputCommands).toEqual(commands);
    expect(overrides).toEqual({ 'view.focus': 'F' });
  });
});

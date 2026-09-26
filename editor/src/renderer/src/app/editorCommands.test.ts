import { describe, expect, it } from 'vitest';

import { EditorCommandRegistry, normalizeShortcut } from './editorCommands';

describe('editor command registry', () => {
  it('keeps stable command ids unique', () => {
    const registry = new EditorCommandRegistry();
    registry.register({ id: 'editor.save', title: 'Save' });

    expect(() => registry.register({ id: 'editor.save', title: 'Save Again' })).toThrow(
      "Editor command 'editor.save' is already registered",
    );
    expect(registry.get('editor.save')?.title).toBe('Save');
  });

  it('searches title, category, keywords, and stable ids', () => {
    const registry = new EditorCommandRegistry();
    registry.register({ id: 'editor.save', title: 'Save', category: 'File', keywords: ['write'] });
    registry.register({
      id: 'viewport.frame-selection',
      title: 'Frame Selection',
      category: 'Viewport',
      keywords: ['focus'],
    });

    expect(registry.search('frame')[0].command.id).toBe('viewport.frame-selection');
    expect(registry.search('viewport')[0].command.id).toBe('viewport.frame-selection');
    expect(registry.search('focus')[0].command.id).toBe('viewport.frame-selection');
    expect(registry.search('editor.save')[0].command.id).toBe('editor.save');
  });

  it('detects conflicts after user shortcut overrides are applied', () => {
    const registry = new EditorCommandRegistry();
    registry.register({ id: 'editor.save', title: 'Save', defaultShortcut: 'Ctrl+S' });
    registry.register({ id: 'editor.search', title: 'Search', defaultShortcut: 'Ctrl+F' });

    const conflicts = registry.shortcutConflicts({ 'editor.search': ' ctrl + s ' });
    expect(conflicts.get('ctrl+s')).toEqual(['editor.save', 'editor.search']);
  });

  it('normalizes shortcut spelling for deterministic comparison', () => {
    expect(normalizeShortcut(' Ctrl + Shift + P ')).toBe('ctrl+shift+p');
  });
});

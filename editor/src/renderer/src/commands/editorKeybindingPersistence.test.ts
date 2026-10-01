import { describe, expect, it } from 'vitest';
import {
  deserializeEditorKeybindingOverrides,
  EDITOR_KEYBINDING_OVERRIDES_VERSION,
  serializeEditorKeybindingOverrides,
} from './editorKeybindingPersistence';

describe('editor keybinding persistence', () => {
  it('serializes overrides deterministically while preserving disabled bindings', () => {
    expect(
      serializeEditorKeybindingOverrides({
        'scene.save': ' Ctrl + Shift + S ',
        'edit.undo': null,
        'unused.command': undefined,
      }),
    ).toEqual({
      version: EDITOR_KEYBINDING_OVERRIDES_VERSION,
      overrides: {
        'edit.undo': null,
        'scene.save': 'Ctrl + Shift + S',
      },
    });
  });

  it('round-trips unknown command ids for the validation layer', () => {
    const persisted = serializeEditorKeybindingOverrides({
      'legacy.command': 'Ctrl+L',
      'scene.save': null,
    });

    expect(deserializeEditorKeybindingOverrides(persisted)).toEqual({
      overrides: {
        'legacy.command': 'Ctrl+L',
        'scene.save': null,
      },
    });
  });

  it('rejects malformed payloads atomically', () => {
    expect(
      deserializeEditorKeybindingOverrides({
        version: EDITOR_KEYBINDING_OVERRIDES_VERSION,
        overrides: { 'scene.save': 42 },
      }),
    ).toEqual({ overrides: {}, diagnostic: 'invalid-payload' });
  });

  it('rejects unsupported versions without guessing a migration', () => {
    expect(deserializeEditorKeybindingOverrides({ version: 2, overrides: { 'scene.save': 'Ctrl+S' } })).toEqual({
      overrides: {},
      diagnostic: 'unsupported-version',
    });
  });
});

import type { EditorKeybindingOverrides } from './editorCommands';

export const EDITOR_KEYBINDING_OVERRIDES_VERSION = 1 as const;

export type PersistedEditorKeybindingOverrides = {
  version: typeof EDITOR_KEYBINDING_OVERRIDES_VERSION;
  overrides: Record<string, string | null>;
};

export type EditorKeybindingLoadResult = {
  overrides: EditorKeybindingOverrides;
  diagnostic?: 'invalid-payload' | 'unsupported-version';
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/**
 * Serialize user keybinding overrides independently from command defaults.
 * Stable command IDs are the persistence key; undefined entries are omitted and
 * null remains an explicit disabled binding.
 */
export const serializeEditorKeybindingOverrides = (
  overrides: EditorKeybindingOverrides,
): PersistedEditorKeybindingOverrides => ({
  version: EDITOR_KEYBINDING_OVERRIDES_VERSION,
  overrides: Object.fromEntries(
    Object.entries(overrides)
      .filter(([, shortcut]) => shortcut !== undefined)
      .sort(([leftId], [rightId]) => leftId.localeCompare(rightId))
      .map(([commandId, shortcut]) => [commandId, shortcut === null ? null : shortcut!.trim()]),
  ),
});

/**
 * Decode persisted overrides defensively. Unknown command IDs are intentionally
 * preserved so the existing validation layer can surface stale settings instead
 * of silently discarding user configuration.
 */
export const deserializeEditorKeybindingOverrides = (payload: unknown): EditorKeybindingLoadResult => {
  if (!isRecord(payload) || typeof payload.version !== 'number' || !isRecord(payload.overrides)) {
    return { overrides: {}, diagnostic: 'invalid-payload' };
  }
  if (payload.version !== EDITOR_KEYBINDING_OVERRIDES_VERSION) {
    return { overrides: {}, diagnostic: 'unsupported-version' };
  }

  const overrides: Record<string, string | null> = {};
  for (const [commandId, shortcut] of Object.entries(payload.overrides)) {
    if (!commandId.trim() || (shortcut !== null && typeof shortcut !== 'string')) {
      return { overrides: {}, diagnostic: 'invalid-payload' };
    }
    overrides[commandId] = shortcut;
  }
  return { overrides };
};

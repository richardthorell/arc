import {
  editorShortcutConflicts,
  resolveEditorKeybindings,
  type EditorCommand,
  type EditorKeybindingOverrides,
} from './editorCommands';

export type EditorKeybindingSettingsRow = {
  commandId: string;
  title: string;
  category?: string;
  shortcut?: string;
  source: 'default' | 'override' | 'disabled';
  conflictCommandIds: string[];
};

const normalizeShortcut = (shortcut: string): string => shortcut.trim().toLocaleLowerCase().replace(/\s+/g, '');

/**
 * Build a deterministic, UI-ready view of editor keybindings.
 *
 * Settings surfaces should consume this projection instead of independently
 * resolving defaults/overrides or reimplementing shortcut conflict detection.
 */
export const editorKeybindingSettingsRows = (
  commands: readonly EditorCommand[],
  overrides: EditorKeybindingOverrides = {},
): EditorKeybindingSettingsRow[] => {
  const commandById = new Map(commands.map((command) => [command.id, command]));
  const conflictsByShortcut = new Map(
    editorShortcutConflicts(commands, overrides).map((conflict) => [conflict.shortcut, conflict.commandIds]),
  );

  return resolveEditorKeybindings(commands, overrides)
    .flatMap((binding): EditorKeybindingSettingsRow[] => {
      const command = commandById.get(binding.commandId);
      if (!command) return [];

      const conflictCommandIds = binding.shortcut
        ? (conflictsByShortcut.get(normalizeShortcut(binding.shortcut)) ?? []).filter(
            (commandId) => commandId !== binding.commandId,
          )
        : [];

      return [
        {
          commandId: binding.commandId,
          title: command.title,
          ...(command.category ? { category: command.category } : {}),
          ...(binding.shortcut ? { shortcut: binding.shortcut } : {}),
          source: binding.source,
          conflictCommandIds,
        },
      ];
    })
    .sort(
      (left, right) =>
        (left.category ?? '').localeCompare(right.category ?? '') ||
        left.title.localeCompare(right.title) ||
        left.commandId.localeCompare(right.commandId),
    );
};

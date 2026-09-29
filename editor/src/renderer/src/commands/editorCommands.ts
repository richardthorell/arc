export type EditorCommand = {
  id: string;
  title: string;
  category?: string;
  keywords?: readonly string[];
  defaultShortcut?: string;
};

export type EditorCommandMatch = {
  command: EditorCommand;
  score: number;
};

const normalizedTerms = (value: string): string[] => value.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);

const searchableText = (command: EditorCommand): string[] => [
  command.title,
  command.category ?? '',
  command.id,
  ...(command.keywords ?? []),
];

const scoreCommand = (command: EditorCommand, terms: readonly string[]): number | undefined => {
  if (terms.length === 0) return 0;

  const fields = searchableText(command).map((field) => field.toLocaleLowerCase());
  let score = 0;
  for (const term of terms) {
    const title = command.title.toLocaleLowerCase();
    if (title === term) score += 100;
    else if (title.startsWith(term)) score += 50;
    else if (title.includes(term)) score += 25;
    else if (fields.some((field) => field.includes(term))) score += 10;
    else return undefined;
  }
  return score;
};

/**
 * Domain-neutral registry for editor commands.
 *
 * Command IDs are stable API: UI labels and shortcuts may change without breaking
 * automation, menus, palette entries, or persisted keybinding overrides.
 */
export class EditorCommandRegistry {
  private readonly commands = new Map<string, EditorCommand>();

  register(command: EditorCommand): void {
    const id = command.id.trim();
    if (!id) throw new Error('Editor command ID must not be empty');
    if (this.commands.has(id)) throw new Error(`Editor command already registered: ${id}`);
    this.commands.set(id, { ...command, id });
  }

  get(id: string): EditorCommand | undefined {
    return this.commands.get(id);
  }

  list(): EditorCommand[] {
    return [...this.commands.values()].sort((left, right) => left.id.localeCompare(right.id));
  }

  search(query: string): EditorCommandMatch[] {
    const terms = normalizedTerms(query);
    return this.list()
      .flatMap((command) => {
        const score = scoreCommand(command, terms);
        return score === undefined ? [] : [{ command, score }];
      })
      .sort((left, right) => right.score - left.score || left.command.title.localeCompare(right.command.title));
  }
}

export type EditorKeybindingOverrides = Readonly<Record<string, string | null | undefined>>;

export type ResolvedEditorKeybinding = {
  commandId: string;
  shortcut?: string;
  source: 'default' | 'override' | 'disabled';
};

export type ShortcutConflict = {
  shortcut: string;
  commandIds: string[];
};

export type KeybindingOverrideDiagnostic = {
  commandId: string;
  kind: 'unknown-command' | 'empty-shortcut';
};

const normalizeShortcut = (shortcut: string): string => shortcut.trim().toLocaleLowerCase().replace(/\s+/g, '');

/** Resolve defaults and user overrides without mutating the command registry. Null explicitly disables a default binding. */
export const resolveEditorKeybindings = (
  commands: readonly EditorCommand[],
  overrides: EditorKeybindingOverrides = {},
): ResolvedEditorKeybinding[] =>
  commands
    .map((command): ResolvedEditorKeybinding => {
      if (Object.prototype.hasOwnProperty.call(overrides, command.id)) {
        const override = overrides[command.id];
        if (override === null) return { commandId: command.id, source: 'disabled' };
        if (override !== undefined) {
          const shortcut = override.trim();
          return shortcut
            ? { commandId: command.id, shortcut, source: 'override' }
            : { commandId: command.id, source: 'disabled' };
        }
      }
      return command.defaultShortcut
        ? { commandId: command.id, shortcut: command.defaultShortcut.trim(), source: 'default' }
        : { commandId: command.id, source: 'default' };
    })
    .sort((left, right) => left.commandId.localeCompare(right.commandId));

/** Validate persisted overrides so stale command IDs and accidental empty bindings are visible to Settings UI. */
export const validateEditorKeybindingOverrides = (
  commands: readonly EditorCommand[],
  overrides: EditorKeybindingOverrides,
): KeybindingOverrideDiagnostic[] => {
  const known = new Set(commands.map((command) => command.id));
  return Object.entries(overrides)
    .flatMap(([commandId, shortcut]): KeybindingOverrideDiagnostic[] => {
      if (!known.has(commandId)) return [{ commandId, kind: 'unknown-command' }];
      if (typeof shortcut === 'string' && !shortcut.trim()) return [{ commandId, kind: 'empty-shortcut' }];
      return [];
    })
    .sort((left, right) => left.commandId.localeCompare(right.commandId) || left.kind.localeCompare(right.kind));
};

/** Returns deterministic conflicts from the effective keybinding set without mutating command or override state. */
export const editorShortcutConflicts = (
  commands: readonly EditorCommand[],
  overrides: EditorKeybindingOverrides = {},
): ShortcutConflict[] => {
  const bindings = new Map<string, string[]>();
  for (const binding of resolveEditorKeybindings(commands, overrides)) {
    if (!binding.shortcut || binding.source === 'disabled') continue;
    const normalized = normalizeShortcut(binding.shortcut);
    if (!normalized) continue;
    const commandIds = bindings.get(normalized) ?? [];
    commandIds.push(binding.commandId);
    bindings.set(normalized, commandIds);
  }

  return [...bindings.entries()]
    .filter(([, commandIds]) => commandIds.length > 1)
    .map(([shortcut, commandIds]) => ({ shortcut, commandIds: commandIds.sort() }))
    .sort((left, right) => left.shortcut.localeCompare(right.shortcut));
};

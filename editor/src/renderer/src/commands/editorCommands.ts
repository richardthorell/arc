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

const normalizedTerms = (value: string): string[] =>
  value
    .trim()
    .toLocaleLowerCase()
    .split(/\s+/)
    .filter(Boolean);

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

export type ShortcutConflict = {
  shortcut: string;
  commandIds: string[];
};

const normalizeShortcut = (shortcut: string): string => shortcut.trim().toLocaleLowerCase().replace(/\s+/g, '');

/** Returns deterministic conflicts without mutating command or keybinding state. */
export const editorShortcutConflicts = (
  commands: readonly EditorCommand[],
  overrides: Readonly<Record<string, string | undefined>> = {},
): ShortcutConflict[] => {
  const bindings = new Map<string, string[]>();
  for (const command of commands) {
    const shortcut = overrides[command.id] ?? command.defaultShortcut;
    if (!shortcut) continue;
    const normalized = normalizeShortcut(shortcut);
    if (!normalized) continue;
    const commandIds = bindings.get(normalized) ?? [];
    commandIds.push(command.id);
    bindings.set(normalized, commandIds);
  }

  return [...bindings.entries()]
    .filter(([, commandIds]) => commandIds.length > 1)
    .map(([shortcut, commandIds]) => ({ shortcut, commandIds: commandIds.sort() }))
    .sort((left, right) => left.shortcut.localeCompare(right.shortcut));
};

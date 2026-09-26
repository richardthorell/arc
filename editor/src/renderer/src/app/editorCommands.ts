export type EditorCommandId = string;

export interface EditorCommand {
  id: EditorCommandId;
  title: string;
  category?: string;
  keywords?: readonly string[];
  defaultShortcut?: string;
}

export interface EditorCommandMatch {
  command: EditorCommand;
  score: number;
}

const normalize = (value: string): string => value.trim().toLocaleLowerCase();

export class EditorCommandRegistry {
  private readonly commands = new Map<EditorCommandId, EditorCommand>();

  register(command: EditorCommand): void {
    if (!command.id.trim()) throw new Error('Editor command id must not be empty');
    if (!command.title.trim()) throw new Error(`Editor command '${command.id}' must have a title`);
    if (this.commands.has(command.id)) throw new Error(`Editor command '${command.id}' is already registered`);
    this.commands.set(command.id, { ...command });
  }

  unregister(id: EditorCommandId): boolean {
    return this.commands.delete(id);
  }

  get(id: EditorCommandId): EditorCommand | undefined {
    return this.commands.get(id);
  }

  list(): EditorCommand[] {
    return [...this.commands.values()];
  }

  search(query: string): EditorCommandMatch[] {
    const needle = normalize(query);
    if (!needle) return this.list().map((command) => ({ command, score: 0 }));

    return this.list()
      .map((command) => ({ command, score: scoreCommand(command, needle) }))
      .filter((match) => match.score >= 0)
      .sort((a, b) => b.score - a.score || a.command.title.localeCompare(b.command.title));
  }

  shortcutConflicts(shortcuts: Readonly<Record<EditorCommandId, string | undefined>> = {}): Map<string, EditorCommandId[]> {
    const byShortcut = new Map<string, EditorCommandId[]>();
    for (const command of this.commands.values()) {
      const shortcut = shortcuts[command.id] ?? command.defaultShortcut;
      if (!shortcut?.trim()) continue;
      const key = normalizeShortcut(shortcut);
      byShortcut.set(key, [...(byShortcut.get(key) ?? []), command.id]);
    }
    return new Map([...byShortcut].filter(([, ids]) => ids.length > 1));
  }
}

function scoreCommand(command: EditorCommand, needle: string): number {
  const title = normalize(command.title);
  const id = normalize(command.id);
  const category = normalize(command.category ?? '');
  const keywords = (command.keywords ?? []).map(normalize);

  if (title === needle) return 100;
  if (title.startsWith(needle)) return 80;
  if (title.includes(needle)) return 60;
  if (id.includes(needle)) return 50;
  if (category.includes(needle)) return 40;
  if (keywords.some((keyword) => keyword.includes(needle))) return 30;
  return -1;
}

export function normalizeShortcut(shortcut: string): string {
  return shortcut
    .split('+')
    .map((part) => normalize(part))
    .filter(Boolean)
    .join('+');
}

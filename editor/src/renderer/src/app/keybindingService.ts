import { commandRegistry } from './commandRegistry';
import type { CommandContext, CommandId } from './workbenchTypes';

export type KeybindingOverrides = Partial<Record<CommandId, readonly string[]>>;

export type KeybindingMatch = {
  command: CommandId;
  chordPending: boolean;
};

const modifierOrder = ['Ctrl', 'Alt', 'Shift', 'Meta'] as const;

const normalizeKeyName = (key: string) => {
  if (key === ' ') return 'Space';
  if (key === 'Esc') return 'Escape';
  if (key.length === 1) return key.toUpperCase();
  return key[0].toUpperCase() + key.slice(1);
};

export const normalizeBinding = (binding: string) =>
  binding
    .trim()
    .split(/\s+/)
    .map((stroke) => {
      const parts = stroke
        .split('+')
        .map((part) => part.trim())
        .filter(Boolean);
      const modifiers = modifierOrder.filter((modifier) =>
        parts.some((part) => part.toLocaleLowerCase() === modifier.toLocaleLowerCase()),
      );
      const key = parts.find(
        (part) => !modifierOrder.some((modifier) => modifier.toLocaleLowerCase() === part.toLocaleLowerCase()),
      );
      return [...modifiers, normalizeKeyName(key ?? '')].filter(Boolean).join('+');
    })
    .join(' ');

export const eventStroke = (event: KeyboardEvent) => {
  const parts: string[] = [];
  if (event.ctrlKey) parts.push('Ctrl');
  if (event.altKey) parts.push('Alt');
  if (event.shiftKey) parts.push('Shift');
  if (event.metaKey) parts.push('Meta');
  if (!['Control', 'Alt', 'Shift', 'Meta'].includes(event.key)) parts.push(normalizeKeyName(event.key));
  return parts.join('+');
};

export class KeybindingService {
  private overrides: KeybindingOverrides;
  private pendingStroke = '';
  private pendingDeadline = 0;

  constructor(overrides: KeybindingOverrides = {}) {
    this.overrides = overrides;
  }

  setOverrides(overrides: KeybindingOverrides) {
    this.overrides = overrides;
    this.cancelChord();
  }

  bindings(command: CommandId): readonly string[] {
    return this.overrides[command] ?? commandRegistry[command].defaultKeybindings ?? [];
  }

  primaryBinding(command: CommandId): string {
    return this.bindings(command)[0] ?? '';
  }

  conflicts(): Map<string, CommandId[]> {
    const result = new Map<string, CommandId[]>();
    for (const command of Object.keys(commandRegistry) as CommandId[]) {
      for (const binding of this.bindings(command)) {
        const normalized = normalizeBinding(binding);
        result.set(normalized, [...(result.get(normalized) ?? []), command]);
      }
    }
    for (const [binding, commands] of result) if (commands.length < 2) result.delete(binding);
    return result;
  }

  match(event: KeyboardEvent, context: CommandContext, now = Date.now()): KeybindingMatch | null {
    if (now > this.pendingDeadline) this.cancelChord();
    const stroke = eventStroke(event);
    const candidate = this.pendingStroke ? `${this.pendingStroke} ${stroke}` : stroke;
    let prefix = false;
    for (const command of Object.keys(commandRegistry) as CommandId[]) {
      const registration = commandRegistry[command];
      if (registration.enabled && !registration.enabled(context)) continue;
      for (const binding of this.bindings(command)) {
        const normalized = normalizeBinding(binding);
        if (normalized === candidate) {
          this.cancelChord();
          return { command, chordPending: false };
        }
        if (normalized.startsWith(`${candidate} `)) prefix = true;
      }
    }
    if (prefix) {
      this.pendingStroke = candidate;
      this.pendingDeadline = now + 1500;
      return { command: 'view.commandPalette', chordPending: true };
    }
    this.cancelChord();
    return null;
  }

  cancelChord() {
    this.pendingStroke = '';
    this.pendingDeadline = 0;
  }
}

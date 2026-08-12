import { useEffect, useMemo, useRef, useState } from 'react';
import { Command, Search } from 'lucide-react';

import { allCommands } from './commandRegistry';
import type { CommandContext, CommandId } from './workbenchTypes';

type CommandPaletteProps = {
  context: CommandContext;
  shortcut: (command: CommandId) => string;
  onClose: () => void;
  onCommand: (command: CommandId) => void;
};

const scoreCommand = (query: string, label: string, category: string, description: string) => {
  if (!query) return 1;
  const words = query.toLocaleLowerCase().split(/\s+/).filter(Boolean);
  const haystack = `${label} ${category} ${description}`.toLocaleLowerCase();
  if (!words.every((word) => haystack.includes(word))) return 0;
  if (label.toLocaleLowerCase().startsWith(query.toLocaleLowerCase())) return 4;
  if (label.toLocaleLowerCase().includes(query.toLocaleLowerCase())) return 3;
  return 2;
};

export function CommandPalette({ context, shortcut, onClose, onCommand }: CommandPaletteProps) {
  const [query, setQuery] = useState('');
  const [selected, setSelected] = useState(0);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const commands = useMemo(
    () =>
      allCommands
        .map((command) => ({
          command,
          score: scoreCommand(query, command.label, command.category, command.description),
        }))
        .filter((entry) => entry.score > 0)
        .sort((left, right) => right.score - left.score || left.command.label.localeCompare(right.command.label)),
    [query],
  );

  useEffect(() => inputRef.current?.focus(), []);
  useEffect(() => setSelected((value) => Math.min(value, Math.max(0, commands.length - 1))), [commands.length]);

  const execute = (command: CommandId) => {
    const registration = allCommands.find((candidate) => candidate.id === command);
    if (registration?.enabled && !registration.enabled(context)) return;
    onClose();
    onCommand(command);
  };

  return (
    <div className="command-palette-backdrop" onPointerDown={onClose} role="presentation">
      <section
        aria-label="Command Palette"
        aria-modal="true"
        className="command-palette"
        onPointerDown={(event) => event.stopPropagation()}
        role="dialog"
      >
        <div className="command-palette-search">
          <Search size={17} />
          <input
            ref={inputRef}
            aria-label="Search commands"
            onChange={(event) => {
              setQuery(event.target.value);
              setSelected(0);
            }}
            onKeyDown={(event) => {
              if (event.key === 'Escape') onClose();
              else if (event.key === 'ArrowDown') {
                event.preventDefault();
                setSelected((value) => Math.min(value + 1, commands.length - 1));
              } else if (event.key === 'ArrowUp') {
                event.preventDefault();
                setSelected((value) => Math.max(value - 1, 0));
              } else if (event.key === 'Enter' && commands[selected]) execute(commands[selected].command.id);
            }}
            placeholder="Type a command or action..."
            value={query}
          />
        </div>
        <div className="command-palette-results" role="listbox">
          {commands.map(({ command }, index) => {
            const enabled = !command.enabled || command.enabled(context);
            return (
              <button
                aria-selected={selected === index}
                className={selected === index ? 'selected' : ''}
                disabled={!enabled}
                key={command.id}
                onClick={() => execute(command.id)}
                onPointerMove={() => setSelected(index)}
                role="option"
                title={!enabled ? command.disabledReason?.(context) : command.description}
                type="button"
              >
                <Command size={14} />
                <span>
                  <strong>{command.label}</strong>
                  <small>
                    {command.category} · {command.description}
                  </small>
                </span>
                {shortcut(command.id) && <kbd>{shortcut(command.id)}</kbd>}
              </button>
            );
          })}
          {!commands.length && <div className="command-palette-empty">No matching commands</div>}
        </div>
      </section>
    </div>
  );
}

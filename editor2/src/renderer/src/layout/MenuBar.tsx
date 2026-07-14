import { useEffect, useRef, useState } from 'react';

import type { CommandId } from '../app/workbenchTypes';
import { UiButton } from '../ui';
import { WindowControls } from './WindowControls';

type MenuBarProps = {
  projectTitle: string;
  onCommand: (command: CommandId) => void;
};

const menuItems = ['File', 'Edit', 'View', 'Scene', 'Render', 'Tools', 'Window', 'Help'] as const;
type MenuItem = typeof menuItems[number];

const menuCommands: Partial<Record<MenuItem, { label: string; command: CommandId }[]>> = {
  File: [
    { label: 'Open Scene...', command: 'file.open' },
    { label: 'Import Scene Into Current...', command: 'file.importScene' },
  ],
  Window: [
    { label: 'Reset Layout', command: 'layout.reset' },
  ],
};

export function MenuBar({ projectTitle, onCommand }: MenuBarProps) {
  const [openMenu, setOpenMenu] = useState<MenuItem | null>(null);
  const menuRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const close = (event: PointerEvent) => {
      if (!menuRef.current?.contains(event.target as Node)) {
        setOpenMenu(null);
      }
    };

    window.addEventListener('pointerdown', close);
    return () => window.removeEventListener('pointerdown', close);
  }, []);

  const runMenuCommand = (command: CommandId) => {
    setOpenMenu(null);
    onCommand(command);
  };

  return (
    <header className="workbench-titlebar">
      <div className="titlebar-left">
        <div className="arc-app-brand" aria-label="arc editor">
          <svg className="arc-logo-mark" viewBox="0 0 28 24" role="img" aria-hidden="true">
            <path d="M2 13.2 8.2 5.4l4.1 4.2-6 8.1Z" />
            <path d="M9.1 4.1 15.6 1l10.5 9.8-4.3 4.1Z" />
            <path d="M13 13.5h6.1l6.1 7.1h-7.2Z" />
          </svg>
          <strong>arc</strong>
        </div>

        <nav ref={menuRef} className="menu-bar" aria-label="Main menu">
          {menuItems.map((item) => {
            const commands = menuCommands[item];
            const expanded = openMenu === item;
            return (
              <div key={item} className="menu-bar-item">
                <UiButton
                  aria-expanded={expanded}
                  aria-haspopup={commands ? 'menu' : undefined}
                  onClick={() => commands ? setOpenMenu(expanded ? null : item) : setOpenMenu(null)}
                  variant="ghost"
                >
                  {item}
                </UiButton>
                {commands && expanded && (
                  <div className="menu-dropdown" role="menu">
                    {commands.map((entry) => (
                      <UiButton key={entry.command} role="menuitem" onClick={() => runMenuCommand(entry.command)} variant="ghost">
                        {entry.label}
                      </UiButton>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </nav>
      </div>

      <div className="window-title" aria-label={`Current scene ${projectTitle}`}>
        <span>{projectTitle}</span>
      </div>

      <WindowControls />
    </header>
  );
}

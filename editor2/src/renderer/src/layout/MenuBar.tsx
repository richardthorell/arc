import type { CommandId } from '../app/workbenchTypes';
import { WindowControls } from './WindowControls';

type MenuBarProps = {
  projectTitle: string;
  onCommand: (command: CommandId) => void;
};

const menuItems = ['File', 'Edit', 'View', 'Scene', 'Render', 'Tools', 'Window', 'Help'] as const;

export function MenuBar({ projectTitle, onCommand }: MenuBarProps) {
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

        <nav className="menu-bar" aria-label="Main menu">
          {menuItems.map((item) => (
            <button key={item} onClick={item === 'Window' ? () => onCommand('layout.reset') : undefined}>
              {item}
            </button>
          ))}
        </nav>
      </div>

      <div className="window-title" aria-label={`Current scene ${projectTitle}`}>
        <span>{projectTitle}</span>
      </div>

      <WindowControls />
    </header>
  );
}

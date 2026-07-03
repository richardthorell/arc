import { Search } from 'lucide-react';

import type { CommandId } from '../app/workbenchTypes';
import { WindowControls } from './WindowControls';

type MenuBarProps = {
  projectTitle: string;
  onCommand: (command: CommandId) => void;
};

export function MenuBar({ projectTitle, onCommand }: MenuBarProps) {
  return (
    <header className="workbench-titlebar">
      <div className="arc-app-brand">
        <div className="arc-logo-mark">a</div>
        <strong>arc</strong>
      </div>
      <nav className="menu-bar">
        <button>File</button>
        <button>Edit</button>
        <button>View</button>
        <button>Scene</button>
        <button>Render</button>
        <button>Tools</button>
        <button onClick={() => onCommand('layout.reset')}>Window</button>
        <button>Help</button>
      </nav>
      <label className="command-center">
        <Search size={14} />
        <input placeholder="Search commands, assets, entities" />
      </label>
      <div className="window-title">{projectTitle}</div>
      <WindowControls />
    </header>
  );
}

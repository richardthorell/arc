import { Search } from 'lucide-react';

import type { CommandId } from '../app/workbenchTypes';

type MenuBarProps = {
  projectTitle: string;
  onCommand: (command: CommandId) => void;
};

export function MenuBar({ projectTitle, onCommand }: MenuBarProps) {
  return (
    <header className="workbench-titlebar">
      <div className="traffic-spacer" />
      <nav className="menu-bar">
        <button>File</button>
        <button>Edit</button>
        <button>Selection</button>
        <button>View</button>
        <button>Scene</button>
        <button>Render</button>
        <button>Tools</button>
        <button onClick={() => onCommand('layout.reset')}>Window</button>
      </nav>
      <label className="command-center">
        <Search size={14} />
        <input placeholder="Search commands, assets, entities" />
      </label>
      <div className="window-title">{projectTitle}</div>
    </header>
  );
}

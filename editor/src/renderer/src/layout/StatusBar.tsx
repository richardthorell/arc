import { Circle, GitBranch } from 'lucide-react';

import type { StartupState } from '../app/workbenchTypes';

type StatusBarProps = {
  startupState: StartupState | null;
  activeScene?: string;
  lastCommand: string;
  aiControl?: string;
};

export function StatusBar({ startupState, activeScene, lastCommand, aiControl }: StatusBarProps) {
  return (
    <footer className="status-bar">
      <span>
        <GitBranch size={13} /> main
      </span>
      <span>
        <Circle size={10} /> {startupState?.engineHostConnected ? 'host connected' : 'mock host'}
      </span>
      <span>{activeScene ?? 'no scene'}</span>
      {aiControl && <span className="status-ai-control">{aiControl}</span>}
      <span className="status-spacer" />
      <span>{lastCommand}</span>
      <span>editor {startupState?.appVersion ?? '...'}</span>
    </footer>
  );
}

import { WindowControls } from '../layout/WindowControls';
import { UiLab } from './UiLab';

import './uiLabWindow.css';

export function UiLabWindow() {
  return (
    <div className="ui-lab-window">
      <header className="ui-lab-window-titlebar">
        <div className="ui-lab-window-title">
          <strong>arc</strong>
          <span>UI Lab</span>
        </div>
        <WindowControls />
      </header>
      <UiLab />
    </div>
  );
}

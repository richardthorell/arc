import { useState } from 'react';

import { WindowControls } from '../layout/WindowControls';
import { UiTab, UiTabs } from '../ui';
import { UiLab } from './UiLab';
import { UiLabDialogs } from './UiLabDialogs';
import { UiLabMaterialGallery } from './UiLabMaterialGallery';
import { UiLabPanels } from './UiLabPanels';
import { UiLabThemePicker } from './UiLabThemePicker';

import './uiLabWindow.css';

type UiLabPage = 'controls' | 'panels' | 'materials' | 'dialogs';

export function UiLabWindow() {
  const [page, setPage] = useState<UiLabPage>('controls');

  return (
    <div className="ui-lab-window">
      <header className="ui-lab-window-titlebar">
        <div className="ui-lab-window-title">
          <strong>arc</strong>
          <span>UI Lab</span>
        </div>
        <div className="ui-lab-window-actions">
          <UiLabThemePicker />
          <WindowControls />
        </div>
      </header>
      <UiTabs className="ui-lab-page-tabs" aria-label="UI Lab pages">
        <UiTab active={page === 'controls'} onClick={() => setPage('controls')}>
          Controls
        </UiTab>
        <UiTab active={page === 'panels'} onClick={() => setPage('panels')}>
          Panels
        </UiTab>
        <UiTab active={page === 'materials'} onClick={() => setPage('materials')}>
          Material Gallery
        </UiTab>
        <UiTab active={page === 'dialogs'} onClick={() => setPage('dialogs')}>
          Dialogs
        </UiTab>
      </UiTabs>
      <div className="ui-lab-page-scroll">
        {page === 'controls' && <UiLab />}
        {page === 'panels' && <UiLabPanels />}
        {page === 'materials' && <UiLabMaterialGallery />}
        {page === 'dialogs' && <UiLabDialogs />}
      </div>
    </div>
  );
}

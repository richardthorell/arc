import type { ReactNode } from 'react';

import { getPanel } from '../app/panelRegistry';
import type { DockRegion, WorkbenchPanelId } from '../app/workbenchTypes';

type DockHostProps = {
  region: DockRegion;
  panelIds: WorkbenchPanelId[];
  activePanelId: WorkbenchPanelId;
  onActivePanelChange: (panel: WorkbenchPanelId) => void;
  renderPanel: (panel: WorkbenchPanelId) => ReactNode;
};

export function DockHost({ region, panelIds, activePanelId, onActivePanelChange, renderPanel }: DockHostProps) {
  const activePanel = getPanel(activePanelId);

  return (
    <section className={`dock-host dock-${region}`}>
      <header className="dock-tabs">
        {panelIds.map((panelId) => {
          const panel = getPanel(panelId);
          const Icon = panel.icon;
          return (
            <button
              key={panel.id}
              className={panel.id === activePanelId ? 'dock-tab active' : 'dock-tab'}
              onClick={() => onActivePanelChange(panel.id)}
              title={panel.title}
            >
              <Icon size={14} />
              <span>{panel.title}</span>
            </button>
          );
        })}
      </header>
      <div className="dock-content" data-panel={activePanel.id}>
        {renderPanel(activePanel.id)}
      </div>
    </section>
  );
}

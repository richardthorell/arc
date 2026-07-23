import type { ReactNode } from 'react';
import { MoreVertical, X } from 'lucide-react';

import { getPanel } from '../app/panelRegistry';
import type { DockRegion, WorkbenchPanelId } from '../app/workbenchTypes';
import { UiIconButton, UiTab, UiTabs } from '../ui';

type DockHostProps = {
  region: DockRegion;
  panelIds: WorkbenchPanelId[];
  activePanelId: WorkbenchPanelId;
  onActivePanelChange: (panel: WorkbenchPanelId) => void;
  renderPanel: (panel: WorkbenchPanelId) => ReactNode;
};

export function DockHost({ region, panelIds, activePanelId, onActivePanelChange, renderPanel }: DockHostProps) {
  const resolvedActivePanelId = panelIds.includes(activePanelId) ? activePanelId : panelIds[0];
  const activePanel = getPanel(resolvedActivePanelId);

  return (
    <section className={`dock-host dock-${region}`}>
      <UiTabs className="dock-tabs">
        <div className="dock-tab-strip">
          {panelIds.map((panelId) => {
            const panel = getPanel(panelId);
            const Icon = panel.icon;
            return (
              <UiTab
                active={panel.id === resolvedActivePanelId}
                key={panel.id}
                className="dock-tab"
                onClick={() => onActivePanelChange(panel.id)}
                title={panel.title}
              >
                <Icon size={14} />
                <span>{panel.title}</span>
                <X className="dock-tab-close" size={12} />
              </UiTab>
            );
          })}
        </div>
        <UiIconButton className="dock-header-action" label={`${activePanel.title} panel actions`}>
          <MoreVertical size={14} />
        </UiIconButton>
      </UiTabs>
      <div className="dock-content" data-panel={activePanel.id}>
        {renderPanel(activePanel.id)}
      </div>
    </section>
  );
}

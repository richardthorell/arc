import { createRoot, type Root } from 'react-dom/client';
import type { HeaderPartInitParameters, IHeaderRenderer } from 'dockview';

import { panelRegistry } from '../app/panelRegistry';
import type { WorkbenchPanelId } from '../app/workbenchTypes';
import { UiTab } from '../ui';

export type PanelTabPresentation = {
  title: string;
  icon: (typeof panelRegistry)[WorkbenchPanelId]['icon'];
  closeable: boolean;
};

export function getPanelTabPresentation(component: string, instanceTitle?: string): PanelTabPresentation {
  const descriptor = panelRegistry[component as WorkbenchPanelId];
  return {
    title: instanceTitle || descriptor?.title || component,
    icon: descriptor?.icon ?? null,
    closeable: descriptor?.closeable !== false,
  };
}

export class PanelDockTabRenderer implements IHeaderRenderer {
  readonly element = document.createElement('div');
  private root: Root | null = null;
  private parameters: HeaderPartInitParameters | null = null;
  private subscriptions: Array<{ dispose(): void }> = [];

  constructor() {
    this.element.className = 'workspace-dock-tab-host';
  }

  init(parameters: HeaderPartInitParameters) {
    this.parameters = parameters;
    this.root = createRoot(this.element);
    this.subscriptions.push(parameters.api.onDidActiveChange(() => this.render()));
    this.subscriptions.push(parameters.api.onDidTitleChange(() => this.render()));
    this.render();
  }

  private render() {
    if (!this.parameters) return;
    const { api } = this.parameters;
    const presentation = getPanelTabPresentation(api.component, api.title);
    const Icon = presentation.icon;
    const closeProps = presentation.closeable
      ? {
          closeLabel: `Close ${presentation.title}`,
          onClose: () => api.close(),
        }
      : {};

    this.root?.render(
      <UiTab
        active={api.isActive}
        className="workspace-dock-tab"
        icon={Icon ? <Icon aria-hidden="true" size={14} /> : undefined}
        onClick={() => api.setActive()}
        title={presentation.title}
        {...closeProps}
      >
        {presentation.title}
      </UiTab>,
    );
  }

  dispose() {
    for (const subscription of this.subscriptions.splice(0)) subscription.dispose();
    this.parameters = null;
    const root = this.root;
    this.root = null;
    if (root) queueMicrotask(() => root.unmount());
  }
}

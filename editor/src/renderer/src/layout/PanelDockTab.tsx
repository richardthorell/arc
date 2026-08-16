import { createRoot, type Root } from 'react-dom/client';
import { X } from 'lucide-react';
import type { ITabRenderer, TabPartInitParameters } from 'dockview';

import { panelRegistry } from '../app/panelRegistry';
import type { WorkbenchPanelId } from '../app/workbenchTypes';

export type PanelTabPresentation = {
  title: string;
  icon: (typeof panelRegistry)[WorkbenchPanelId]['icon'];
  closeable: boolean;
};

const registryPanelId = (component: string): WorkbenchPanelId | null => {
  if (component === 'viewport' || component.startsWith('viewport-')) return 'viewport';
  return component in panelRegistry ? (component as WorkbenchPanelId) : null;
};

export function getPanelTabPresentation(component: string, instanceTitle?: string): PanelTabPresentation {
  const registryId = registryPanelId(component);
  const descriptor = registryId ? panelRegistry[registryId] : undefined;
  return {
    title: instanceTitle || descriptor?.title || component,
    icon: descriptor?.icon ?? null,
    closeable: descriptor?.closeable !== false,
  };
}

/**
 * Dockview owns the actual tab element so its drag/reorder/docking semantics stay
 * native. This renderer only supplies ARC's icon / label / close composition;
 * WorkspaceDock.css styles Dockview's outer .dv-tab with the same ARC tab
 * surface used by UiTab.
 */
export class PanelDockTabRenderer implements ITabRenderer {
  readonly element = document.createElement('div');
  private root: Root | null = null;
  private parameters: TabPartInitParameters | null = null;
  private subscriptions: Array<{ dispose(): void }> = [];

  constructor() {
    this.element.className = 'workspace-dock-tab-content';
  }

  init(parameters: TabPartInitParameters) {
    this.parameters = parameters;
    this.root = createRoot(this.element);
    this.subscriptions.push(parameters.api.onDidTitleChange(() => this.render()));
    this.render();
  }

  private render() {
    if (!this.parameters) return;
    const { api } = this.parameters;
    const presentation = getPanelTabPresentation(api.component || api.id, api.title);
    const Icon = presentation.icon;

    this.root?.render(
      <>
        {Icon && (
          <span aria-hidden="true" className="ui-tab-icon workspace-dock-tab-icon">
            <Icon size={14} />
          </span>
        )}
        <span className="ui-tab-label workspace-dock-tab-label">{presentation.title}</span>
        {presentation.closeable && (
          <button
            aria-label={`Close ${presentation.title}`}
            className="ui-tab-close workspace-dock-tab-close"
            onClick={(event) => {
              event.stopPropagation();
              api.close();
            }}
            onPointerDown={(event) => event.stopPropagation()}
            title={`Close ${presentation.title}`}
            type="button"
          >
            <X aria-hidden="true" size={12} />
          </button>
        )}
      </>,
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

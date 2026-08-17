import { createRoot, type Root } from 'react-dom/client';
import { useEffect, useRef, useState } from 'react';
import {
  createDockview,
  type DockviewApi,
  type GroupPanelPartInitParameters,
  type IContentRenderer,
  type SerializedDockview,
  themeAbyss,
} from 'dockview';
import 'dockview/dist/styles/dockview.css';

import {
  isSidebarPanel,
  panelRegistry,
  sidebarPanelIds,
  type SidebarPanelId,
} from '../app/panelRegistry';
import type { WorkbenchPanelId } from '../app/workbenchTypes';
import { PanelDockTabRenderer, getPanelTabPresentation } from './PanelDockTab';
import './WorkspaceDock.css';

export type WorkspaceLayoutName = 'Level Design' | 'Materials' | 'Profiling';

type WorkspaceDockProps = {
  projectKey: string;
  renderPanel: (panel: WorkbenchPanelId, instanceId?: string, onMaximizeToggle?: () => void) => React.ReactNode;
  requestedLayout?: WorkspaceLayoutName | 'Reset' | null;
  requestedPanel?: WorkbenchPanelId | null;
  onRequestHandled?: () => void;
  onReady?: (api: DockviewApi) => void;
  requestedViewportCount?: 1 | 2 | 3 | 4;
};

const storageKey = (projectKey: string, name: string) => `arc.editor.workspace.v3.${projectKey}.${name}`;
const panelTabComponent = 'arc-panel-tab';

class ReactPanelRenderer implements IContentRenderer {
  readonly element = document.createElement('div');
  private root: Root | null = null;
  private panel: WorkbenchPanelId;
  private renderPanel: () => WorkspaceDockProps['renderPanel'];
  private parameters: GroupPanelPartInitParameters | null = null;

  constructor(panel: WorkbenchPanelId, renderPanel: () => WorkspaceDockProps['renderPanel']) {
    this.panel = panel;
    this.renderPanel = renderPanel;
    this.element.className = `workspace-dock-panel workspace-dock-panel-${panel}`;
  }

  init(parameters: GroupPanelPartInitParameters) {
    this.parameters = parameters;
    this.root = createRoot(this.element);
    this.updateContent();
  }

  updateContent() {
    const viewportId = this.parameters?.params.viewportId as string | undefined;
    const toggleMaximize = () =>
      this.parameters?.api.isMaximized() ? this.parameters.api.exitMaximized() : this.parameters?.api.maximize();
    this.root?.render(this.renderPanel()(this.panel, viewportId, toggleMaximize));
  }

  dispose() {
    queueMicrotask(() => this.root?.unmount());
  }
}

const addPanel = (
  api: DockviewApi,
  panel: WorkbenchPanelId,
  referencePanel?: WorkbenchPanelId,
  direction?: 'left' | 'right' | 'above' | 'below' | 'within',
) => {
  if (isSidebarPanel(panel)) return;
  const descriptor = panelRegistry[panel];
  api.addPanel({
    id: panel,
    component: panel,
    tabComponent: panelTabComponent,
    title: descriptor.title,
    minimumWidth: descriptor.minimumWidth,
    minimumHeight: descriptor.minimumHeight,
    inactive: Boolean(referencePanel && direction === 'within'),
    ...(referencePanel ? { position: { referencePanel, direction } } : {}),
  });
};

const removeSidebarPanelsFromDock = (api: DockviewApi) => {
  for (const panelId of sidebarPanelIds) api.getPanel(panelId)?.api.close();
};

const createLayout = (api: DockviewApi, name: WorkspaceLayoutName) => {
  api.clear();
  addPanel(api, 'viewport');
  if (name === 'Materials') {
    addPanel(api, 'assetExplorer', 'viewport', 'left');
    addPanel(api, 'shaderEditor', 'viewport', 'within');
    addPanel(api, 'inspector', 'viewport', 'right');
    addPanel(api, 'contentBrowser', 'viewport', 'below');
    addPanel(api, 'console', 'contentBrowser', 'within');
    return;
  }
  if (name === 'Profiling') {
    addPanel(api, 'renderGraph', 'viewport', 'within');
    addPanel(api, 'profiler', 'viewport', 'right');
    addPanel(api, 'console', 'viewport', 'below');
    addPanel(api, 'buildOutput', 'console', 'within');
    return;
  }
  addPanel(api, 'inspector', 'viewport', 'right');
  addPanel(api, 'lighting', 'inspector', 'within');
  addPanel(api, 'worldSettings', 'inspector', 'within');
  addPanel(api, 'contentBrowser', 'viewport', 'below');
  addPanel(api, 'console', 'contentBrowser', 'within');
  addPanel(api, 'buildOutput', 'contentBrowser', 'within');
};

export function WorkspaceDock({
  projectKey,
  renderPanel,
  requestedLayout,
  requestedPanel,
  onRequestHandled,
  onReady,
  requestedViewportCount,
}: WorkspaceDockProps) {
  const host = useRef<HTMLDivElement | null>(null);
  const api = useRef<DockviewApi | null>(null);
  const renderPanelRef = useRef(renderPanel);
  const renderers = useRef(new Set<ReactPanelRenderer>());
  const [activeSidebarPanel, setActiveSidebarPanel] = useState<SidebarPanelId>('hierarchy');
  renderPanelRef.current = renderPanel;

  useEffect(() => {
    if (!host.current) return;
    const dock = createDockview(host.current, {
      theme: themeAbyss,
      floatingGroupDragHandle: 'titlebar',
      popoutUrl: window.location.href,
      defaultTabComponent: panelTabComponent,
      getTabContextMenuItems: ({ panel }) =>
        getPanelTabPresentation(panel.api.component, panel.api.title).closeable
          ? ['close', 'closeOthers', 'closeAll']
          : [],
      createTabComponent: ({ name }) => (name === panelTabComponent ? new PanelDockTabRenderer() : undefined),
      createComponent: ({ name }) => {
        const renderer = new ReactPanelRenderer(name as WorkbenchPanelId, () => renderPanelRef.current);
        renderers.current.add(renderer);
        const dispose = renderer.dispose.bind(renderer);
        renderer.dispose = () => {
          renderers.current.delete(renderer);
          dispose();
        };
        return renderer;
      },
    });
    api.current = dock;
    const saved = window.localStorage.getItem(storageKey(projectKey, 'current'));
    try {
      if (saved) dock.fromJSON(JSON.parse(saved) as SerializedDockview);
      else createLayout(dock, 'Level Design');
    } catch {
      createLayout(dock, 'Level Design');
    }
    // Layouts from older editor versions may still contain Hierarchy/Search/AI/VCS
    // as Dockview tabs. They now belong exclusively to the fixed primary sidebar.
    removeSidebarPanelsFromDock(dock);
    if (!dock.activePanel) createLayout(dock, 'Level Design');
    window.localStorage.setItem(storageKey(projectKey, 'current'), JSON.stringify(dock.toJSON()));

    const layoutSubscription = dock.onDidLayoutChange(() => {
      window.localStorage.setItem(storageKey(projectKey, 'current'), JSON.stringify(dock.toJSON()));
    });
    const observer = new ResizeObserver(() =>
      dock.layout(host.current?.clientWidth ?? 0, host.current?.clientHeight ?? 0),
    );
    observer.observe(host.current);
    onReady?.(dock);
    return () => {
      observer.disconnect();
      layoutSubscription.dispose();
      dock.dispose();
      api.current = null;
    };
  }, [onReady, projectKey]);

  useEffect(() => {
    for (const renderer of renderers.current) renderer.updateContent();
  });

  useEffect(() => {
    const dock = api.current;
    if (!dock || !requestedLayout) return;
    if (requestedLayout === 'Reset') createLayout(dock, 'Level Design');
    else {
      const saved = window.localStorage.getItem(storageKey(projectKey, requestedLayout));
      if (saved) dock.fromJSON(JSON.parse(saved) as SerializedDockview);
      else createLayout(dock, requestedLayout);
      removeSidebarPanelsFromDock(dock);
      if (!dock.activePanel) createLayout(dock, requestedLayout);
    }
    onRequestHandled?.();
  }, [onRequestHandled, projectKey, requestedLayout]);

  useEffect(() => {
    if (!requestedPanel) return;
    if (isSidebarPanel(requestedPanel)) {
      setActiveSidebarPanel(requestedPanel);
      onRequestHandled?.();
      return;
    }

    const dock = api.current;
    if (!dock) return;
    let panel = dock.getPanel(requestedPanel);
    if (!panel) {
      addPanel(dock, requestedPanel, dock.activePanel?.id as WorkbenchPanelId | undefined, 'within');
      panel = dock.getPanel(requestedPanel);
    }
    panel?.api.setActive();
    panel?.focus();
    onRequestHandled?.();
  }, [onRequestHandled, requestedPanel]);

  useEffect(() => {
    const dock = api.current;
    if (!dock || !requestedViewportCount) return;
    for (let index = 2; index <= 4; ++index) {
      const id = `viewport-${index}`;
      const existing = dock.getPanel(id);
      if (index <= requestedViewportCount && !existing) {
        dock.addPanel({
          id,
          component: 'viewport',
          tabComponent: panelTabComponent,
          title: `Viewport ${index}`,
          params: { viewportId: id },
          position: { referencePanel: 'viewport', direction: index % 2 === 0 ? 'right' : 'below' },
        });
      } else if (index > requestedViewportCount && existing) existing.api.close();
    }
  }, [requestedViewportCount]);

  return (
    <div className="workspace-dock-shell">
      <aside
        aria-label={`${panelRegistry[activeSidebarPanel].title} sidebar`}
        className={`primary-sidebar primary-sidebar-${activeSidebarPanel}`}
      >
        {renderPanel(activeSidebarPanel)}
      </aside>
      <div className="workspace-dock" ref={host} />
    </div>
  );
}

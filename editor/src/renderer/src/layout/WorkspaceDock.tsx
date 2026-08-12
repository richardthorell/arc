import { createRoot, type Root } from 'react-dom/client';
import { useEffect, useRef } from 'react';
import {
  createDockview,
  type DockviewApi,
  type GroupPanelPartInitParameters,
  type IContentRenderer,
  type SerializedDockview,
  themeAbyss,
} from 'dockview';
import 'dockview/dist/styles/dockview.css';

import { panelRegistry } from '../app/panelRegistry';
import type { WorkbenchPanelId } from '../app/workbenchTypes';

export type WorkspaceLayoutName = 'Level Design' | 'Materials' | 'Profiling';

type WorkspaceDockProps = {
  projectKey: string;
  renderPanel: (panel: WorkbenchPanelId) => React.ReactNode;
  requestedLayout?: WorkspaceLayoutName | 'Reset' | null;
  requestedPanel?: WorkbenchPanelId | null;
  onRequestHandled?: () => void;
  onReady?: (api: DockviewApi) => void;
};

const storageKey = (projectKey: string, name: string) => `arc.editor.workspace.v3.${projectKey}.${name}`;

class ReactPanelRenderer implements IContentRenderer {
  readonly element = document.createElement('div');
  private root: Root | null = null;
  private panel: WorkbenchPanelId;
  private renderPanel: () => (panel: WorkbenchPanelId) => React.ReactNode;

  constructor(panel: WorkbenchPanelId, renderPanel: () => (panel: WorkbenchPanelId) => React.ReactNode) {
    this.panel = panel;
    this.renderPanel = renderPanel;
    this.element.className = 'workspace-dock-panel';
  }

  init(_parameters: GroupPanelPartInitParameters) {
    this.root = createRoot(this.element);
    this.updateContent();
  }

  updateContent() {
    this.root?.render(this.renderPanel()(this.panel));
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
  const descriptor = panelRegistry[panel];
  api.addPanel({
    id: panel,
    component: panel,
    title: descriptor.title,
    minimumWidth: descriptor.minimumWidth,
    minimumHeight: descriptor.minimumHeight,
    inactive: Boolean(referencePanel && direction === 'within'),
    ...(referencePanel ? { position: { referencePanel, direction } } : {}),
  });
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
    addPanel(api, 'hierarchy', 'viewport', 'left');
    addPanel(api, 'renderGraph', 'viewport', 'within');
    addPanel(api, 'profiler', 'viewport', 'right');
    addPanel(api, 'console', 'viewport', 'below');
    addPanel(api, 'buildOutput', 'console', 'within');
    return;
  }
  addPanel(api, 'hierarchy', 'viewport', 'left');
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
}: WorkspaceDockProps) {
  const host = useRef<HTMLDivElement | null>(null);
  const api = useRef<DockviewApi | null>(null);
  const renderPanelRef = useRef(renderPanel);
  const renderers = useRef(new Set<ReactPanelRenderer>());
  renderPanelRef.current = renderPanel;

  useEffect(() => {
    if (!host.current) return;
    const dock = createDockview(host.current, {
      theme: themeAbyss,
      floatingGroupDragHandle: 'titlebar',
      popoutUrl: window.location.href,
      getTabContextMenuItems: () => ['close', 'closeOthers', 'closeAll'],
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
    }
    onRequestHandled?.();
  }, [onRequestHandled, projectKey, requestedLayout]);

  useEffect(() => {
    const dock = api.current;
    if (!dock || !requestedPanel) return;
    let panel = dock.getPanel(requestedPanel);
    if (!panel) {
      addPanel(dock, requestedPanel, dock.activePanel?.id as WorkbenchPanelId | undefined, 'within');
      panel = dock.getPanel(requestedPanel);
    }
    panel?.api.setActive();
    panel?.focus();
    onRequestHandled?.();
  }, [onRequestHandled, requestedPanel]);

  return <div className="workspace-dock" ref={host} />;
}

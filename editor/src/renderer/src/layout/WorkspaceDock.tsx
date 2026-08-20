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
  activityRegistry,
  isSidebarPanel,
  panelRegistry,
  sidebarPanelIds,
  type SidebarPanelId,
} from '../app/panelRegistry';
import type { WorkbenchPanelId } from '../app/workbenchTypes';
import { useEditorDocuments } from '../editors/editorDocuments';
import type { EditorDocumentKind } from '../editors/editorTypes';
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

// v5 restores Hierarchy as the default left-side Level Design panel after the
// global utility rail stopped owning it. Document-owned workspace snapshots use
// the same version so existing Level layouts can migrate without being lost.
const storageKey = (projectKey: string, name: string) => `arc.editor.workspace.v5.${projectKey}.${name}`;
const editorWorkspaceStorageKey = (projectKey: string, kind: EditorDocumentKind) =>
  storageKey(projectKey, `editor-${kind}`);
const workbenchLayoutStorageKey = 'arc.editor.workbench.layout.v2';
const panelTabComponent = 'arc-panel-tab';

const initialSidebarPanel = (): SidebarPanelId => {
  try {
    const saved = window.localStorage.getItem(workbenchLayoutStorageKey);
    if (!saved) return 'search';
    const activeActivity = (JSON.parse(saved) as { activeActivity?: string }).activeActivity;
    const activity = activityRegistry.find((entry) => entry.id === activeActivity);
    return activity && isSidebarPanel(activity.panelId) ? activity.panelId : 'search';
  } catch {
    return 'search';
  }
};

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

  // Hierarchy is a Level Editor/workspace panel, not a global utility. Keep it
  // open on the left in the default Level Design layout just as it was before
  // the utility rail was introduced.
  addPanel(api, 'hierarchy', 'viewport', 'left');
  addPanel(api, 'inspector', 'viewport', 'right');
  addPanel(api, 'lighting', 'inspector', 'within');
  addPanel(api, 'worldSettings', 'inspector', 'within');
  addPanel(api, 'contentBrowser', 'viewport', 'below');
  addPanel(api, 'console', 'contentBrowser', 'within');
  addPanel(api, 'buildOutput', 'contentBrowser', 'within');
};

const createEditorWorkspace = (api: DockviewApi, kind: EditorDocumentKind) => {
  if (kind === 'shader') {
    // Shader source currently has no document-specific supporting panels. The
    // editor host therefore owns the complete Dockview workspace. The global
    // utility rail/drawer lives outside this layout and remains available.
    api.clear();
    addPanel(api, 'viewport');
    return;
  }
  createLayout(api, 'Level Design');
};

const readEditorWorkspace = (projectKey: string, kind: EditorDocumentKind) => {
  const saved = window.localStorage.getItem(editorWorkspaceStorageKey(projectKey, kind));
  if (saved) return saved;
  // PR #71 stored the live Level Design layout under `current`. Use it as the
  // migration source the first time a document-owned Level workspace is used.
  return kind === 'level' ? window.localStorage.getItem(storageKey(projectKey, 'current')) : null;
};

const persistEditorWorkspace = (api: DockviewApi, projectKey: string, kind: EditorDocumentKind) => {
  const serialized = JSON.stringify(api.toJSON());
  window.localStorage.setItem(editorWorkspaceStorageKey(projectKey, kind), serialized);
  // Keep the legacy Level key current while the old layout presets still exist.
  if (kind === 'level') window.localStorage.setItem(storageKey(projectKey, 'current'), serialized);
};

const restoreEditorWorkspace = (api: DockviewApi, projectKey: string, kind: EditorDocumentKind) => {
  const saved = readEditorWorkspace(projectKey, kind);
  try {
    if (saved) api.fromJSON(JSON.parse(saved) as SerializedDockview);
    else createEditorWorkspace(api, kind);
  } catch {
    createEditorWorkspace(api, kind);
  }
  removeSidebarPanelsFromDock(api);
  if (!api.activePanel) createEditorWorkspace(api, kind);
  persistEditorWorkspace(api, projectKey, kind);
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
  const { activeDocument } = useEditorDocuments();
  const activeEditorKind: EditorDocumentKind = activeDocument?.kind ?? 'level';
  const dockEditorKind = useRef<EditorDocumentKind>(activeEditorKind);
  const [activeSidebarPanel, setActiveSidebarPanel] = useState<SidebarPanelId>(initialSidebarPanel);
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
    dockEditorKind.current = activeEditorKind;
    restoreEditorWorkspace(dock, projectKey, activeEditorKind);

    const layoutSubscription = dock.onDidLayoutChange(() => {
      persistEditorWorkspace(dock, projectKey, dockEditorKind.current);
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
    // Dockview is intentionally created once per project. Document changes are
    // restored into the existing instance by the effect below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [onReady, projectKey]);

  useEffect(() => {
    for (const renderer of renderers.current) renderer.updateContent();
  });

  useEffect(() => {
    const dock = api.current;
    if (!dock || dockEditorKind.current === activeEditorKind) return;

    persistEditorWorkspace(dock, projectKey, dockEditorKind.current);
    dockEditorKind.current = activeEditorKind;
    restoreEditorWorkspace(dock, projectKey, activeEditorKind);
  }, [activeEditorKind, projectKey]);

  useEffect(() => {
    const dock = api.current;
    if (!dock || !requestedLayout) return;
    // The existing named presets are Level Editor layouts. Asset documents own
    // their workspace and should not be replaced by a Level/Materials/Profiling
    // preset just because one was requested through legacy chrome.
    if (dockEditorKind.current !== 'level') {
      onRequestHandled?.();
      return;
    }
    if (requestedLayout === 'Reset') createLayout(dock, 'Level Design');
    else {
      const saved = window.localStorage.getItem(storageKey(projectKey, requestedLayout));
      if (saved) dock.fromJSON(JSON.parse(saved) as SerializedDockview);
      else createLayout(dock, requestedLayout);
      removeSidebarPanelsFromDock(dock);
      if (!dock.activePanel) createLayout(dock, requestedLayout);
    }
    persistEditorWorkspace(dock, projectKey, 'level');
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
    if (!dock || !requestedViewportCount || dockEditorKind.current !== 'level') return;
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
  }, [requestedViewportCount, activeEditorKind]);

  return (
    <div className={`workspace-dock-shell workspace-dock-shell-editor-${activeEditorKind}`}>
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

import { createRoot, type Root } from 'react-dom/client';
import { useEffect, useRef, useState, type CSSProperties, type KeyboardEvent as ReactKeyboardEvent } from 'react';
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
import type { ActivityId, WorkbenchPanelId } from '../app/workbenchTypes';
import type { EditorDocument, EditorDocumentKind } from '../editors/editorTypes';
import { PanelDockTabRenderer, getPanelTabPresentation } from './PanelDockTab';
import './WorkspaceDock.css';

export type WorkspaceLayoutName = 'Level Design' | 'Materials' | 'Profiling';

type WorkspaceDockProps = {
  document: EditorDocument;
  active: boolean;
  activeActivity: ActivityId;
  projectKey: string;
  renderPanel: (panel: WorkbenchPanelId, instanceId?: string, onMaximizeToggle?: () => void) => React.ReactNode;
  requestedLayout?: WorkspaceLayoutName | 'Reset' | null;
  requestedPanel?: WorkbenchPanelId | null;
  onRequestHandled?: () => void;
  onReady?: (api: DockviewApi) => void;
  requestedViewportCount?: 1 | 2 | 3 | 4;
  sidebarExpanded?: boolean;
};

// v8 groups scene structure and details into one wider right column: Hierarchy
// above Inspector, with the viewport and utility dock kept together on the left.
// The new key makes existing v7 snapshots pick up the new scene defaults.
const storageKey = (projectKey: string, name: string) => `arc.editor.workspace.v8.${projectKey}.${name}`;
export const editorWorkspaceStorageKey = (projectKey: string, kind: EditorDocumentKind) => {
  const versionedKind =
    kind === 'texture'
      ? 'editor-texture-v2'
      : kind === 'flow'
        ? 'editor-flow-v2'
        : kind === 'model' || kind === 'skeleton'
          ? `editor-${kind}-v2`
          : `editor-${kind}`;
  return storageKey(projectKey, versionedKind);
};
const workbenchLayoutStorageKey = 'arc.editor.workbench.layout.v2';
const panelTabComponent = 'arc-panel-tab';
const defaultBottomPanelHeight = 220;
export const defaultSceneRightColumnWidth = 560;
export const defaultHierarchyPanelHeight = 360;
const sidebarWidthStorageKey = 'arc.editor.utility-sidebar.width.v1';
export const defaultSidebarWidth = 320;
export const minimumSidebarWidth = 240;
export const maximumSidebarWidth = 640;

const documentOwnedWorkspaceKinds = new Set<EditorDocumentKind>([
  'shader',
  'material',
  'flow',
  'texture',
  'model',
  'skeleton',
]);

export const usesDocumentOwnedWorkspace = (kind: EditorDocumentKind) => documentOwnedWorkspaceKinds.has(kind);

export const supportsRequestedWorkspacePanel = (kind: EditorDocumentKind, panel: WorkbenchPanelId) =>
  kind === 'level' || panel === 'viewport' || isSidebarPanel(panel);

export const clampSidebarWidth = (value: number) =>
  Math.min(maximumSidebarWidth, Math.max(minimumSidebarWidth, Math.round(value)));

const initialSidebarWidth = () => {
  const saved = Number(window.localStorage.getItem(sidebarWidthStorageKey));
  return Number.isFinite(saved) && saved > 0 ? clampSidebarWidth(saved) : defaultSidebarWidth;
};

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
  initialHeight?: number,
  initialWidth?: number,
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
    initialHeight,
    initialWidth,
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

  // Keep scene structure and details together in a shared right column. Hierarchy
  // sits above Inspector, while Lighting and World Settings share the Hierarchy
  // group as tabs. The content/console/build group remains below the viewport only.
  addPanel(api, 'inspector', 'viewport', 'right', undefined, defaultSceneRightColumnWidth);
  addPanel(api, 'hierarchy', 'inspector', 'above', defaultHierarchyPanelHeight);
  addPanel(api, 'lighting', 'hierarchy', 'within');
  addPanel(api, 'worldSettings', 'hierarchy', 'within');
  addPanel(api, 'contentBrowser', 'viewport', 'below', defaultBottomPanelHeight);
  addPanel(api, 'console', 'contentBrowser', 'within');
  addPanel(api, 'buildOutput', 'contentBrowser', 'within');
};

const createEditorWorkspace = (api: DockviewApi, kind: EditorDocumentKind) => {
  if (usesDocumentOwnedWorkspace(kind)) {
    // Document editors own the complete workspace. Shader, Material, Flow, Texture,
    // Model, and Skeleton compose their document-specific supporting regions internally,
    // so Dockview only needs the primary EditorHost surface. The global utility
    // rail/drawer lives outside this layout and remains available.
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
  document,
  active,
  activeActivity,
  projectKey,
  renderPanel,
  requestedLayout,
  requestedPanel,
  onRequestHandled,
  onReady,
  requestedViewportCount,
  sidebarExpanded = false,
}: WorkspaceDockProps) {
  const shell = useRef<HTMLDivElement | null>(null);
  const host = useRef<HTMLDivElement | null>(null);
  const api = useRef<DockviewApi | null>(null);
  const renderPanelRef = useRef(renderPanel);
  const activeRef = useRef(active);
  const renderedActivityRef = useRef<boolean | null>(null);
  const renderers = useRef(new Set<ReactPanelRenderer>());
  const activeEditorKind = document.kind;
  const dockEditorKind = useRef<EditorDocumentKind>(activeEditorKind);
  const [activeSidebarPanel, setActiveSidebarPanel] = useState<SidebarPanelId>(initialSidebarPanel);
  const [sidebarWidth, setSidebarWidth] = useState(initialSidebarWidth);
  const sidebarWidthRef = useRef(sidebarWidth);
  const [resizingSidebar, setResizingSidebar] = useState(false);
  renderPanelRef.current = renderPanel;
  activeRef.current = active;

  const updateSidebarWidth = (value: number) => {
    const next = clampSidebarWidth(value);
    sidebarWidthRef.current = next;
    setSidebarWidth(next);
  };

  useEffect(() => {
    if (!resizingSidebar) return;

    const move = (event: PointerEvent) => {
      const bounds = shell.current?.getBoundingClientRect();
      if (bounds) updateSidebarWidth(event.clientX - bounds.left);
    };
    const stop = () => {
      setResizingSidebar(false);
      window.localStorage.setItem(sidebarWidthStorageKey, String(sidebarWidthRef.current));
    };

    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', stop, { once: true });
    window.addEventListener('pointercancel', stop, { once: true });
    return () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', stop);
      window.removeEventListener('pointercancel', stop);
    };
  }, [resizingSidebar]);

  const resizeSidebarWithKeyboard = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    let next = sidebarWidthRef.current;
    if (event.key === 'ArrowLeft') next -= 16;
    else if (event.key === 'ArrowRight') next += 16;
    else if (event.key === 'Home') next = minimumSidebarWidth;
    else if (event.key === 'End') next = maximumSidebarWidth;
    else return;
    event.preventDefault();
    updateSidebarWidth(next);
    window.localStorage.setItem(sidebarWidthStorageKey, String(clampSidebarWidth(next)));
  };

  useEffect(() => {
    const panel = activityRegistry.find((entry) => entry.id === activeActivity)?.panelId;
    if (panel && isSidebarPanel(panel)) setActiveSidebarPanel(panel);
  }, [activeActivity]);

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
      if (activeRef.current) persistEditorWorkspace(dock, projectKey, dockEditorKind.current);
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
    // Each open document owns its Dockview instance. Hiding a document must not
    // dispose its panels or the native viewport resources they own.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [onReady, projectKey]);

  useEffect(() => {
    const activityChanged = renderedActivityRef.current !== active;
    renderedActivityRef.current = active;
    // Publish the hide transition once, then avoid rerendering parked editors
    // for unrelated workbench updates until their document is shown again.
    if (!active && !activityChanged) return;
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
    if (!supportsRequestedWorkspacePanel(dockEditorKind.current, requestedPanel)) {
      onRequestHandled?.();
      return;
    }
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
    <div
      className={`workspace-dock-shell workspace-dock-shell-editor-${activeEditorKind}`}
      ref={shell}
      style={{ '--arc-utility-sidebar-width': `${sidebarWidth}px` } as CSSProperties}
    >
      <aside
        aria-label={`${panelRegistry[activeSidebarPanel].title} sidebar`}
        className={`primary-sidebar primary-sidebar-${activeSidebarPanel}`}
      >
        {active && renderPanel(activeSidebarPanel)}
        {sidebarExpanded && (
          <div
            aria-label="Resize utility sidebar"
            aria-orientation="vertical"
            aria-valuemax={maximumSidebarWidth}
            aria-valuemin={minimumSidebarWidth}
            aria-valuenow={sidebarWidth}
            className={`primary-sidebar-resize-handle${resizingSidebar ? ' is-resizing' : ''}`}
            onKeyDown={resizeSidebarWithKeyboard}
            onPointerDown={(event) => {
              event.preventDefault();
              event.currentTarget.setPointerCapture?.(event.pointerId);
              setResizingSidebar(true);
            }}
            role="separator"
            tabIndex={0}
          />
        )}
      </aside>
      <div className="workspace-dock" ref={host} />
    </div>
  );
}

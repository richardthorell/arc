import { useCallback, useMemo, useSyncExternalStore } from 'react';

import type { EditorDocument } from '../editors/editorTypes';

/** Workspace tabs can host saved assets or editor tools; neither requires the other's lifecycle. */
export type WorkspaceAssetTab = { type: 'asset'; id: string; document: EditorDocument };
export type WorkspaceViewTab = {
  type: 'view';
  id: string;
  viewId: string;
  title: string;
  closeable: boolean;
};
export type WorkspaceTab = WorkspaceAssetTab | WorkspaceViewTab;

type ViewTabState = { views: WorkspaceViewTab[]; activeViewId: string | null };

const emptyState: ViewTabState = { views: [], activeViewId: null };
let state = emptyState;
const listeners = new Set<() => void>();

const publish = (next: ViewTabState) => {
  if (next === state) return;
  state = next;
  for (const listener of listeners) listener();
};

const subscribe = (listener: () => void) => {
  listeners.add(listener);
  return () => listeners.delete(listener);
};

const getSnapshot = () => state;

export const viewTabId = (viewId: string) => `view:${viewId}`;

/** One tab for each view ID; opening an existing view activates rather than duplicates it. */
export const openWorkspaceView = (viewId: string, title: string, closeable = true) => {
  const id = viewTabId(viewId);
  const view = { type: 'view' as const, id, viewId, title, closeable };
  const views = state.views.some((item) => item.id === id)
    ? state.views.map((item) => (item.id === id ? view : item))
    : [...state.views, view];
  publish({ views, activeViewId: id });
};

export const activateWorkspaceView = (id: string) => {
  if (state.views.some((view) => view.id === id)) publish({ ...state, activeViewId: id });
};

/** Asset activation clears only the focused tool, not its open tab or its running service. */
export const deactivateWorkspaceView = () => {
  if (state.activeViewId) publish({ ...state, activeViewId: null });
};

export const closeWorkspaceView = (id: string) => {
  const views = state.views.filter((view) => view.id !== id || !view.closeable);
  if (views.length === state.views.length) return;
  publish({
    views,
    activeViewId: state.activeViewId === id ? (views.at(-1)?.id ?? null) : state.activeViewId,
  });
};

export const resetWorkspaceViews = () => publish(emptyState);

export const useWorkspaceTabs = (documents: readonly EditorDocument[], activeDocumentId: string | null) => {
  const viewState = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
  const tabs = useMemo<WorkspaceTab[]>(
    () => [
      ...documents.map((document): WorkspaceAssetTab => ({ type: 'asset', id: document.id, document })),
      ...viewState.views,
    ],
    [documents, viewState.views],
  );

  const activateTab = useCallback((id: string, activateAsset: (documentId: string) => void) => {
    if (state.views.some((view) => view.id === id)) activateWorkspaceView(id);
    else {
      deactivateWorkspaceView();
      activateAsset(id);
    }
  }, []);

  return {
    tabs,
    activeTabId: viewState.activeViewId ?? activeDocumentId,
    activeView: viewState.views.find((view) => view.id === viewState.activeViewId) ?? null,
    activateTab,
  };
};

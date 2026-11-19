import type { ReactNode } from 'react';

import type { WorkspaceViewTab } from '../app/workspaceTabs';
import type { EditorDocument, EditorRegistry } from './editorTypes';
import { EditorToolbarHost } from './EditorHost';

/**
 * Always reserve the same workbench grid row. Asset editors provide their
 * registered toolbar; tool tabs can provide an independent toolbar instead.
 */
export function WorkspaceToolbarHost({
  document,
  activeView,
  registry,
  buildsToolbar,
}: {
  document: EditorDocument | null;
  activeView: WorkspaceViewTab | null;
  registry: EditorRegistry;
  buildsToolbar?: ReactNode;
}) {
  if (activeView?.viewId === 'builds') {
    return <div className="editor-toolbar-host editor-toolbar-builds">{buildsToolbar}</div>;
  }
  return <EditorToolbarHost document={activeView ? null : document} registry={registry} />;
}

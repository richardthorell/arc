import type { WorkspaceViewTab } from '../app/workspaceTabs';
import type { EditorDocument, EditorRegistry } from './editorTypes';
import { EditorToolbarHost } from './EditorHost';

/**
 * The workbench grid reserves a toolbar row for every workspace tab.
 * Tool views do not show asset actions, but must still render an empty
 * toolbar slot so that the center region and status bar retain their rows.
 */
export function WorkspaceToolbarHost({
  document,
  activeView,
  registry,
}: {
  document: EditorDocument | null;
  activeView: WorkspaceViewTab | null;
  registry: EditorRegistry;
}) {
  return <EditorToolbarHost document={activeView ? null : document} registry={registry} />;
}

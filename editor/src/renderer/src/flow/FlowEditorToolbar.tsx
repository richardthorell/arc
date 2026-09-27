import { Lock, RefreshCw, RotateCcw, RotateCw, Save, Workflow } from 'lucide-react';

import type { EditorDocument } from '../editors/editorTypes';
import { UiButton, UiEditorToolbar, UiToolbarGroup, UiToolbarSeparator } from '../ui';
import {
  redoFlowGraph,
  reloadFlowDocument,
  saveFlowDocument,
  undoFlowGraph,
  useFlowDocumentState,
} from './flowDocumentState';

import '../tools/tools.css';

export function FlowEditorToolbar({ document }: { document: EditorDocument }) {
  const state = useFlowDocumentState(document);
  const busy = state.loading || state.saving;
  const canUndo = !document.readOnly && state.historyIndex > 0;
  const canRedo = !document.readOnly && state.historyIndex + 1 < state.history.length;

  return (
    <UiEditorToolbar
      className="flow-document-toolbar"
      left={
        <>
          <UiToolbarGroup className="flow-document-toolbar-label">
            <Workflow size={15} />
            <span>Flow</span>
          </UiToolbarGroup>
          <UiToolbarSeparator />
          <UiToolbarGroup>
            <UiButton
              disabled={busy || document.readOnly || !document.dirty}
              onClick={() => void saveFlowDocument(document)}
              variant="toolbar"
            >
              <Save size={13} /> Save
            </UiButton>
            <UiButton disabled={busy} onClick={() => void reloadFlowDocument(document)} variant="toolbar">
              <RefreshCw size={13} /> Reload
            </UiButton>
          </UiToolbarGroup>
          <UiToolbarSeparator />
          <UiToolbarGroup>
            <UiButton disabled={!canUndo} onClick={() => undoFlowGraph(document)} variant="toolbar">
              <RotateCcw size={13} /> Undo
            </UiButton>
            <UiButton disabled={!canRedo} onClick={() => redoFlowGraph(document)} variant="toolbar">
              <RotateCw size={13} /> Redo
            </UiButton>
          </UiToolbarGroup>
        </>
      }
      right={
        <>
          {document.readOnly && (
            <UiToolbarGroup className="flow-document-readonly">
              <Lock size={13} /> Read-only
            </UiToolbarGroup>
          )}
          <UiToolbarGroup className="flow-document-runtime-state">Authoring only · runtime in F3</UiToolbarGroup>
        </>
      }
    />
  );
}

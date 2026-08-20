import { Check, Circle, Lock, RefreshCw, RotateCcw, RotateCw, Save, Upload } from 'lucide-react';

import type { EditorDocument } from '../editors/editorTypes';
import { UiButton } from '../ui';
import {
  compileMaterialDocument,
  redoMaterialGraph,
  reloadMaterialDocument,
  saveAndPublishMaterialDocument,
  saveMaterialDocument,
  undoMaterialGraph,
  useMaterialDocumentState,
} from './materialDocumentState';

import '../tools/tools.css';

export function MaterialEditorToolbar({ document }: { document: EditorDocument }) {
  const state = useMaterialDocumentState(document);
  const busy = state.loading || state.saving || state.compiling;
  const canUndo = !document.readOnly && state.historyIndex > 0;
  const canRedo = !document.readOnly && state.historyIndex + 1 < state.history.length;

  return (
    <div className="main-toolbar material-document-toolbar">
      <div className="toolbar-left">
        <span className="toolbar-group material-document-toolbar-label">
          <Circle size={15} />
          <span>Material</span>
        </span>
        <span className="toolbar-separator" />
        <UiButton
          disabled={busy || document.readOnly || !document.dirty}
          onClick={() => void saveMaterialDocument(document)}
          variant="toolbar"
        >
          <Save size={13} /> Save
        </UiButton>
        <UiButton disabled={busy} onClick={() => void compileMaterialDocument(document)} variant="toolbar">
          <Check size={13} /> Compile
        </UiButton>
        <UiButton
          disabled={busy || document.readOnly || !document.assetGuid}
          onClick={() => void saveAndPublishMaterialDocument(document)}
          variant="primary"
        >
          <Upload size={13} /> Save &amp; Compile
        </UiButton>
        <UiButton disabled={busy} onClick={() => void reloadMaterialDocument(document)} variant="toolbar">
          <RefreshCw size={13} /> Reload
        </UiButton>
        <span className="toolbar-separator" />
        <UiButton disabled={!canUndo} onClick={() => undoMaterialGraph(document)} variant="toolbar">
          <RotateCcw size={13} /> Undo
        </UiButton>
        <UiButton disabled={!canRedo} onClick={() => redoMaterialGraph(document)} variant="toolbar">
          <RotateCw size={13} /> Redo
        </UiButton>
      </div>
      <div className="toolbar-right">
        {document.readOnly && (
          <span className="toolbar-group material-document-readonly">
            <Lock size={13} /> Read-only
          </span>
        )}
        <span className="toolbar-group material-document-compile-state">
          {state.compilation.succeeded ? 'IR ready' : 'Graph errors'} · {state.compilation.ir.expressions.length}{' '}
          expressions
        </span>
      </div>
    </div>
  );
}

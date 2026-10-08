import type { EditorDocument } from '../editors/editorTypes';
import { UiButton } from '../ui';
import {
  redoMaterialInstanceDocument,
  refreshMaterialInstancePreview,
  saveMaterialInstanceDocument,
  undoMaterialInstanceDocument,
  useMaterialInstanceDocumentState,
} from './materialInstanceDocumentState';

export function MaterialInstanceEditorToolbar({ document }: { document: EditorDocument }) {
  const state = useMaterialInstanceDocumentState(document);
  return (
    <div className="editor-toolbar">
      <UiButton disabled={state.historyIndex <= 0} onClick={() => void undoMaterialInstanceDocument(document)}>
        Undo
      </UiButton>
      <UiButton
        disabled={state.historyIndex + 1 >= state.history.length}
        onClick={() => void redoMaterialInstanceDocument(document)}
      >
        Redo
      </UiButton>
      <UiButton disabled={state.previewLoading} onClick={() => void refreshMaterialInstancePreview(document)}>
        {state.previewLoading ? 'Refreshing…' : 'Refresh Preview'}
      </UiButton>
      <UiButton
        disabled={document.readOnly || state.saving}
        onClick={() => void saveMaterialInstanceDocument(document)}
      >
        {state.saving ? 'Saving…' : 'Save'}
      </UiButton>
    </div>
  );
}

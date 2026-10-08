import type { EditorDocument } from '../editors/editorTypes';
import { UiButton } from '../ui';
import {
  saveMaterialFunctionDocument,
  useMaterialFunctionDocumentState,
  validateMaterialFunctionDocument,
} from './materialFunctionDocumentState';

export function MaterialFunctionEditorToolbar({ document }: { document: EditorDocument }) {
  const state = useMaterialFunctionDocumentState(document);
  return (
    <div className="editor-toolbar">
      <UiButton disabled={state.validating} onClick={() => void validateMaterialFunctionDocument(document)}>
        {state.validating ? 'Validating…' : 'Validate'}
      </UiButton>
      <UiButton
        disabled={document.readOnly || state.saving || state.validating}
        onClick={() => void saveMaterialFunctionDocument(document)}
      >
        {state.saving ? 'Saving…' : 'Save'}
      </UiButton>
    </div>
  );
}

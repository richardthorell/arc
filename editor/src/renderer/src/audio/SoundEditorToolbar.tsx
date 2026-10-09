import type { EditorDocument } from '../editors/editorTypes';
import { UiButton } from '../ui';
import { saveSoundDocument, useSoundDocumentState } from './soundDocumentState';

export function SoundEditorToolbar({ document }: { document: EditorDocument }) {
  const state = useSoundDocumentState(document);
  return (
    <div className="editor-toolbar">
      <UiButton
        disabled={document.readOnly || state.saving || !state.loaded}
        onClick={() => void saveSoundDocument(document)}
      >
        {state.saving ? 'Saving…' : 'Save'}
      </UiButton>
    </div>
  );
}

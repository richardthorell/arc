import { describe, expect, it } from 'vitest';

import {
  activateEditorDocument,
  closeEditorDocument,
  emptyEditorDocumentsState,
  openEditorDocument,
  syncSingletonEditorDocument,
} from './editorDocuments';
import type { EditorDocument } from './editorTypes';

const level = (id: string, title = 'World'): EditorDocument => ({
  id,
  kind: 'level',
  title,
  dirty: false,
  readOnly: false,
});

describe('editor document state', () => {
  it('keeps the level editor singleton while refreshing its metadata', () => {
    const opened = openEditorDocument(emptyEditorDocumentsState, level('level:a'), false);
    const synced = syncSingletonEditorDocument(opened, 'level', { ...level('level:b', 'Updated'), dirty: true });

    expect(synced.documents).toEqual([{ ...level('level:b', 'Updated'), dirty: true }]);
    expect(synced.activeDocumentId).toBe('level:b');
  });

  it('supports generic activation and close behavior for future document kinds', () => {
    const opened = openEditorDocument(emptyEditorDocumentsState, level('level:a'));
    expect(activateEditorDocument(opened, 'level:a').activeDocumentId).toBe('level:a');
    expect(closeEditorDocument(opened, 'level:a')).toEqual(emptyEditorDocumentsState);
  });
});

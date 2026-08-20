import { describe, expect, it } from 'vitest';

import {
  activateEditorDocument,
  closeEditorDocument,
  emptyEditorDocumentsState,
  openEditorDocument,
  syncSingletonEditorDocument,
  updateEditorDocument,
} from './editorDocuments';
import type { EditorDocument } from './editorTypes';

const level = (id: string, title = 'World'): EditorDocument => ({
  id,
  kind: 'level',
  title,
  dirty: false,
  readOnly: false,
});

const shader = (id: string, title = 'pbr_lit.hlsl'): EditorDocument => ({
  id,
  kind: 'shader',
  title,
  path: `Assets/Shaders/${title}`,
  assetId: id,
  assetGuid: id,
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

  it('keeps multiple shader documents alongside the singleton level document', () => {
    const withLevel = openEditorDocument(emptyEditorDocumentsState, level('level:world'), false);
    const withFirstShader = openEditorDocument(withLevel, shader('shader:a'));
    const withSecondShader = openEditorDocument(withFirstShader, shader('shader:b', 'shadow.hlsl'));

    expect(withSecondShader.documents.map((document) => document.id)).toEqual([
      'level:world',
      'shader:a',
      'shader:b',
    ]);
    expect(withSecondShader.activeDocumentId).toBe('shader:b');
  });

  it('preserves dirty state when an already-open asset is activated again', () => {
    const opened = openEditorDocument(emptyEditorDocumentsState, shader('shader:a'));
    const dirty = updateEditorDocument(opened, 'shader:a', { dirty: true });
    const reopened = openEditorDocument(dirty, { ...shader('shader:a'), title: 'Renamed.hlsl' });

    expect(reopened.documents[0]).toMatchObject({ title: 'Renamed.hlsl', dirty: true });
    expect(reopened.activeDocumentId).toBe('shader:a');
  });

  it('supports generic activation, updates, and close behavior', () => {
    const opened = openEditorDocument(emptyEditorDocumentsState, shader('shader:a'));
    const updated = updateEditorDocument(opened, 'shader:a', { dirty: true });
    expect(updated.documents[0].dirty).toBe(true);
    expect(activateEditorDocument(updated, 'shader:a').activeDocumentId).toBe('shader:a');
    expect(closeEditorDocument(updated, 'shader:a')).toEqual(emptyEditorDocumentsState);
  });
});

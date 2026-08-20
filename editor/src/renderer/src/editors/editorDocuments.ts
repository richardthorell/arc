import { useCallback, useMemo, useState } from 'react';

import type { EditorDocument, EditorDocumentKind } from './editorTypes';

export type EditorDocumentsState = {
  documents: EditorDocument[];
  activeDocumentId: string | null;
};

export const emptyEditorDocumentsState: EditorDocumentsState = {
  documents: [],
  activeDocumentId: null,
};

export const openEditorDocument = (
  state: EditorDocumentsState,
  document: EditorDocument,
  allowMultiple = true,
): EditorDocumentsState => {
  if (!allowMultiple) {
    const firstSameKind = state.documents.findIndex((entry) => entry.kind === document.kind);
    const withoutKind = state.documents.filter((entry) => entry.kind !== document.kind);
    const insertionIndex = firstSameKind < 0 ? withoutKind.length : Math.min(firstSameKind, withoutKind.length);
    const documents = [...withoutKind];
    documents.splice(insertionIndex, 0, document);
    return { documents, activeDocumentId: document.id };
  }

  const existingIndex = state.documents.findIndex((entry) => entry.id === document.id);
  if (existingIndex < 0)
    return { documents: [...state.documents, document], activeDocumentId: document.id };

  const documents = [...state.documents];
  documents[existingIndex] = document;
  return { documents, activeDocumentId: document.id };
};

export const syncSingletonEditorDocument = (
  state: EditorDocumentsState,
  kind: EditorDocumentKind,
  document: EditorDocument | null,
): EditorDocumentsState => {
  const active = state.documents.find((entry) => entry.id === state.activeDocumentId);
  const firstSameKind = state.documents.findIndex((entry) => entry.kind === kind);
  const withoutKind = state.documents.filter((entry) => entry.kind !== kind);

  if (!document) {
    const activeDocumentId = active?.kind === kind ? (withoutKind[0]?.id ?? null) : state.activeDocumentId;
    return { documents: withoutKind, activeDocumentId };
  }

  const insertionIndex = firstSameKind < 0 ? withoutKind.length : Math.min(firstSameKind, withoutKind.length);
  const documents = [...withoutKind];
  documents.splice(insertionIndex, 0, document);
  const activeDocumentId = !state.activeDocumentId || active?.kind === kind ? document.id : state.activeDocumentId;
  return { documents, activeDocumentId };
};

export const closeEditorDocument = (state: EditorDocumentsState, documentId: string): EditorDocumentsState => {
  const index = state.documents.findIndex((entry) => entry.id === documentId);
  if (index < 0) return state;
  const documents = state.documents.filter((entry) => entry.id !== documentId);
  if (state.activeDocumentId !== documentId) return { ...state, documents };
  return {
    documents,
    activeDocumentId: documents[Math.min(index, documents.length - 1)]?.id ?? null,
  };
};

export const activateEditorDocument = (state: EditorDocumentsState, documentId: string): EditorDocumentsState =>
  state.documents.some((entry) => entry.id === documentId) ? { ...state, activeDocumentId: documentId } : state;

export const useEditorDocuments = () => {
  const [state, setState] = useState<EditorDocumentsState>(emptyEditorDocumentsState);

  const openDocument = useCallback((document: EditorDocument, allowMultiple = true) => {
    setState((current) => openEditorDocument(current, document, allowMultiple));
  }, []);

  const syncSingletonDocument = useCallback((kind: EditorDocumentKind, document: EditorDocument | null) => {
    setState((current) => syncSingletonEditorDocument(current, kind, document));
  }, []);

  const activateDocument = useCallback((documentId: string) => {
    setState((current) => activateEditorDocument(current, documentId));
  }, []);

  const closeDocument = useCallback((documentId: string) => {
    setState((current) => closeEditorDocument(current, documentId));
  }, []);

  const activeDocument = useMemo(
    () => state.documents.find((entry) => entry.id === state.activeDocumentId) ?? null,
    [state.activeDocumentId, state.documents],
  );

  return {
    ...state,
    activeDocument,
    openDocument,
    syncSingletonDocument,
    activateDocument,
    closeDocument,
  };
};

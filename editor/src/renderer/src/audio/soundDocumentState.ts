import { useEffect, useState } from 'react';

import { updateEditorDocumentInStore } from '../editors/editorDocuments';
import type { EditorDocument } from '../editors/editorTypes';
import { setPathValue } from '../inspector/propertySchema';
import { createDefaultSoundAsset, parseSoundAsset, type SoundAsset } from './soundTypes';

export type SoundDocumentState = {
  documentId: string;
  path: string;
  readOnly: boolean;
  asset: SoundAsset;
  confirmed: string;
  loaded: boolean;
  loading: boolean;
  saving: boolean;
  message: string;
};

const states = new Map<string, SoundDocumentState>();
const listeners = new Map<string, Set<() => void>>();

const serializeSoundAsset = (asset: SoundAsset) => `${JSON.stringify(asset, null, 2)}\n`;

const initialState = (document: EditorDocument): SoundDocumentState => ({
  documentId: document.id,
  path: document.path ?? '',
  readOnly: document.readOnly,
  asset: createDefaultSoundAsset(),
  confirmed: '',
  loaded: false,
  loading: false,
  saving: false,
  message: '',
});

const emit = (documentId: string) => {
  for (const listener of listeners.get(documentId) ?? []) listener();
};

const setState = (documentId: string, patch: Partial<SoundDocumentState>) => {
  const current = states.get(documentId);
  if (!current) return;
  states.set(documentId, { ...current, ...patch });
  emit(documentId);
};

const ensureState = (document: EditorDocument) => {
  const current = states.get(document.id);
  if (!current || current.path !== (document.path ?? '')) {
    const next = initialState(document);
    states.set(document.id, next);
    return next;
  }
  if (current.readOnly !== document.readOnly) {
    const next = { ...current, readOnly: document.readOnly };
    states.set(document.id, next);
    return next;
  }
  return current;
};

const subscribe = (documentId: string, listener: () => void) => {
  const set = listeners.get(documentId) ?? new Set<() => void>();
  set.add(listener);
  listeners.set(documentId, set);
  return () => {
    set.delete(listener);
    if (set.size === 0) listeners.delete(documentId);
  };
};

const updateDirtyState = (document: EditorDocument, asset: SoundAsset, confirmed: string) =>
  updateEditorDocumentInStore(document.id, {
    dirty: !document.readOnly && serializeSoundAsset(asset) !== confirmed,
  });

export const loadSoundDocument = async (document: EditorDocument, force = false): Promise<boolean> => {
  const current = ensureState(document);
  if (!document.path) return false;
  if (!force && (current.loaded || current.loading)) return true;

  setState(document.id, { loading: true, message: '' });
  try {
    const file = await window.arc.projects.readText(
      document.path,
      document.assetScope === 'builtin' ? 'builtin' : 'project',
    );
    const asset = parseSoundAsset(JSON.parse(file.text));
    const confirmed = serializeSoundAsset(asset);
    setState(document.id, {
      asset,
      confirmed,
      loaded: true,
      loading: false,
      message: document.readOnly ? 'Engine Sound opened read-only' : '',
    });
    updateEditorDocumentInStore(document.id, { dirty: false });
    return true;
  } catch (error) {
    setState(document.id, {
      loading: false,
      message: error instanceof Error ? error.message : String(error),
    });
    return false;
  }
};

export const setSoundDocumentValue = (
  document: EditorDocument,
  path: string,
  value: unknown,
  settled = true,
) => {
  const current = ensureState(document);
  if (current.readOnly || document.readOnly) return;
  const next = setPathValue(current.asset, path, value);
  setState(document.id, { asset: next, message: settled ? '' : current.message });
  updateDirtyState(document, next, current.confirmed);
};

export const saveSoundDocument = async (document: EditorDocument): Promise<boolean> => {
  const current = ensureState(document);
  if (current.readOnly || document.readOnly || !document.path) return false;

  try {
    const validated = parseSoundAsset(current.asset);
    const text = serializeSoundAsset(validated);
    setState(document.id, { saving: true, message: '' });
    await window.arc.projects.writeText(document.path, text);
    setState(document.id, {
      asset: validated,
      saving: false,
      confirmed: text,
      message: 'Sound saved',
    });
    updateEditorDocumentInStore(document.id, { dirty: false });
    return true;
  } catch (error) {
    setState(document.id, {
      saving: false,
      message: error instanceof Error ? error.message : String(error),
    });
    return false;
  }
};

export const disposeSoundDocument = (documentId: string) => {
  states.delete(documentId);
  listeners.delete(documentId);
};

export const useSoundDocumentState = (document: EditorDocument) => {
  const [, rerender] = useState(0);
  const state = ensureState(document);

  useEffect(() => subscribe(document.id, () => rerender((value) => value + 1)), [document.id]);
  useEffect(() => {
    void loadSoundDocument(document);
  }, [document]);

  return states.get(document.id) ?? state;
};

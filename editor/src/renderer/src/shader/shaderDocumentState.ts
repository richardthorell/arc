import { useEffect, useState } from 'react';

import type { EditorDocument } from '../editors/editorTypes';
import { updateEditorDocumentInStore } from '../editors/editorDocuments';
import type { HostResponse } from '../inspector/inspectorTypes';

export type ShaderDocumentState = {
  documentId: string;
  path: string;
  guid?: string;
  scope?: EditorDocument['assetScope'];
  readOnly: boolean;
  source: string;
  confirmed: string;
  modifiedAt: string;
  message: string;
  compiling: boolean;
  loading: boolean;
  loaded: boolean;
};

const states = new Map<string, ShaderDocumentState>();
const listeners = new Map<string, Set<() => void>>();

const initialState = (document: EditorDocument): ShaderDocumentState => ({
  documentId: document.id,
  path: document.path ?? '',
  guid: document.assetGuid,
  scope: document.assetScope,
  readOnly: document.readOnly,
  source: '',
  confirmed: '',
  modifiedAt: '',
  message: '',
  compiling: false,
  loading: false,
  loaded: false,
});

const emit = (documentId: string) => {
  for (const listener of listeners.get(documentId) ?? []) listener();
};

const setState = (documentId: string, patch: Partial<ShaderDocumentState>) => {
  const current = states.get(documentId);
  if (!current) return;
  states.set(documentId, { ...current, ...patch });
  emit(documentId);
};

const ensureState = (document: EditorDocument) => {
  const current = states.get(document.id);
  if (!current) {
    const next = initialState(document);
    states.set(document.id, next);
    return next;
  }

  if (current.path !== (document.path ?? '') || current.scope !== document.assetScope) {
    const next = initialState(document);
    states.set(document.id, next);
    emit(document.id);
    return next;
  }

  if (current.guid !== document.assetGuid || current.readOnly !== document.readOnly) {
    const next = { ...current, guid: document.assetGuid, readOnly: document.readOnly };
    states.set(document.id, next);
    return next;
  }

  return current;
};

const subscribe = (documentId: string, listener: () => void) => {
  const documentListeners = listeners.get(documentId) ?? new Set<() => void>();
  documentListeners.add(listener);
  listeners.set(documentId, documentListeners);
  return () => {
    documentListeners.delete(listener);
    if (documentListeners.size === 0) listeners.delete(documentId);
  };
};

export const getShaderDocumentState = (document: EditorDocument) => ensureState(document);

export const loadShaderDocument = async (document: EditorDocument, force = false): Promise<boolean> => {
  const current = ensureState(document);
  if (!document.path) {
    setState(document.id, { message: 'Shader source path is unavailable' });
    return false;
  }
  if (!force && (current.loaded || current.loading)) return true;

  setState(document.id, { loading: true, message: '' });
  try {
    const scope = document.assetScope === 'builtin' ? 'builtin' : 'project';
    const file = await window.arc.projects.readText(document.path, scope);
    const latest = states.get(document.id);
    if (!latest || latest.path !== document.path || latest.scope !== document.assetScope) return false;
    setState(document.id, {
      source: file.text,
      confirmed: file.text,
      modifiedAt: file.modifiedAt,
      loading: false,
      loaded: true,
      message: '',
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

export const setShaderDocumentSource = (document: EditorDocument, source: string) => {
  const current = ensureState(document);
  if (document.readOnly || current.readOnly) return;
  setState(document.id, { source });
  updateEditorDocumentInStore(document.id, { dirty: source !== current.confirmed });
};

export const saveShaderDocument = async (document: EditorDocument): Promise<boolean> => {
  const current = ensureState(document);
  if (document.readOnly || current.readOnly) {
    setState(document.id, { message: 'This shader is read-only' });
    return false;
  }
  if (!document.path) {
    setState(document.id, { message: 'Shader source path is unavailable' });
    return false;
  }

  const savedSource = current.source;
  try {
    await window.arc.projects.writeText(document.path, savedSource);
    const latest = states.get(document.id) ?? current;
    setState(document.id, {
      confirmed: savedSource,
      modifiedAt: new Date().toISOString(),
      message: 'Shader saved',
    });
    updateEditorDocumentInStore(document.id, { dirty: latest.source !== savedSource });
    return true;
  } catch (error) {
    setState(document.id, { message: error instanceof Error ? error.message : String(error) });
    return false;
  }
};

export const compileShaderDocument = async (document: EditorDocument): Promise<boolean> => {
  const current = ensureState(document);
  if (document.readOnly || current.readOnly) {
    setState(document.id, { message: 'Built-in shaders are read-only and cannot be recompiled from the editor' });
    return false;
  }
  const guid = document.assetGuid ?? current.guid;
  if (!guid) {
    setState(document.id, { message: 'This shader has no registered asset GUID and cannot be compiled' });
    return false;
  }

  setState(document.id, { compiling: true });
  try {
    const response = (await window.arc.host.command('asset.reimport', { guid })) as HostResponse;
    setState(document.id, {
      compiling: false,
      message: response.succeeded ? 'Shader compilation queued' : response.error || 'Shader compilation failed',
    });
    return response.succeeded;
  } catch (error) {
    setState(document.id, {
      compiling: false,
      message: error instanceof Error ? error.message : String(error),
    });
    return false;
  }
};

export const saveAndCompileShaderDocument = async (document: EditorDocument): Promise<boolean> => {
  if (!(await saveShaderDocument(document))) return false;
  const compiled = await compileShaderDocument(document);
  if (compiled) setState(document.id, { message: 'Shader saved and compilation queued' });
  return compiled;
};

export const reloadShaderDocument = async (document: EditorDocument): Promise<boolean> => {
  const current = ensureState(document);
  if (current.source !== current.confirmed) {
    const discard = window.confirm(`Discard unsaved changes to ${document.title}?`);
    if (!discard) return false;
  }
  return loadShaderDocument(document, true);
};

export const disposeShaderDocument = (documentId: string) => {
  states.delete(documentId);
  listeners.delete(documentId);
};

export const useShaderDocumentState = (document: EditorDocument) => {
  const [, forceUpdate] = useState(0);
  const state = ensureState(document);

  useEffect(() => subscribe(document.id, () => forceUpdate((value) => value + 1)), [document.id]);
  useEffect(() => {
    void loadShaderDocument(document);
  }, [document.assetScope, document.id, document.path]);

  return states.get(document.id) ?? state;
};

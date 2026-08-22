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
  diagnostics: ShaderCompileDiagnostic[];
  reflection: ShaderCompileReflection | null;
};

export type ShaderCompileDiagnostic = {
  severity: 'information' | 'warning' | 'error';
  code?: string;
  message: string;
  path?: string;
  line?: number;
  column?: number;
  graphNode?: string;
};

export type ShaderCompileReflection = {
  compiler: string;
  target: string;
  entryPoint: string;
  bytecodeBytes: number;
  parameters: Array<{ id: number; name: string; offset: number; size: number }>;
  resources: Array<{ name: string; set: number; binding: number; count: number }>;
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
  diagnostics: [],
  reflection: null,
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
  setState(document.id, { compiling: true, diagnostics: [] });
  try {
    const extension = current.path.split('.').pop()?.toLocaleLowerCase();
    const stage = extension === 'vert'
      ? 'vertex'
      : extension === 'comp' || /\[shader\s*\(\s*["']compute["']\s*\)\]/i.test(current.source)
        ? 'compute'
        : 'fragment';
    const response = (await window.arc.host.command('shader.compile', {
      path: current.path,
      source: current.source,
      entryPoint: 'main',
      stage,
      domain: stage === 'compute' ? 'compute' : 'surface',
    })) as HostResponse<
      ShaderCompileReflection & { succeeded: boolean; message: string; diagnostics: ShaderCompileDiagnostic[] }
    >;
    const result = response.payload;
    const succeeded = response.succeeded && result?.succeeded === true;
    setState(document.id, {
      compiling: false,
      diagnostics: result?.diagnostics ?? [],
      reflection: succeeded && result ? result : current.reflection,
      message: result?.message ?? response.error ?? 'Shader compilation failed',
    });
    return succeeded;
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
  if (compiled) setState(document.id, { message: 'Shader saved and transient generation compiled' });
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
  }, [document]);

  return states.get(document.id) ?? state;
};

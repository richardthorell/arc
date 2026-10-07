import { useEffect, useState } from 'react';

import { updateEditorDocumentInStore } from '../editors/editorDocuments';
import type { EditorDocument } from '../editors/editorTypes';
import {
  cloneMaterialGraph,
  createDefaultMaterialFunction,
  isMaterialGraph,
  type MaterialFunctionAssetJson,
  type MaterialGraph,
  type MaterialGraphViewport,
} from './materialGraphTypes';

type HostResponse<T = unknown> = {
  succeeded: boolean;
  error?: string;
  payload?: T;
};

export type MaterialFunctionDocumentState = {
  documentId: string;
  path: string;
  readOnly: boolean;
  asset: MaterialFunctionAssetJson;
  graph: MaterialGraph;
  confirmed: string;
  history: MaterialGraph[];
  historyIndex: number;
  loaded: boolean;
  loading: boolean;
  saving: boolean;
  validating: boolean;
  message: string;
};

const states = new Map<string, MaterialFunctionDocumentState>();
const listeners = new Map<string, Set<() => void>>();

const initialState = (document: EditorDocument): MaterialFunctionDocumentState => {
  const name = document.title.replace(/\.arcmatfn$/i, '') || 'New Material Function';
  const asset = createDefaultMaterialFunction(name);
  return {
    documentId: document.id,
    path: document.path ?? '',
    readOnly: document.readOnly,
    asset,
    graph: cloneMaterialGraph(asset.graph),
    confirmed: '',
    history: [cloneMaterialGraph(asset.graph)],
    historyIndex: 0,
    loaded: false,
    loading: false,
    saving: false,
    validating: false,
    message: '',
  };
};

const emit = (documentId: string) => {
  for (const listener of listeners.get(documentId) ?? []) listener();
};

const setState = (documentId: string, patch: Partial<MaterialFunctionDocumentState>) => {
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

const serialize = (asset: MaterialFunctionAssetJson, graph: MaterialGraph) =>
  `${JSON.stringify({ ...asset, graph: cloneMaterialGraph(graph) }, null, 2)}\n`;

const setDirty = (document: EditorDocument, asset: MaterialFunctionAssetJson, graph: MaterialGraph, confirmed: string) =>
  updateEditorDocumentInStore(document.id, {
    dirty: !document.readOnly && serialize(asset, graph) !== confirmed,
  });

const parseFunction = (value: unknown, document: EditorDocument): MaterialFunctionAssetJson => {
  if (!value || typeof value !== 'object') throw new Error('Material Function document must be an object');
  const candidate = value as Partial<MaterialFunctionAssetJson>;
  if (candidate.kind !== 'materialFunction' || candidate.version !== 1)
    throw new Error('Material Function must use materialFunction schema v1');
  if (typeof candidate.name !== 'string' || !candidate.name.trim()) throw new Error('Material Function requires a name');
  if (!Array.isArray(candidate.inputs) || !Array.isArray(candidate.outputs) || candidate.outputs.length === 0)
    throw new Error('Material Function requires input/output arrays and at least one output');
  if (!isMaterialGraph(candidate.graph)) throw new Error('Material Function contains an invalid graph');
  return {
    kind: 'materialFunction',
    version: 1,
    name: candidate.name,
    description: typeof candidate.description === 'string' ? candidate.description : '',
    inputs: candidate.inputs,
    outputs: candidate.outputs,
    graph: candidate.graph,
  };
};

export const loadMaterialFunctionDocument = async (document: EditorDocument, force = false): Promise<boolean> => {
  const current = ensureState(document);
  if (!document.path) return false;
  if (!force && (current.loaded || current.loading)) return true;
  setState(document.id, { loading: true, message: '' });
  try {
    const file = await window.arc.projects.readText(
      document.path,
      document.assetScope === 'builtin' ? 'builtin' : 'project',
    );
    const asset = parseFunction(JSON.parse(file.text), document);
    const graph = cloneMaterialGraph(asset.graph);
    const confirmed = serialize(asset, graph);
    setState(document.id, {
      asset,
      graph,
      confirmed,
      history: [cloneMaterialGraph(graph)],
      historyIndex: 0,
      loaded: true,
      loading: false,
      message: document.readOnly ? 'Engine Material Function opened read-only' : '',
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

export const replaceMaterialFunctionGraph = (
  document: EditorDocument,
  graph: MaterialGraph,
  options: { recordHistory?: boolean; message?: string } = {},
) => {
  const current = ensureState(document);
  if (current.readOnly || document.readOnly) return;
  const next = cloneMaterialGraph(graph);
  let history = current.history;
  let historyIndex = current.historyIndex;
  if (options.recordHistory !== false) {
    history = current.history.slice(0, current.historyIndex + 1);
    if (JSON.stringify(history.at(-1)) !== JSON.stringify(next)) history.push(cloneMaterialGraph(next));
    historyIndex = history.length - 1;
  }
  setState(document.id, { graph: next, history, historyIndex, message: options.message ?? '' });
  setDirty(document, current.asset, next, current.confirmed);
};

export const replaceMaterialFunctionViewport = (document: EditorDocument, viewport: MaterialGraphViewport) => {
  const current = ensureState(document);
  replaceMaterialFunctionGraph(document, { ...current.graph, viewport }, { recordHistory: false });
};

export const replaceMaterialFunctionAsset = (
  document: EditorDocument,
  updater: (asset: MaterialFunctionAssetJson) => MaterialFunctionAssetJson,
) => {
  const current = ensureState(document);
  if (current.readOnly || document.readOnly) return;
  const asset = updater({ ...current.asset, inputs: [...current.asset.inputs], outputs: [...current.asset.outputs] });
  setState(document.id, { asset, message: '' });
  setDirty(document, asset, current.graph, current.confirmed);
};

export const undoMaterialFunctionGraph = (document: EditorDocument) => {
  const current = ensureState(document);
  if (current.historyIndex <= 0 || current.readOnly) return;
  const historyIndex = current.historyIndex - 1;
  const graph = cloneMaterialGraph(current.history[historyIndex]);
  setState(document.id, { graph, historyIndex, message: 'Undo Material Function graph edit' });
  setDirty(document, current.asset, graph, current.confirmed);
};

export const redoMaterialFunctionGraph = (document: EditorDocument) => {
  const current = ensureState(document);
  if (current.historyIndex + 1 >= current.history.length || current.readOnly) return;
  const historyIndex = current.historyIndex + 1;
  const graph = cloneMaterialGraph(current.history[historyIndex]);
  setState(document.id, { graph, historyIndex, message: 'Redo Material Function graph edit' });
  setDirty(document, current.asset, graph, current.confirmed);
};

export const validateMaterialFunctionDocument = async (document: EditorDocument): Promise<boolean> => {
  const current = ensureState(document);
  const source = serialize(current.asset, current.graph);
  setState(document.id, { validating: true, message: '' });
  try {
    const response = (await window.arc.host.command('shader.compile', {
      path: document.path ?? document.title,
      source,
      entryPoint: 'main',
      stage: 'fragment',
      domain: 'materialFunction',
    })) as HostResponse<{ succeeded?: boolean; message?: string }>;
    const succeeded = response.succeeded && response.payload?.succeeded === true;
    setState(document.id, {
      validating: false,
      message: succeeded
        ? 'Material Function validated successfully'
        : response.payload?.message || response.error || 'Material Function validation failed',
    });
    return succeeded;
  } catch (error) {
    setState(document.id, {
      validating: false,
      message: error instanceof Error ? error.message : String(error),
    });
    return false;
  }
};

export const saveMaterialFunctionDocument = async (document: EditorDocument): Promise<boolean> => {
  const current = ensureState(document);
  if (current.readOnly || document.readOnly || !document.path) return false;
  if (!(await validateMaterialFunctionDocument(document))) return false;
  const latest = ensureState(document);
  const text = serialize(latest.asset, latest.graph);
  setState(document.id, { saving: true });
  try {
    await window.arc.projects.writeText(document.path, text);
    setState(document.id, { saving: false, confirmed: text, message: 'Material Function saved' });
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

export const disposeMaterialFunctionDocument = (documentId: string) => {
  states.delete(documentId);
  listeners.delete(documentId);
};

export const useMaterialFunctionDocumentState = (document: EditorDocument) => {
  const [, rerender] = useState(0);
  const state = ensureState(document);

  useEffect(() => subscribe(document.id, () => rerender((value) => value + 1)), [document.id]);
  useEffect(() => {
    void loadMaterialFunctionDocument(document);
  }, [document]);

  return states.get(document.id) ?? state;
};

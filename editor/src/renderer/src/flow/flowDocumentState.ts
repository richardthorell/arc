import { useEffect, useState } from 'react';

import { updateEditorDocumentInStore } from '../editors/editorDocuments';
import type { EditorDocument } from '../editors/editorTypes';
import {
  cloneFlowGraph,
  createFlowAsset,
  flowGraphFromAsset,
  isFlowAssetJson,
  type FlowAssetJson,
  type FlowGraph,
} from './flowGraphTypes';

export type FlowDocumentState = {
  documentId: string;
  path: string;
  scope: EditorDocument['assetScope'];
  readOnly: boolean;
  asset: FlowAssetJson;
  graph: FlowGraph;
  confirmedGraph: string;
  history: FlowGraph[];
  historyIndex: number;
  loading: boolean;
  saving: boolean;
  loaded: boolean;
  message: string;
};

const states = new Map<string, FlowDocumentState>();
const listeners = new Map<string, Set<() => void>>();

const graphFingerprint = (graph: FlowGraph) => JSON.stringify(graph);

const initialState = (document: EditorDocument): FlowDocumentState => {
  const asset = createFlowAsset(document.title.replace(/\.arcflow$/i, ''));
  return {
    documentId: document.id,
    path: document.path ?? '',
    scope: document.assetScope,
    readOnly: document.readOnly,
    asset,
    graph: cloneFlowGraph(asset.graph),
    confirmedGraph: '',
    history: [],
    historyIndex: -1,
    loading: false,
    saving: false,
    loaded: false,
    message: '',
  };
};

const emit = (documentId: string) => {
  for (const listener of listeners.get(documentId) ?? []) listener();
};

const setState = (documentId: string, patch: Partial<FlowDocumentState>) => {
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
    if (current) emit(document.id);
    return next;
  }
  if (current.scope !== document.assetScope || current.readOnly !== document.readOnly) {
    const next = { ...current, scope: document.assetScope, readOnly: document.readOnly };
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

const updateDirtyState = (document: EditorDocument, graph: FlowGraph, confirmedGraph: string) =>
  updateEditorDocumentInStore(document.id, { dirty: graphFingerprint(graph) !== confirmedGraph });

export const getFlowDocumentState = (document: EditorDocument) => ensureState(document);

export const loadFlowDocument = async (document: EditorDocument, force = false): Promise<boolean> => {
  const current = ensureState(document);
  if (!document.path) {
    setState(document.id, { message: 'Flow asset path is unavailable' });
    return false;
  }
  if (!force && (current.loaded || current.loading)) return true;

  setState(document.id, { loading: true, message: '' });
  try {
    const file = await window.arc.projects.readText(
      document.path,
      document.assetScope === 'builtin' ? 'builtin' : 'project',
    );
    const parsed = JSON.parse(file.text) as unknown;
    if (!isFlowAssetJson(parsed)) throw new Error('Flow asset must use the Flow authoring schema v1');
    const graph = flowGraphFromAsset(parsed);
    const confirmedGraph = graphFingerprint(graph);
    setState(document.id, {
      asset: parsed,
      graph,
      confirmedGraph,
      history: [cloneFlowGraph(graph)],
      historyIndex: 0,
      loading: false,
      loaded: true,
      message: document.readOnly ? 'Flow graph opened read-only' : '',
    });
    updateEditorDocumentInStore(document.id, { dirty: false });
    return true;
  } catch (error) {
    setState(document.id, {
      loading: false,
      loaded: false,
      message: error instanceof Error ? error.message : String(error),
    });
    return false;
  }
};

export const replaceFlowGraph = (
  document: EditorDocument,
  graph: FlowGraph,
  options: { recordHistory?: boolean; message?: string } = {},
) => {
  const current = ensureState(document);
  if (document.readOnly || current.readOnly) return;
  const nextGraph = cloneFlowGraph(graph);
  let history = current.history;
  let historyIndex = current.historyIndex;
  if (options.recordHistory !== false) {
    history = current.history.slice(0, current.historyIndex + 1);
    const latest = history.at(-1);
    if (!latest || graphFingerprint(latest) !== graphFingerprint(nextGraph)) history.push(cloneFlowGraph(nextGraph));
    if (history.length > 80) history = history.slice(history.length - 80);
    historyIndex = history.length - 1;
  }
  setState(document.id, {
    graph: nextGraph,
    history,
    historyIndex,
    message: options.message ?? '',
  });
  updateDirtyState(document, nextGraph, current.confirmedGraph);
};

export const undoFlowGraph = (document: EditorDocument) => {
  const current = ensureState(document);
  if (document.readOnly || current.historyIndex <= 0) return false;
  const historyIndex = current.historyIndex - 1;
  const graph = cloneFlowGraph(current.history[historyIndex]);
  setState(document.id, { graph, historyIndex, message: 'Undo Flow graph edit' });
  updateDirtyState(document, graph, current.confirmedGraph);
  return true;
};

export const redoFlowGraph = (document: EditorDocument) => {
  const current = ensureState(document);
  if (document.readOnly || current.historyIndex + 1 >= current.history.length) return false;
  const historyIndex = current.historyIndex + 1;
  const graph = cloneFlowGraph(current.history[historyIndex]);
  setState(document.id, { graph, historyIndex, message: 'Redo Flow graph edit' });
  updateDirtyState(document, graph, current.confirmedGraph);
  return true;
};

const serializedFlow = (document: EditorDocument, current: FlowDocumentState) => {
  const asset: FlowAssetJson = {
    version: 1,
    assetType: 'flow',
    name: current.asset.name || document.title.replace(/\.arcflow$/i, ''),
    graph: cloneFlowGraph(current.graph),
  };
  return { asset, text: `${JSON.stringify(asset, null, 2)}\n` };
};

export const saveFlowDocument = async (document: EditorDocument): Promise<boolean> => {
  const current = ensureState(document);
  if (document.readOnly || current.readOnly) {
    setState(document.id, { message: 'This Flow graph is read-only' });
    return false;
  }
  if (!document.path) {
    setState(document.id, { message: 'Flow asset path is unavailable' });
    return false;
  }

  setState(document.id, { saving: true, message: '' });
  try {
    const latest = ensureState(document);
    const serialized = serializedFlow(document, latest);
    await window.arc.projects.writeText(document.path, serialized.text);
    const confirmedGraph = graphFingerprint(latest.graph);
    setState(document.id, {
      asset: serialized.asset,
      confirmedGraph,
      saving: false,
      message: 'Flow graph saved',
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

export const reloadFlowDocument = async (document: EditorDocument): Promise<boolean> => {
  const current = ensureState(document);
  if (
    graphFingerprint(current.graph) !== current.confirmedGraph &&
    !window.confirm(`Discard unsaved changes to ${document.title}?`)
  )
    return false;
  setState(document.id, { loaded: false });
  return loadFlowDocument(document, true);
};

export const disposeFlowDocument = (documentId: string) => {
  states.delete(documentId);
  listeners.delete(documentId);
};

export const useFlowDocumentState = (document: EditorDocument) => {
  const [, forceUpdate] = useState(0);
  const state = ensureState(document);

  useEffect(() => subscribe(document.id, () => forceUpdate((value) => value + 1)), [document.id]);
  useEffect(() => {
    void loadFlowDocument(document);
  }, [document]);

  return states.get(document.id) ?? state;
};

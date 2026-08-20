import { useEffect, useState } from 'react';

import type { EditorDocument } from '../editors/editorTypes';
import { updateEditorDocumentInStore } from '../editors/editorDocuments';
import {
  cloneMaterialGraph,
  materialGraphFromAsset,
  type MaterialAssetJson,
  type MaterialGraph,
} from './materialGraphTypes';
import { compileMaterialGraph, type MaterialCompileResult } from './materialCompiler';

export type MaterialDocumentState = {
  documentId: string;
  path: string;
  scope: EditorDocument['assetScope'];
  readOnly: boolean;
  asset: MaterialAssetJson;
  graph: MaterialGraph;
  confirmedGraph: string;
  history: MaterialGraph[];
  historyIndex: number;
  compilation: MaterialCompileResult;
  previewDataUrl: string;
  previewRevision: number;
  loading: boolean;
  saving: boolean;
  compiling: boolean;
  previewLoading: boolean;
  loaded: boolean;
  message: string;
};

type HostResponse<T = unknown> = {
  succeeded: boolean;
  error?: string;
  payload?: T;
};

type ThumbnailPayload = {
  dataUrl?: string;
};

const emptyGraph = materialGraphFromAsset({});
const states = new Map<string, MaterialDocumentState>();
const listeners = new Map<string, Set<() => void>>();

const graphFingerprint = (graph: MaterialGraph) => JSON.stringify(graph);

const initialState = (document: EditorDocument): MaterialDocumentState => ({
  documentId: document.id,
  path: document.path ?? '',
  scope: document.assetScope,
  readOnly: document.readOnly,
  asset: {},
  graph: cloneMaterialGraph(emptyGraph),
  confirmedGraph: '',
  history: [],
  historyIndex: -1,
  compilation: compileMaterialGraph(emptyGraph),
  previewDataUrl: '',
  previewRevision: 0,
  loading: false,
  saving: false,
  compiling: false,
  previewLoading: false,
  loaded: false,
  message: '',
});

const emit = (documentId: string) => {
  for (const listener of listeners.get(documentId) ?? []) listener();
};

const setState = (documentId: string, patch: Partial<MaterialDocumentState>) => {
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

const updateDirtyState = (document: EditorDocument, graph: MaterialGraph, confirmedGraph: string) =>
  updateEditorDocumentInStore(document.id, { dirty: graphFingerprint(graph) !== confirmedGraph });

const normalizedAsset = (asset: MaterialAssetJson, document: EditorDocument): MaterialAssetJson => ({
  ...asset,
  version: Math.max(3, typeof asset.version === 'number' ? asset.version : 3),
  name: asset.name ?? document.title.replace(/\.arcmat$/i, ''),
  shader: asset.shader ?? 'arc/default_phong',
  domain: asset.domain ?? 'surface',
  blendMode: asset.blendMode ?? 'opaque',
  shadingModel: asset.shadingModel ?? 'standard',
  doubleSided: asset.doubleSided ?? false,
});

export const getMaterialDocumentState = (document: EditorDocument) => ensureState(document);

export const refreshMaterialPreview = async (document: EditorDocument): Promise<boolean> => {
  const current = ensureState(document);
  if (!document.path) return false;
  setState(document.id, { previewLoading: true });
  try {
    const response = (await window.arc.host.query('asset.thumbnail', {
      path: document.path,
      maxSize: 256,
    })) as HostResponse<ThumbnailPayload>;
    if (!response.succeeded || !response.payload?.dataUrl) {
      setState(document.id, {
        previewLoading: false,
        message: response.error || current.message || 'Material preview is unavailable',
      });
      return false;
    }
    setState(document.id, {
      previewDataUrl: response.payload.dataUrl,
      previewRevision: current.previewRevision + 1,
      previewLoading: false,
    });
    return true;
  } catch (error) {
    setState(document.id, {
      previewLoading: false,
      message: error instanceof Error ? error.message : String(error),
    });
    return false;
  }
};

export const loadMaterialDocument = async (document: EditorDocument, force = false): Promise<boolean> => {
  const current = ensureState(document);
  if (!document.path) {
    setState(document.id, { message: 'Material asset path is unavailable' });
    return false;
  }
  if (!force && (current.loaded || current.loading)) return true;

  setState(document.id, { loading: true, message: '' });
  try {
    const file = await window.arc.projects.readText(
      document.path,
      document.assetScope === 'builtin' ? 'builtin' : 'project',
    );
    const parsed = normalizedAsset(JSON.parse(file.text) as MaterialAssetJson, document);
    const graph = materialGraphFromAsset(parsed);
    const fingerprint = graphFingerprint(graph);
    setState(document.id, {
      asset: parsed,
      graph,
      confirmedGraph: fingerprint,
      history: [cloneMaterialGraph(graph)],
      historyIndex: 0,
      compilation: compileMaterialGraph(graph),
      loading: false,
      loaded: true,
      message: document.readOnly ? 'Engine material opened read-only' : '',
    });
    updateEditorDocumentInStore(document.id, { dirty: false });
    void refreshMaterialPreview(document);
    return true;
  } catch (error) {
    setState(document.id, {
      loading: false,
      message: error instanceof Error ? error.message : String(error),
    });
    return false;
  }
};

export const replaceMaterialGraph = (
  document: EditorDocument,
  graph: MaterialGraph,
  options: { recordHistory?: boolean; message?: string } = {},
) => {
  const current = ensureState(document);
  if (document.readOnly || current.readOnly) return;
  const nextGraph = cloneMaterialGraph(graph);
  let history = current.history;
  let historyIndex = current.historyIndex;
  if (options.recordHistory !== false) {
    history = current.history.slice(0, current.historyIndex + 1);
    if (graphFingerprint(history.at(-1) ?? emptyGraph) !== graphFingerprint(nextGraph))
      history.push(cloneMaterialGraph(nextGraph));
    if (history.length > 80) history = history.slice(history.length - 80);
    historyIndex = history.length - 1;
  }
  const compilation = compileMaterialGraph(nextGraph);
  setState(document.id, {
    graph: nextGraph,
    history,
    historyIndex,
    compilation,
    message: options.message ?? '',
  });
  updateDirtyState(document, nextGraph, current.confirmedGraph);
};

export const undoMaterialGraph = (document: EditorDocument) => {
  const current = ensureState(document);
  if (document.readOnly || current.historyIndex <= 0) return false;
  const historyIndex = current.historyIndex - 1;
  const graph = cloneMaterialGraph(current.history[historyIndex]);
  setState(document.id, {
    graph,
    historyIndex,
    compilation: compileMaterialGraph(graph),
    message: 'Undo material graph edit',
  });
  updateDirtyState(document, graph, current.confirmedGraph);
  return true;
};

export const redoMaterialGraph = (document: EditorDocument) => {
  const current = ensureState(document);
  if (document.readOnly || current.historyIndex + 1 >= current.history.length) return false;
  const historyIndex = current.historyIndex + 1;
  const graph = cloneMaterialGraph(current.history[historyIndex]);
  setState(document.id, {
    graph,
    historyIndex,
    compilation: compileMaterialGraph(graph),
    message: 'Redo material graph edit',
  });
  updateDirtyState(document, graph, current.confirmedGraph);
  return true;
};

export const compileMaterialDocument = async (document: EditorDocument): Promise<boolean> => {
  const current = ensureState(document);
  setState(document.id, { compiling: true, message: '' });
  const compilation = compileMaterialGraph(current.graph);
  const errors = compilation.diagnostics.filter((diagnostic) => diagnostic.severity === 'error');
  setState(document.id, {
    compilation,
    compiling: false,
    message: errors.length
      ? `${errors.length} material graph error${errors.length === 1 ? '' : 's'}`
      : `Material IR compiled · ${compilation.ir.expressions.length} expressions · ${compilation.ir.parameters.length} parameters`,
  });
  return compilation.succeeded;
};

const serializedMaterial = (
  document: EditorDocument,
  current: MaterialDocumentState,
  compilation: MaterialCompileResult,
) => {
  const surface = { ...(current.asset.surface ?? {}), ...compilation.surface };
  const textures = { ...(current.asset.textures ?? {}), ...compilation.textures };
  const asset: MaterialAssetJson = {
    ...normalizedAsset(current.asset, document),
    surface,
    textures,
    graph: cloneMaterialGraph(current.graph),
  };
  return { asset, text: `${JSON.stringify(asset, null, 2)}\n` };
};

export const saveMaterialDocument = async (document: EditorDocument): Promise<boolean> => {
  const current = ensureState(document);
  if (document.readOnly || current.readOnly) {
    setState(document.id, { message: 'This material is read-only' });
    return false;
  }
  if (!document.path) {
    setState(document.id, { message: 'Material asset path is unavailable' });
    return false;
  }

  const compilation = compileMaterialGraph(current.graph);
  if (!compilation.succeeded) {
    setState(document.id, { compilation, message: 'Resolve material graph errors before saving' });
    return false;
  }
  setState(document.id, { saving: true, compilation, message: '' });
  try {
    const serialized = serializedMaterial(document, current, compilation);
    await window.arc.projects.writeText(document.path, serialized.text);
    const confirmedGraph = graphFingerprint(current.graph);
    setState(document.id, {
      asset: serialized.asset,
      confirmedGraph,
      saving: false,
      message: 'Material saved',
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

export const publishMaterialDocument = async (document: EditorDocument): Promise<boolean> => {
  if (!(await compileMaterialDocument(document))) return false;
  if (document.dirty && !(await saveMaterialDocument(document))) return false;
  if (!document.assetGuid) {
    setState(document.id, { message: 'Material has no registered asset GUID and cannot be reimported' });
    return false;
  }
  setState(document.id, { compiling: true, message: 'Publishing material…' });
  try {
    const response = (await window.arc.host.command('asset.reimport', { guid: document.assetGuid })) as HostResponse;
    if (!response.succeeded) {
      setState(document.id, { compiling: false, message: response.error || 'Material reimport failed' });
      return false;
    }
    setState(document.id, { compiling: false, message: 'Material published to renderer' });
    window.setTimeout(() => void refreshMaterialPreview(document), 120);
    return true;
  } catch (error) {
    setState(document.id, {
      compiling: false,
      message: error instanceof Error ? error.message : String(error),
    });
    return false;
  }
};

export const saveAndPublishMaterialDocument = async (document: EditorDocument): Promise<boolean> => {
  if (!(await saveMaterialDocument(document))) return false;
  return publishMaterialDocument({ ...document, dirty: false });
};

export const reloadMaterialDocument = async (document: EditorDocument): Promise<boolean> => {
  const current = ensureState(document);
  if (
    graphFingerprint(current.graph) !== current.confirmedGraph &&
    !window.confirm(`Discard unsaved changes to ${document.title}?`)
  )
    return false;
  setState(document.id, { loaded: false });
  return loadMaterialDocument(document, true);
};

export const disposeMaterialDocument = (documentId: string) => {
  states.delete(documentId);
  listeners.delete(documentId);
};

export const useMaterialDocumentState = (document: EditorDocument) => {
  const [, forceUpdate] = useState(0);
  const state = ensureState(document);

  useEffect(() => subscribe(document.id, () => forceUpdate((value) => value + 1)), [document.id]);
  useEffect(() => {
    void loadMaterialDocument(document);
  }, [document.id, document.path]);

  return states.get(document.id) ?? state;
};

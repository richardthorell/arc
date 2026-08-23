import { useEffect, useState } from 'react';

import type { EditorDocument } from '../editors/editorTypes';
import { updateEditorDocumentInStore } from '../editors/editorDocuments';
import {
  compilingMaterialResult,
  emptyMaterialCompileResult,
  nativeMaterialCompileResult,
  type MaterialCompileResult,
  type NativeMaterialCompilePayload,
} from './materialCompiler';
import {
  cloneMaterialGraph,
  createDefaultMaterialGraph,
  isMaterialGraph,
  materialGraphFromAsset,
  type MaterialAssetJson,
  type MaterialGraph,
} from './materialGraphTypes';

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

const emptyGraph = createDefaultMaterialGraph();
const states = new Map<string, MaterialDocumentState>();
const listeners = new Map<string, Set<() => void>>();
const compileTimers = new Map<string, ReturnType<typeof setTimeout>>();
const compileGenerations = new Map<string, number>();

const graphFingerprint = (graph: MaterialGraph) => JSON.stringify(graph);
const materialShaderPath = (asset: MaterialAssetJson) =>
  typeof asset.shaderPath === 'string' && asset.shaderPath.trim() ? asset.shaderPath.trim() : '';

const normalizedAsset = (asset: MaterialAssetJson, document: EditorDocument): MaterialAssetJson => {
  if (asset.version !== 4) throw new Error('Material asset must use authoring schema v4');
  const legacyFields = ['shader', 'surface', 'textures', 'advanced'].filter((field) => field in asset);
  if (legacyFields.length > 0)
    throw new Error(`Legacy material fields are no longer supported: ${legacyFields.join(', ')}`);

  const shaderPath = materialShaderPath(asset);
  const hasGraph = isMaterialGraph(asset.graph);
  if (hasGraph === Boolean(shaderPath))
    throw new Error(
      hasGraph
        ? 'Material must use either a graph or shaderPath, not both'
        : 'Material must provide a graph or shaderPath',
    );

  return {
    ...asset,
    version: 4,
    name: asset.name ?? document.title.replace(/\.arcmat$/i, ''),
    domain: asset.domain ?? 'surface',
    blendMode: asset.blendMode ?? 'opaque',
    shadingModel: asset.shadingModel ?? 'standard',
    doubleSided: asset.doubleSided ?? false,
  };
};

const canonicalAssetFields = (asset: MaterialAssetJson, document: EditorDocument): MaterialAssetJson => {
  const canonical = { ...normalizedAsset(asset, document) };
  delete canonical.shaderPath;
  delete canonical.graph;
  return canonical;
};

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
  compilation: emptyMaterialCompileResult(),
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

const cancelScheduledCompile = (documentId: string) => {
  const timer = compileTimers.get(documentId);
  if (timer !== undefined) clearTimeout(timer);
  compileTimers.delete(documentId);
};

const scheduleNativeCompile = (document: EditorDocument) => {
  cancelScheduledCompile(document.id);
  const timer = setTimeout(() => {
    compileTimers.delete(document.id);
    void compileMaterialDocument(document, { quiet: true });
  }, 180);
  compileTimers.set(document.id, timer);
};

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
    const customShader = materialShaderPath(parsed);
    const graph = customShader ? createDefaultMaterialGraph() : materialGraphFromAsset(parsed);
    const fingerprint = graphFingerprint(graph);
    setState(document.id, {
      asset: parsed,
      graph,
      confirmedGraph: fingerprint,
      history: customShader ? [] : [cloneMaterialGraph(graph)],
      historyIndex: customShader ? -1 : 0,
      compilation: emptyMaterialCompileResult(),
      loading: false,
      loaded: true,
      message: document.readOnly ? 'Engine material opened read-only' : '',
    });
    updateEditorDocumentInStore(document.id, { dirty: false });
    if (!customShader) scheduleNativeCompile(document);
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
  if (document.readOnly || current.readOnly || materialShaderPath(current.asset)) return;
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
  setState(document.id, {
    graph: nextGraph,
    history,
    historyIndex,
    compilation: emptyMaterialCompileResult(),
    message: options.message ?? '',
  });
  updateDirtyState(document, nextGraph, current.confirmedGraph);
  scheduleNativeCompile(document);
};

export const undoMaterialGraph = (document: EditorDocument) => {
  const current = ensureState(document);
  if (document.readOnly || materialShaderPath(current.asset) || current.historyIndex <= 0) return false;
  const historyIndex = current.historyIndex - 1;
  const graph = cloneMaterialGraph(current.history[historyIndex]);
  setState(document.id, {
    graph,
    historyIndex,
    compilation: emptyMaterialCompileResult(),
    message: 'Undo material graph edit',
  });
  updateDirtyState(document, graph, current.confirmedGraph);
  scheduleNativeCompile(document);
  return true;
};

export const redoMaterialGraph = (document: EditorDocument) => {
  const current = ensureState(document);
  if (document.readOnly || materialShaderPath(current.asset) || current.historyIndex + 1 >= current.history.length)
    return false;
  const historyIndex = current.historyIndex + 1;
  const graph = cloneMaterialGraph(current.history[historyIndex]);
  setState(document.id, {
    graph,
    historyIndex,
    compilation: emptyMaterialCompileResult(),
    message: 'Redo material graph edit',
  });
  updateDirtyState(document, graph, current.confirmedGraph);
  scheduleNativeCompile(document);
  return true;
};

export const compileMaterialDocument = async (
  document: EditorDocument,
  options: { quiet?: boolean } = {},
): Promise<boolean> => {
  const current = ensureState(document);
  const customShader = materialShaderPath(current.asset);
  if (customShader) {
    if (!options.quiet)
      setState(document.id, {
        message: `Custom Material Shader '${customShader}' is validated during asset cook; reimport does not run the cooker`,
      });
    return false;
  }

  cancelScheduledCompile(document.id);
  const generation = (compileGenerations.get(document.id) ?? 0) + 1;
  compileGenerations.set(document.id, generation);
  setState(document.id, {
    compiling: true,
    compilation: compilingMaterialResult(current.compilation),
    message: options.quiet ? current.message : '',
  });

  try {
    const response = (await window.arc.host.command('shader.compile', {
      path: `${document.path ?? document.title}.generated.slang`,
      source: JSON.stringify(current.graph),
      entryPoint: 'main',
      stage: 'fragment',
      domain: 'materialGraph',
    })) as HostResponse<NativeMaterialCompilePayload>;
    if (compileGenerations.get(document.id) !== generation) return false;

    const compilation = nativeMaterialCompileResult(
      response.succeeded,
      response.payload,
      response.error || response.payload?.message || 'Native material compilation failed',
    );
    const errors = compilation.diagnostics.filter((diagnostic) => diagnostic.severity === 'error');
    setState(document.id, {
      compilation,
      compiling: false,
      message: options.quiet
        ? compilation.succeeded
          ? ''
          : `${errors.length || 1} native material compiler error${errors.length === 1 ? '' : 's'}`
        : compilation.succeeded
          ? 'Native Material IR compiled successfully'
          : response.payload?.message ||
            response.error ||
            compilation.diagnostics[0]?.message ||
            'Native material compilation failed',
    });
    return compilation.succeeded;
  } catch (error) {
    if (compileGenerations.get(document.id) !== generation) return false;
    const message = error instanceof Error ? error.message : String(error);
    setState(document.id, {
      compiling: false,
      compilation: {
        status: 'failed',
        succeeded: false,
        diagnostics: [{ severity: 'error', message }],
      },
      message,
    });
    return false;
  }
};

const serializedMaterial = (document: EditorDocument, current: MaterialDocumentState) => {
  const customShader = materialShaderPath(current.asset);
  const canonical = canonicalAssetFields(current.asset, document);
  const asset: MaterialAssetJson = customShader
    ? {
        ...canonical,
        shaderPath: customShader,
        graph: null,
      }
    : {
        ...canonical,
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

  if (!materialShaderPath(current.asset) && !(await compileMaterialDocument(document, { quiet: true }))) {
    setState(document.id, { message: 'Resolve native material compiler errors before saving' });
    return false;
  }
  setState(document.id, { saving: true, message: '' });
  try {
    const latest = ensureState(document);
    const serialized = serializedMaterial(document, latest);
    await window.arc.projects.writeText(document.path, serialized.text);
    const confirmedGraph = graphFingerprint(latest.graph);
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
  const current = ensureState(document);
  const customShader = materialShaderPath(current.asset);
  if (!customShader && !(await compileMaterialDocument(document))) return false;
  if (document.dirty && !(await saveMaterialDocument(document))) return false;
  if (!document.assetGuid) {
    setState(document.id, { message: 'Material has no registered asset GUID and cannot be reimported' });
    return false;
  }
  setState(document.id, { compiling: true, message: customShader ? 'Reimporting material…' : 'Publishing material…' });
  try {
    const response = (await window.arc.host.command('asset.reimport', { guid: document.assetGuid })) as HostResponse;
    if (!response.succeeded) {
      setState(document.id, {
        compiling: false,
        message: response.error || 'Material reimport failed',
      });
      return false;
    }
    const latest = ensureState(document);
    setState(document.id, {
      compiling: false,
      compilation: latest.compilation,
      message: customShader
        ? 'Material reimported; custom Material Shader validation runs during asset cook'
        : 'Material published to renderer',
    });
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
  cancelScheduledCompile(document.id);
  setState(document.id, { loaded: false });
  return loadMaterialDocument(document, true);
};

export const disposeMaterialDocument = (documentId: string) => {
  cancelScheduledCompile(documentId);
  compileGenerations.delete(documentId);
  states.delete(documentId);
  listeners.delete(documentId);
};

export const useMaterialDocumentState = (document: EditorDocument) => {
  const [, forceUpdate] = useState(0);
  const state = ensureState(document);

  useEffect(() => subscribe(document.id, () => forceUpdate((value) => value + 1)), [document.id]);
  useEffect(() => {
    void loadMaterialDocument(document);
  }, [document]);

  return states.get(document.id) ?? state;
};

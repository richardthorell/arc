import { useEffect, useState } from 'react';

import type { EditorDocument } from '../editors/editorTypes';
import { updateEditorDocumentInStore } from '../editors/editorDocuments';
import type { AssetItem } from '../services/editorHostTypes';
import {
  findMaterialInstanceAsset,
  loadMaterialInstanceParentModel,
  materialAssetReference,
  type MaterialInstanceParentModel,
} from './materialInstanceAuthoring';
import {
  deserializeMaterialInstanceAsset,
  MATERIAL_INSTANCE_ASSET_VERSION,
  serializeMaterialInstanceAsset,
  type MaterialInstanceAsset,
  type MaterialInstanceAssetReference,
} from './materialInstancePersistence';

type HostResponse<T = unknown> = {
  succeeded: boolean;
  error?: string;
  payload?: T;
};

type HostAsset = {
  guid?: string;
  typeId?: string;
  importerId?: string;
  title?: string;
  path?: string;
  sourcePath?: string;
  scope?: AssetItem['scope'];
  readOnly?: boolean;
  state?: AssetItem['status'];
  kind?: AssetItem['kind'];
  generation?: number;
};

type HostAssetsPayload = {
  assets?: HostAsset[];
};

export type MaterialInstanceDocumentState = {
  documentId: string;
  path: string;
  readOnly: boolean;
  asset: MaterialInstanceAsset;
  confirmed: string;
  assets: AssetItem[];
  parentModel: MaterialInstanceParentModel | null;
  history: MaterialInstanceAsset[];
  historyIndex: number;
  loaded: boolean;
  loading: boolean;
  saving: boolean;
  previewLoading: boolean;
  previewDataUrl: string;
  message: string;
};

const emptyReference = (): MaterialInstanceAssetReference => ({ guid: '', pathHint: '' });
const emptyAsset = (document: EditorDocument): MaterialInstanceAsset => ({
  version: MATERIAL_INSTANCE_ASSET_VERSION,
  name: document.title.replace(/\.arcmatinst$/i, '') || 'Material Instance',
  parent: emptyReference(),
  parameterOverrides: [],
  functionOverrides: [],
});

const states = new Map<string, MaterialInstanceDocumentState>();
const listeners = new Map<string, Set<() => void>>();

const cloneAsset = (asset: MaterialInstanceAsset): MaterialInstanceAsset =>
  structuredClone(asset) as MaterialInstanceAsset;

const initialState = (document: EditorDocument): MaterialInstanceDocumentState => ({
  documentId: document.id,
  path: document.path ?? '',
  readOnly: document.readOnly,
  asset: emptyAsset(document),
  confirmed: '',
  assets: [],
  parentModel: null,
  history: [],
  historyIndex: -1,
  loaded: false,
  loading: false,
  saving: false,
  previewLoading: false,
  previewDataUrl: '',
  message: '',
});

const emit = (documentId: string) => {
  for (const listener of listeners.get(documentId) ?? []) listener();
};

const setState = (documentId: string, patch: Partial<MaterialInstanceDocumentState>) => {
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

const hostAssets = async (): Promise<AssetItem[]> => {
  const response = (await window.arc.host.query('project.assets')) as HostResponse<HostAssetsPayload>;
  if (!response.succeeded) return [];
  return (response.payload?.assets ?? []).flatMap((asset, index) => {
    const path = asset.path ?? '';
    const kind =
      asset.kind ??
      (path.endsWith('.arcmatinst')
        ? 'materialInstance'
        : path.endsWith('.arcmat')
          ? 'material'
          : path.endsWith('.arcmatfn')
            ? 'materialFunction'
            : 'unknown');
    if (!path && !asset.sourcePath) return [];
    return [
      {
        id: asset.guid ?? asset.sourcePath ?? path ?? `asset-${index}`,
        guid: asset.guid,
        typeId: asset.typeId,
        importerId: asset.importerId,
        name: asset.title?.trim() || (asset.sourcePath || path).split('/').at(-1) || 'Asset',
        title: asset.title,
        path,
        sourcePath: asset.sourcePath,
        scope: asset.scope,
        readOnly: asset.readOnly,
        kind,
        status: asset.state ?? 'ready',
        generation: asset.generation,
      } satisfies AssetItem,
    ];
  });
};

const refreshParentModel = async (
  document: EditorDocument,
  asset: MaterialInstanceAsset,
  assets: AssetItem[],
): Promise<MaterialInstanceParentModel | null> => {
  if (!asset.parent.guid || !asset.parent.pathHint) return null;
  try {
    return await loadMaterialInstanceParentModel(asset.parent, assets);
  } catch (error) {
    setState(document.id, { message: error instanceof Error ? error.message : String(error) });
    return null;
  }
};

const setDirty = (document: EditorDocument, asset: MaterialInstanceAsset, confirmed: string) =>
  updateEditorDocumentInStore(document.id, {
    dirty: !document.readOnly && serializeMaterialInstanceAsset(asset) !== confirmed,
  });

export const loadMaterialInstanceDocument = async (document: EditorDocument, force = false): Promise<boolean> => {
  const current = ensureState(document);
  if (!document.path) return false;
  if (!force && (current.loaded || current.loading)) return true;
  setState(document.id, { loading: true, message: '' });
  try {
    const [file, assets] = await Promise.all([
      window.arc.projects.readText(document.path, document.assetScope === 'builtin' ? 'builtin' : 'project'),
      hostAssets(),
    ]);
    const asset = deserializeMaterialInstanceAsset(file.text);
    if (!asset) throw new Error('Material Instance uses an invalid or unsupported schema.');
    const confirmed = serializeMaterialInstanceAsset(asset);
    const parentModel = await refreshParentModel(document, asset, assets);
    setState(document.id, {
      asset,
      confirmed,
      assets,
      parentModel,
      history: [cloneAsset(asset)],
      historyIndex: 0,
      loaded: true,
      loading: false,
      message: document.readOnly ? 'Engine Material Instance opened read-only' : '',
    });
    updateEditorDocumentInStore(document.id, { dirty: false });
    void refreshMaterialInstancePreview(document);
    return true;
  } catch (error) {
    setState(document.id, {
      loading: false,
      message: error instanceof Error ? error.message : String(error),
    });
    return false;
  }
};

export const replaceMaterialInstanceAsset = async (
  document: EditorDocument,
  updater: (asset: MaterialInstanceAsset) => MaterialInstanceAsset,
  message = '',
) => {
  const current = ensureState(document);
  if (document.readOnly || current.readOnly) return;
  const next = updater(cloneAsset(current.asset));
  let history = current.history.slice(0, current.historyIndex + 1);
  if (serializeMaterialInstanceAsset(history.at(-1) ?? current.asset) !== serializeMaterialInstanceAsset(next))
    history.push(cloneAsset(next));
  if (history.length > 80) history = history.slice(history.length - 80);
  const parentChanged =
    next.parent.guid !== current.asset.parent.guid || next.parent.pathHint !== current.asset.parent.pathHint;
  const parentModel = parentChanged ? await refreshParentModel(document, next, current.assets) : current.parentModel;
  setState(document.id, {
    asset: next,
    parentModel,
    history,
    historyIndex: history.length - 1,
    message,
  });
  setDirty(document, next, current.confirmed);
};

const restoreHistory = async (document: EditorDocument, historyIndex: number) => {
  const current = ensureState(document);
  const asset = cloneAsset(current.history[historyIndex]);
  const parentChanged =
    asset.parent.guid !== current.asset.parent.guid || asset.parent.pathHint !== current.asset.parent.pathHint;
  const parentModel = parentChanged ? await refreshParentModel(document, asset, current.assets) : current.parentModel;
  setState(document.id, {
    asset,
    parentModel,
    historyIndex,
    message: historyIndex < current.historyIndex ? 'Undo Material Instance edit' : 'Redo Material Instance edit',
  });
  setDirty(document, asset, current.confirmed);
};

export const undoMaterialInstanceDocument = async (document: EditorDocument) => {
  const current = ensureState(document);
  if (document.readOnly || current.historyIndex <= 0) return false;
  await restoreHistory(document, current.historyIndex - 1);
  return true;
};

export const redoMaterialInstanceDocument = async (document: EditorDocument) => {
  const current = ensureState(document);
  if (document.readOnly || current.historyIndex + 1 >= current.history.length) return false;
  await restoreHistory(document, current.historyIndex + 1);
  return true;
};

export const setMaterialInstanceParent = async (document: EditorDocument, parentAsset: AssetItem) => {
  const reference = materialAssetReference(parentAsset);
  if (!reference) return;
  await replaceMaterialInstanceAsset(
    document,
    (asset) => ({
      ...asset,
      parent: reference,
      parameterOverrides: [],
      functionOverrides: [],
    }),
    'Parent changed; instance overrides were reset',
  );
};

export const refreshMaterialInstancePreview = async (document: EditorDocument): Promise<boolean> => {
  const current = ensureState(document);
  const path = document.path;
  if (!path) return false;
  setState(document.id, { previewLoading: true });
  try {
    let response = (await window.arc.host.query('asset.thumbnail', { path, maxSize: 256 })) as HostResponse<{
      dataUrl?: string;
    }>;
    if ((!response.succeeded || !response.payload?.dataUrl) && current.parentModel) {
      response = (await window.arc.host.query('asset.thumbnail', {
        path: current.parentModel.asset.sourcePath || current.parentModel.asset.path,
        maxSize: 256,
      })) as HostResponse<{ dataUrl?: string }>;
    }
    setState(document.id, {
      previewLoading: false,
      previewDataUrl: response.succeeded ? (response.payload?.dataUrl ?? '') : '',
    });
    return response.succeeded && Boolean(response.payload?.dataUrl);
  } catch {
    setState(document.id, { previewLoading: false });
    return false;
  }
};

export const saveMaterialInstanceDocument = async (document: EditorDocument): Promise<boolean> => {
  const current = ensureState(document);
  if (document.readOnly || current.readOnly || !document.path) return false;
  setState(document.id, { saving: true, message: '' });
  try {
    const text = serializeMaterialInstanceAsset(current.asset);
    await window.arc.projects.writeText(document.path, text);
    setState(document.id, { confirmed: text, saving: false, message: 'Material Instance saved' });
    updateEditorDocumentInStore(document.id, { dirty: false });
    await refreshMaterialInstancePreview(document);
    return true;
  } catch (error) {
    setState(document.id, {
      saving: false,
      message: error instanceof Error ? error.message : String(error),
    });
    return false;
  }
};

export const openMaterialInstanceParent = (document: EditorDocument) => {
  const current = ensureState(document);
  const parent =
    current.parentModel?.asset ?? findMaterialInstanceAsset(current.assets, current.asset.parent, 'material');
  return parent ?? null;
};

export const disposeMaterialInstanceDocument = (documentId: string) => {
  states.delete(documentId);
  listeners.delete(documentId);
};

export const useMaterialInstanceDocumentState = (document: EditorDocument) => {
  const [, rerender] = useState(0);
  const state = ensureState(document);
  useEffect(() => subscribe(document.id, () => rerender((value) => value + 1)), [document.id]);
  useEffect(() => {
    void loadMaterialInstanceDocument(document);
  }, [document]);
  return states.get(document.id) ?? state;
};

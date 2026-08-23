import { createElement } from 'react';
import { Circle, FileCode2, Globe2 } from 'lucide-react';

import type { AssetItem } from '../services/editorHostTypes';
import { MaterialEditor } from '../material/MaterialEditor';
import { MaterialEditorToolbar } from '../material/MaterialEditorToolbar';
import { disposeMaterialDocument, saveMaterialDocument } from '../material/materialDocumentState';
import { ShaderSourceEditor } from '../shader/ShaderSourceEditor';
import { ShaderSourceEditorToolbar } from '../shader/ShaderSourceEditorToolbar';
import { disposeShaderDocument, saveShaderDocument } from '../shader/shaderDocumentState';
import { getActiveEditorDocument, openEditorDocumentInStore } from './editorDocuments';
import type {
  EditorDocument,
  EditorDocumentKind,
  EditorRegistration,
  EditorRegistry,
  EditorRegistrySeed,
} from './editorTypes';

type HostAssetIdentity = {
  guid?: string;
  path?: string;
  scope?: AssetItem['scope'];
  readOnly?: boolean;
  state?: AssetItem['status'];
  typeId?: string;
  importerId?: string;
};

type HostProjectAssetsPayload = {
  assets?: HostAssetIdentity[];
};

type HostResponse<T = unknown> = {
  succeeded: boolean;
  payload?: T;
};

const registrationPollIntervalMs = 50;
const registrationTimeoutMs = 2500;
const normalizedAssetPath = (value: string) => value.replaceAll('\\', '/').replace(/^\.\//, '').toLowerCase();
const sleep = (milliseconds: number) => new Promise<void>((resolve) => window.setTimeout(resolve, milliseconds));

const registeredAssetFromHost = (asset: AssetItem, candidate: HostAssetIdentity): AssetItem => ({
  ...asset,
  id: candidate.guid || asset.id,
  guid: candidate.guid || asset.guid,
  path: candidate.path || asset.path,
  scope: candidate.scope ?? asset.scope,
  readOnly: candidate.readOnly ?? asset.readOnly,
  status: candidate.state ?? asset.status,
  typeId: candidate.typeId ?? asset.typeId,
  importerId: candidate.importerId ?? asset.importerId,
});

export const resolveRegisteredEditorAsset = async (
  asset: AssetItem,
  timeoutMs = registrationTimeoutMs,
): Promise<AssetItem | null> => {
  if (asset.guid) return asset;
  if (!asset.path || typeof window === 'undefined' || !window.arc?.host?.query) return null;

  const expectedPath = normalizedAssetPath(asset.path);
  const deadline = Date.now() + timeoutMs;
  do {
    try {
      const response = (await window.arc.host.query('project.assets')) as
        HostResponse<HostProjectAssetsPayload> | undefined;
      const registered = response?.succeeded
        ? response.payload?.assets?.find(
            (candidate) => Boolean(candidate.guid) && normalizedAssetPath(candidate.path ?? '') === expectedPath,
          )
        : undefined;
      if (registered) return registeredAssetFromHost(asset, registered);
    } catch {
      return null;
    }

    if (Date.now() >= deadline) break;
    await sleep(registrationPollIntervalMs);
  } while (true);

  return null;
};

const shaderRegistration: EditorRegistration = {
  kind: 'shader',
  title: 'Shader Source Editor',
  icon: FileCode2,
  allowMultiple: true,
  closeable: true,
  canOpenAsset: (asset) => asset.kind === 'shader',
  createDocument: (asset) => ({
    id: `shader:${asset.guid ?? asset.path}`,
    kind: 'shader',
    title: asset.name,
    path: asset.path,
    assetId: asset.id,
    assetGuid: asset.guid,
    assetScope: asset.scope,
    dirty: false,
    readOnly: asset.scope === 'builtin' || Boolean(asset.readOnly),
  }),
  render: (document) => createElement(ShaderSourceEditor, { document }),
  renderToolbar: (document) => createElement(ShaderSourceEditorToolbar, { document }),
  save: saveShaderDocument,
  onClosed: (document) => disposeShaderDocument(document.id),
};

const materialRegistration: EditorRegistration = {
  kind: 'material',
  title: 'Material Editor',
  icon: Circle,
  allowMultiple: true,
  closeable: true,
  canOpenAsset: (asset) => asset.kind === 'material',
  createDocument: (asset) => ({
    id: `material:${asset.guid ?? asset.path}`,
    kind: 'material',
    title: asset.name,
    path: asset.path,
    assetId: asset.id,
    assetGuid: asset.guid,
    assetScope: asset.scope,
    dirty: false,
    readOnly: asset.scope === 'builtin' || Boolean(asset.readOnly),
  }),
  render: (document) => createElement(MaterialEditor, { document }),
  renderToolbar: (document) => createElement(MaterialEditorToolbar, { document }),
  save: saveMaterialDocument,
  onClosed: (document) => disposeMaterialDocument(document.id),
};

let currentRegistry: EditorRegistry | null = null;

export const createEditorRegistry = (registrations: EditorRegistrySeed): EditorRegistry => {
  const registry = {
    ...registrations,
    level: { ...registrations.level, icon: Globe2 },
    shader: registrations.shader ?? shaderRegistration,
    material: registrations.material ?? materialRegistration,
  } as EditorRegistry;
  currentRegistry = registry;
  return registry;
};

export const getEditorRegistration = (registry: EditorRegistry, kind: EditorDocumentKind) => registry[kind];

export const createEditorDocumentForAsset = (
  asset: AssetItem,
  registry: EditorRegistry | null = currentRegistry,
): { document: EditorDocument; registration: EditorRegistration } | null => {
  if (!registry) return null;
  for (const registration of Object.values(registry)) {
    if (!registration.canOpenAsset?.(asset) || !registration.createDocument) continue;
    return { document: registration.createDocument(asset), registration };
  }
  return null;
};

const openResolvedAssetEditorDocument = (asset: AssetItem, registry: EditorRegistry) => {
  const target = createEditorDocumentForAsset(asset, registry);
  if (!target) return false;
  openEditorDocumentInStore(target.document, target.registration.allowMultiple);
  return true;
};

export const openAssetEditorDocument = (asset: AssetItem, registry: EditorRegistry | null = currentRegistry) => {
  const target = createEditorDocumentForAsset(asset, registry);
  if (!target || !registry) return false;

  const needsRegistration = !asset.guid && (asset.kind === 'material' || asset.kind === 'shader');
  if (!needsRegistration || typeof window === 'undefined' || !window.arc?.host?.query) {
    openEditorDocumentInStore(target.document, target.registration.allowMultiple);
    return true;
  }

  // Newly-authored assets are written to disk before the native asset monitor has
  // necessarily assigned their stable GUID. Keep the current (usually level)
  // editor active long enough for its viewport to drive the source monitor, then
  // open the canonical registered asset so native previews/reimports use a GUID.
  void resolveRegisteredEditorAsset(asset).then((registered) => {
    openResolvedAssetEditorDocument(registered ?? asset, registry);
  });
  return true;
};

export const saveActiveEditorDocument = async (registry: EditorRegistry | null = currentRegistry) => {
  const document = getActiveEditorDocument();
  if (!document || !registry) return false;
  const registration = registry[document.kind];
  if (!registration.save) return false;
  return registration.save(document);
};

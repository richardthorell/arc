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
  projectRoot?: string;
  assetRoot?: string;
  assets?: HostAssetIdentity[];
};

type HostResponse<T = unknown> = {
  succeeded: boolean;
  payload?: T;
};

const registrationPollIntervalMs = 50;
const registrationTimeoutMs = 2500;
const cleanAssetPath = (value: string) =>
  value
    .replaceAll('\\', '/')
    .replace(/\/+/g, '/')
    .replace(/^\.\//, '')
    .replace(/^\/|\/$/g, '');
const normalizedAssetPath = (value: string) => cleanAssetPath(value).toLowerCase();
const sleep = (milliseconds: number) => new Promise<void>((resolve) => window.setTimeout(resolve, milliseconds));

const projectRelativeHostAssetPath = (payload: HostProjectAssetsPayload, candidate: HostAssetIdentity) => {
  const candidatePath = cleanAssetPath(candidate.path ?? '');
  if (!candidatePath || candidate.scope === 'builtin') return candidatePath;

  const projectRoot = cleanAssetPath(payload.projectRoot ?? '');
  const assetRoot = cleanAssetPath(payload.assetRoot ?? '');
  if (!projectRoot || !assetRoot) return candidatePath;

  const normalizedProjectRoot = projectRoot.toLowerCase();
  const normalizedAssetRoot = assetRoot.toLowerCase();
  if (normalizedAssetRoot === normalizedProjectRoot) return candidatePath;
  if (!normalizedAssetRoot.startsWith(`${normalizedProjectRoot}/`)) return candidatePath;

  const assetRootRelative = assetRoot.slice(projectRoot.length + 1);
  return cleanAssetPath(`${assetRootRelative}/${candidatePath}`);
};

const registeredAssetFromHost = (
  asset: AssetItem,
  candidate: HostAssetIdentity,
  payload: HostProjectAssetsPayload,
): AssetItem => {
  const registryPath = cleanAssetPath(candidate.path ?? '');
  const projectPath = projectRelativeHostAssetPath(payload, candidate);
  const authoredPath = cleanAssetPath(asset.path);
  const path =
    authoredPath && normalizedAssetPath(authoredPath) !== normalizedAssetPath(registryPath)
      ? authoredPath
      : projectPath || authoredPath || registryPath;

  return {
    ...asset,
    id: candidate.guid || asset.id,
    guid: candidate.guid || asset.guid,
    path,
    scope: candidate.scope ?? asset.scope,
    readOnly: candidate.readOnly ?? asset.readOnly,
    status: candidate.state ?? asset.status,
    typeId: candidate.typeId ?? asset.typeId,
    importerId: candidate.importerId ?? asset.importerId,
  };
};

export const resolveRegisteredEditorAsset = async (
  asset: AssetItem,
  timeoutMs = registrationTimeoutMs,
): Promise<AssetItem | null> => {
  if (!asset.path || typeof window === 'undefined' || !window.arc?.host?.query) return null;

  const expectedPath = normalizedAssetPath(asset.path);
  const deadline = Date.now() + timeoutMs;
  do {
    try {
      const response = (await window.arc.host.query('project.assets')) as
        HostResponse<HostProjectAssetsPayload> | undefined;
      const payload = response?.succeeded ? response.payload : undefined;
      const registered = payload?.assets?.find((candidate) => {
        if (!candidate.guid) return false;
        if (asset.guid && candidate.guid === asset.guid) return true;
        const registryPath = normalizedAssetPath(candidate.path ?? '');
        const projectPath = normalizedAssetPath(projectRelativeHostAssetPath(payload, candidate));
        return registryPath === expectedPath || projectPath === expectedPath;
      });
      if (registered && payload) {
        const resolved = registeredAssetFromHost(asset, registered, payload);
        if (asset.kind === 'material' || asset.kind === 'shader') {
          console.info('[material-flow] asset registration resolved', {
            kind: asset.kind,
            authoredPath: asset.path,
            registryPath: registered.path ?? '',
            projectPath: projectRelativeHostAssetPath(payload, registered),
            resolvedPath: resolved.path,
            guid: registered.guid ?? '',
            state: registered.state ?? '',
          });
        }
        return resolved;
      }
    } catch (error) {
      console.warn('[material-flow] asset registration query failed', error);
      return null;
    }

    if (asset.guid || Date.now() >= deadline) break;
    await sleep(registrationPollIntervalMs);
  } while (true);

  console.warn('[material-flow] asset registration unresolved', {
    kind: asset.kind,
    guid: asset.guid ?? '',
    path: asset.path,
  });
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

  const needsCanonicalProjectIdentity =
    asset.scope !== 'builtin' && (asset.kind === 'material' || asset.kind === 'shader');
  if (!needsCanonicalProjectIdentity || typeof window === 'undefined' || !window.arc?.host?.query) {
    openEditorDocumentInStore(target.document, target.registration.allowMultiple);
    return true;
  }

  // Project asset snapshots use paths relative to the native asset root, while
  // project file I/O uses paths relative to the project root. Resolve both newly
  // authored and already-registered materials/shaders before opening the editor.
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

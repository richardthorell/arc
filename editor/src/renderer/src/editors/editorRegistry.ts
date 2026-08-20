import { createElement } from 'react';
import { FileCode2, Globe2 } from 'lucide-react';

import type { AssetItem } from '../services/editorHostTypes';
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

let currentRegistry: EditorRegistry | null = null;

export const createEditorRegistry = (registrations: EditorRegistrySeed): EditorRegistry => {
  const registry = {
    ...registrations,
    level: { ...registrations.level, icon: Globe2 },
    shader: registrations.shader ?? shaderRegistration,
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

export const openAssetEditorDocument = (asset: AssetItem, registry: EditorRegistry | null = currentRegistry) => {
  const target = createEditorDocumentForAsset(asset, registry);
  if (!target) return false;
  openEditorDocumentInStore(target.document, target.registration.allowMultiple);
  return true;
};

export const saveActiveEditorDocument = async (registry: EditorRegistry | null = currentRegistry) => {
  const document = getActiveEditorDocument();
  if (!document || !registry) return false;
  const registration = registry[document.kind];
  if (!registration.save) return false;
  return registration.save(document);
};

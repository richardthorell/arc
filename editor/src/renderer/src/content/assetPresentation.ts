import type { DocumentTypeIconKind } from '../assets/DocumentTypeIcon';
import type { AssetItem } from '../services/editorHostTypes';

export type AssetPresentationKind = AssetItem['kind'] | 'model';

type AssetPresentationSource = Pick<AssetItem, 'kind' | 'path'> & Partial<Pick<AssetItem, 'scope' | 'readOnly'>>;

type AssetStatusSource = Pick<AssetItem, 'kind' | 'scope' | 'readOnly' | 'residency' | 'hasLastGood'> & {
  state: 'unknown' | 'queued' | 'importing' | 'ready' | 'stale' | 'failed';
};

const modelExtensions = new Set(['fbx', 'glb', 'gltf', 'obj']);

export const assetExtension = (asset: Pick<AssetItem, 'path'>) =>
  asset.path.replaceAll('\\', '/').split('/').at(-1)?.split('.').at(-1)?.toLocaleLowerCase() ?? '';

export const isModelAsset = (asset: Pick<AssetItem, 'kind' | 'path'>) =>
  asset.kind === 'scene' && modelExtensions.has(assetExtension(asset));

export const assetPresentationKind = (asset: Pick<AssetItem, 'kind' | 'path'>): AssetPresentationKind => {
  if (assetExtension(asset) === 'arcflow') return 'flow';
  if (isModelAsset(asset) || asset.kind === 'mesh') return 'model';
  return asset.kind;
};

/**
 * Engine scope is a curated library, not a raw dump of renderer implementation
 * files. Built-in shader sources stay registered for compilation/runtime use but
 * are intentionally hidden from user-facing Content Browser surfaces.
 */
export const isContentBrowserAssetVisible = (
  asset: Pick<AssetItem, 'kind'> & Partial<Pick<AssetItem, 'scope'>>,
) => !(asset.scope === 'builtin' && asset.kind === 'shader');

export const assetPresentationLabel = (asset: AssetPresentationSource) => {
  const kind = assetPresentationKind(asset);
  if (kind === 'model') return 'Model';
  if (kind === 'flow') return 'Flow Graph';
  if (kind === 'materialFunction') return 'Material Function';
  if (kind === 'water') return 'Water Preset';
  if (kind === 'shader' && asset.scope === 'builtin' && asset.readOnly) return 'Engine Shader Source';
  return kind.charAt(0).toLocaleUpperCase() + kind.slice(1);
};

/**
 * Built-in materials and shaders are shipped as immutable source files. They
 * are usable directly by the editor/build pipeline and cannot be reimported by
 * the user, so presenting their registry state as stale is misleading.
 */
export const assetPresentationStatus = (asset: AssetStatusSource): AssetItem['status'] => {
  if (asset.state === 'unknown') return 'missing';
  if (
    asset.state === 'stale' &&
    asset.scope === 'builtin' &&
    asset.readOnly &&
    asset.residency === 'source' &&
    !asset.hasLastGood &&
    (asset.kind === 'material' || asset.kind === 'materialFunction' || asset.kind === 'shader')
  )
    return 'source';
  return asset.state;
};

export const assetPresentationIcon = (asset: Pick<AssetItem, 'kind' | 'path'>): DocumentTypeIconKind => {
  const kind = assetPresentationKind(asset);
  if (kind === 'model') return 'mesh';
  if (kind === 'flow') return 'script';
  if (kind === 'environment') return 'image';
  if (kind === 'materialFunction') return 'material';
  if (kind === 'water' || kind === 'unknown') return 'settings';
  return kind;
};

export const assetDragType = (asset: Pick<AssetItem, 'kind' | 'path'>) =>
  assetPresentationKind(asset) === 'model' ? 'mesh' : asset.kind;

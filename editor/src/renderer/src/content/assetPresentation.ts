import type { DocumentTypeIconKind } from '../assets/DocumentTypeIcon';
import type { AssetItem } from '../services/editorHostTypes';

export type AssetPresentationKind = AssetItem['kind'] | 'model';

const modelExtensions = new Set(['fbx', 'glb', 'gltf', 'obj']);

export const assetExtension = (asset: Pick<AssetItem, 'path'>) =>
  asset.path.replaceAll('\\', '/').split('/').at(-1)?.split('.').at(-1)?.toLocaleLowerCase() ?? '';

export const isModelAsset = (asset: Pick<AssetItem, 'kind' | 'path'>) =>
  asset.kind === 'scene' && modelExtensions.has(assetExtension(asset));

export const assetPresentationKind = (asset: Pick<AssetItem, 'kind' | 'path'>): AssetPresentationKind =>
  isModelAsset(asset) ? 'model' : asset.kind;

export const assetPresentationLabel = (asset: Pick<AssetItem, 'kind' | 'path'>) => {
  const kind = assetPresentationKind(asset);
  if (kind === 'model') return 'Model';
  return kind.charAt(0).toLocaleUpperCase() + kind.slice(1);
};

export const assetPresentationIcon = (asset: Pick<AssetItem, 'kind' | 'path'>): DocumentTypeIconKind => {
  const kind = assetPresentationKind(asset);
  return kind === 'model' ? 'mesh' : kind;
};

export const assetDragType = (asset: Pick<AssetItem, 'kind' | 'path'>) =>
  assetPresentationKind(asset) === 'model' ? 'mesh' : asset.kind;

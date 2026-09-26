import type { AssetItem } from '../services/editorHostTypes';
import { assetPresentationKind } from './assetPresentation';

export type AssetSearchMetadata = Pick<AssetItem, 'id' | 'name' | 'title' | 'description' | 'path' | 'kind'> & {
  tags?: readonly string[];
};

export type AssetSearchFacet = {
  kind?: ReturnType<typeof assetPresentationKind>;
  tags?: readonly string[];
};

const normalize = (value: string) => value.trim().toLocaleLowerCase();

const normalizedTags = (asset: AssetSearchMetadata) =>
  [...new Set((asset.tags ?? []).map(normalize).filter(Boolean))].sort((a, b) => a.localeCompare(b));

export const assetSearchText = (asset: AssetSearchMetadata) =>
  [asset.name, asset.title, asset.description, asset.path, ...normalizedTags(asset)]
    .filter((value): value is string => Boolean(value))
    .map(normalize)
    .join('\n');

export const matchesAssetSearch = (asset: AssetSearchMetadata, query: string) => {
  const terms = normalize(query).split(/\s+/).filter(Boolean);
  if (terms.length === 0) return true;
  const searchable = assetSearchText(asset);
  return terms.every((term) => searchable.includes(term));
};

export const matchesAssetFacet = (asset: AssetSearchMetadata, facet: AssetSearchFacet) => {
  if (facet.kind && assetPresentationKind(asset) !== facet.kind) return false;
  const tags = new Set(normalizedTags(asset));
  return (facet.tags ?? [])
    .map(normalize)
    .filter(Boolean)
    .every((tag) => tags.has(tag));
};

export const filterAssets = (assets: readonly AssetSearchMetadata[], query = '', facet: AssetSearchFacet = {}) =>
  assets.filter((asset) => matchesAssetSearch(asset, query) && matchesAssetFacet(asset, facet));

export const collectAssetFacets = (assets: readonly AssetSearchMetadata[]) => {
  const kinds = new Set<ReturnType<typeof assetPresentationKind>>();
  const tags = new Set<string>();
  for (const asset of assets) {
    kinds.add(assetPresentationKind(asset));
    normalizedTags(asset).forEach((tag) => tags.add(tag));
  }
  return {
    kinds: [...kinds].sort((a, b) => a.localeCompare(b)),
    tags: [...tags].sort((a, b) => a.localeCompare(b)),
  };
};

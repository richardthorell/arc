import type { AssetItem } from '../services/editorHostTypes';
import { assetPresentationKind } from './assetPresentation';

export type AssetSearchMetadata = Pick<AssetItem, 'id' | 'name' | 'title' | 'description' | 'path' | 'kind'> & {
  tags?: readonly string[];
};

export type AssetSearchFacet = {
  kind?: ReturnType<typeof assetPresentationKind>;
  tags?: readonly string[];
};

export type AssetFacetCount<T extends string = string> = { value: T; count: number };

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

/** Builds stable facet counts from the current result population without coupling them to grid/list layout. */
export const collectAssetFacetCounts = (assets: readonly AssetSearchMetadata[]) => {
  const kindCounts = new Map<ReturnType<typeof assetPresentationKind>, number>();
  const tagCounts = new Map<string, number>();

  for (const asset of assets) {
    const kind = assetPresentationKind(asset);
    kindCounts.set(kind, (kindCounts.get(kind) ?? 0) + 1);
    for (const tag of normalizedTags(asset)) tagCounts.set(tag, (tagCounts.get(tag) ?? 0) + 1);
  }

  const sortedCounts = <T extends string>(counts: Map<T, number>): AssetFacetCount<T>[] =>
    [...counts.entries()]
      .map(([value, count]) => ({ value, count }))
      .sort((left, right) => left.value.localeCompare(right.value));

  return { kinds: sortedCounts(kindCounts), tags: sortedCounts(tagCounts) };
};

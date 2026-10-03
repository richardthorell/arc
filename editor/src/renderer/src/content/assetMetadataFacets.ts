import type { AssetSearchMetadata } from './assetMetadataSearch';

export type AssetMetadataFacet = {
  value: string;
  count: number;
};

export type AssetMetadataFacets = {
  assetTypes: AssetMetadataFacet[];
  tags: AssetMetadataFacet[];
};

const normalizeFacetValue = (value: string): string => value.trim().toLocaleLowerCase();

const buildFacets = (values: readonly string[]): AssetMetadataFacet[] => {
  const counts = new Map<string, number>();
  for (const rawValue of values) {
    const value = normalizeFacetValue(rawValue);
    if (!value) continue;
    counts.set(value, (counts.get(value) ?? 0) + 1);
  }

  return [...counts.entries()]
    .map(([value, count]) => ({ value, count }))
    .sort((left, right) => right.count - left.count || left.value.localeCompare(right.value));
};

/**
 * Builds stable facet data from asset metadata. Facets intentionally reference metadata values,
 * never storage paths, so the same asset can move between logical scopes without changing search UX.
 */
export const buildAssetMetadataFacets = (assets: readonly AssetSearchMetadata[]): AssetMetadataFacets => ({
  assetTypes: buildFacets(assets.map((asset) => asset.assetType)),
  tags: buildFacets(assets.flatMap((asset) => asset.tags ?? [])),
});

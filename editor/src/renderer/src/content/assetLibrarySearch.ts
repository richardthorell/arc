import type { AssetItem } from '../services/editorHostTypes';

export type AssetSearchFacet = {
  kind: AssetItem['kind'];
  count: number;
};

export type AssetSearchQuery = {
  text?: string;
  kinds?: readonly AssetItem['kind'][];
  tags?: readonly string[];
};

export type AssetSearchResult = {
  assets: readonly AssetItem[];
  facets: readonly AssetSearchFacet[];
};

const normalize = (value: string): string => value.trim().toLocaleLowerCase();

const searchableText = (asset: AssetItem): string[] => [
  asset.name,
  asset.title ?? '',
  asset.description ?? '',
  asset.path,
  ...(asset.tags ?? []),
];

const matchesText = (asset: AssetItem, text: string): boolean => {
  const terms = normalize(text).split(/\s+/).filter(Boolean);
  if (terms.length === 0) return true;
  const haystack = searchableText(asset).map(normalize).join('\n');
  return terms.every((term) => haystack.includes(term));
};

const matchesTags = (asset: AssetItem, tags: readonly string[]): boolean => {
  if (tags.length === 0) return true;
  const assetTags = new Set((asset.tags ?? []).map(normalize));
  return tags.map(normalize).filter(Boolean).every((tag) => assetTags.has(tag));
};

/**
 * Applies metadata predicates without depending on Content Browser layout state.
 * Results retain the input ordering so identical asset snapshots always produce
 * identical search output. Facets describe the text/tag-matched population before
 * a kind filter is applied, allowing the UI to show useful alternative kinds.
 */
export function searchAssetLibrary(
  assets: readonly AssetItem[],
  query: AssetSearchQuery,
): AssetSearchResult {
  const metadataMatches = assets.filter(
    (asset) => matchesText(asset, query.text ?? '') && matchesTags(asset, query.tags ?? []),
  );

  const facetCounts = new Map<AssetItem['kind'], number>();
  for (const asset of metadataMatches) {
    facetCounts.set(asset.kind, (facetCounts.get(asset.kind) ?? 0) + 1);
  }

  const kinds = new Set(query.kinds ?? []);
  const filtered = kinds.size === 0
    ? metadataMatches
    : metadataMatches.filter((asset) => kinds.has(asset.kind));

  const facets = [...facetCounts.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([kind, count]) => ({ kind, count }));

  return { assets: filtered, facets };
}

export type AssetLibraryVirtualViewId = 'favorites' | 'recent' | 'downloads' | 'search-results';

export type AssetLibraryVirtualView = {
  id: AssetLibraryVirtualViewId;
  label: string;
  persistent: boolean;
  acceptsMembershipChanges: boolean;
};

const virtualViews: readonly AssetLibraryVirtualView[] = [
  { id: 'favorites', label: 'Favorites', persistent: true, acceptsMembershipChanges: true },
  { id: 'recent', label: 'Recent', persistent: true, acceptsMembershipChanges: false },
  { id: 'downloads', label: 'Downloads', persistent: true, acceptsMembershipChanges: false },
  { id: 'search-results', label: 'Search Results', persistent: false, acceptsMembershipChanges: false },
];

/**
 * Virtual views are projections over assets, not storage scopes. Keeping the
 * descriptor separate from paths/mounts prevents a view from becoming part of
 * stable asset identity and allows the same asset to appear in many views.
 */
export const assetLibraryVirtualViews = (): readonly AssetLibraryVirtualView[] => virtualViews;

export const assetLibraryVirtualView = (id: AssetLibraryVirtualViewId): AssetLibraryVirtualView => {
  const view = virtualViews.find((candidate) => candidate.id === id);
  if (!view) throw new Error(`Unknown asset library virtual view: ${id satisfies never}`);
  return view;
};

export type AssetLibraryVirtualMembership = Readonly<Record<string, readonly string[]>>;

/**
 * Resolve stable asset IDs for a virtual view without cloning or rewriting the
 * underlying asset records. Search results are supplied per query and remain
 * transient rather than being persisted as collection membership.
 */
export const assetIdsForVirtualView = (
  id: AssetLibraryVirtualViewId,
  membership: AssetLibraryVirtualMembership,
  searchResults: readonly string[] = [],
): readonly string[] => {
  const ids = id === 'search-results' ? searchResults : (membership[id] ?? []);
  return [...new Set(ids)];
};

import type { AssetItem } from '../services/editorHostTypes';

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

const persistentVirtualViewIds = virtualViews.filter((view) => view.persistent).map((view) => view.id);

/**
 * Virtual views are projections over assets, not storage scopes. Keeping the
 * descriptor separate from paths/mounts prevents a view from becoming part of
 * stable asset identity and allows the same asset to appear in many views.
 */
export const assetLibraryVirtualViews = (): readonly AssetLibraryVirtualView[] => virtualViews;

export const assetLibraryVirtualView = (id: AssetLibraryVirtualViewId): AssetLibraryVirtualView => {
  const view = virtualViews.find((candidate) => candidate.id === id);
  if (!view) throw new Error(`Unknown asset library virtual view: ${id}`);
  return view;
};

export type AssetLibraryVirtualMembership = Readonly<Record<string, readonly string[]>>;

const uniqueAssetIds = (ids: readonly string[]): string[] => [...new Set(ids.filter((id) => id.length > 0))];

/**
 * Normalize the persistent part of virtual-view state before it crosses an
 * editor-state persistence boundary. Transient views and unknown future keys
 * are intentionally excluded so Search Results can never leak into saved state.
 */
export const persistentVirtualMembership = (
  membership: AssetLibraryVirtualMembership,
): AssetLibraryVirtualMembership => {
  const persistent: Record<string, readonly string[]> = {};
  for (const id of persistentVirtualViewIds) {
    persistent[id] = uniqueAssetIds(membership[id] ?? []);
  }
  return persistent;
};

/**
 * Restore persisted membership defensively. Malformed values are ignored and
 * all known persistent views are present in the result, which gives settings
 * owners a deterministic shape across upgrades.
 */
export const restorePersistentVirtualMembership = (value: unknown): AssetLibraryVirtualMembership => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return persistentVirtualMembership({});

  const record = value as Record<string, unknown>;
  const membership: Record<string, readonly string[]> = {};
  for (const id of persistentVirtualViewIds) {
    const ids = record[id];
    membership[id] = Array.isArray(ids)
      ? uniqueAssetIds(ids.filter((candidate): candidate is string => typeof candidate === 'string'))
      : [];
  }
  return membership;
};

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
  return uniqueAssetIds(ids);
};

/**
 * Project a virtual view onto the current asset snapshot. Stale membership IDs
 * are ignored and ordering follows the view's stable ID list, so the same
 * AssetItem can safely appear in several views without being copied or having
 * its storage scope rewritten.
 */
export const assetsForVirtualView = <T extends Pick<AssetItem, 'id'>>(
  assets: readonly T[],
  id: AssetLibraryVirtualViewId,
  membership: AssetLibraryVirtualMembership,
  searchResults: readonly string[] = [],
): readonly T[] => {
  const assetsById = new Map(assets.map((asset) => [asset.id, asset] as const));
  return assetIdsForVirtualView(id, membership, searchResults).flatMap((assetId) => {
    const asset = assetsById.get(assetId);
    return asset ? [asset] : [];
  });
};

/**
 * Toggle explicit Favorites membership. Derived views intentionally cannot be
 * changed through this helper so callers cannot accidentally persist search or
 * recency state as user-authored collection membership.
 */
export const setAssetFavorite = (
  membership: AssetLibraryVirtualMembership,
  assetId: string,
  favorite: boolean,
): AssetLibraryVirtualMembership => {
  if (!assetId) return membership;
  const favorites = uniqueAssetIds(membership.favorites ?? []);
  const nextFavorites = favorite
    ? uniqueAssetIds([...favorites, assetId])
    : favorites.filter((candidate) => candidate !== assetId);
  return { ...membership, favorites: nextFavorites };
};

/**
 * Record an asset as most-recently used while keeping the view bounded. The
 * same stable asset ID moves to the front instead of being duplicated.
 */
export const recordRecentAsset = (
  membership: AssetLibraryVirtualMembership,
  assetId: string,
  limit = 50,
): AssetLibraryVirtualMembership => {
  if (!assetId) return membership;
  if (limit <= 0) return { ...membership, recent: [] };
  const recent = uniqueAssetIds(membership.recent ?? []).filter((candidate) => candidate !== assetId);
  return { ...membership, recent: [assetId, ...recent].slice(0, limit) };
};

/**
 * Record a completed remote/import download in deterministic newest-first
 * order. Download membership is derived from completed work rather than being
 * directly user editable.
 */
export const recordDownloadedAsset = (
  membership: AssetLibraryVirtualMembership,
  assetId: string,
): AssetLibraryVirtualMembership => {
  if (!assetId) return membership;
  const downloads = uniqueAssetIds(membership.downloads ?? []).filter((candidate) => candidate !== assetId);
  return { ...membership, downloads: [assetId, ...downloads] };
};

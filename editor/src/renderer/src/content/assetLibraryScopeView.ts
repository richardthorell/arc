import type { AssetItem } from '../services/editorHostTypes';
import { assetLibraryIdentity } from './assetLibraryIdentity';
import type { AssetLibraryMountNavigationItem } from './assetLibraryMounts';
import type { AssetLibraryScopeId } from './assetLibraryScopes';

export type AssetLibraryScopeView = AssetLibraryMountNavigationItem & {
  assetIds: string[];
};

/**
 * Projects authoritative assets into logical Content Browser scopes without
 * deriving identity from the scope. This keeps navigation state independent of
 * storage paths while making unavailable/read-only scopes explicit to the UI.
 */
export function buildAssetLibraryScopeViews(
  navigation: readonly AssetLibraryMountNavigationItem[],
  assets: readonly AssetItem[],
): AssetLibraryScopeView[] {
  const idsByScope = new Map<AssetLibraryScopeId, string[]>();

  for (const asset of assets) {
    const scope = asset.scope ?? 'project';
    const ids = idsByScope.get(scope) ?? [];
    ids.push(assetLibraryIdentity(asset));
    idsByScope.set(scope, ids);
  }

  return navigation.map((item) => ({
    ...item,
    assetIds: item.available ? [...(idsByScope.get(item.scope) ?? [])] : [],
  }));
}

/** Returns the assets visible in a scope while preserving their ARC-owned IDs. */
export function assetsForLibraryScope(view: AssetLibraryScopeView, assets: readonly AssetItem[]): AssetItem[] {
  if (!view.available) return [];
  const visibleIds = new Set(view.assetIds);
  return assets.filter((asset) => visibleIds.has(assetLibraryIdentity(asset)));
}

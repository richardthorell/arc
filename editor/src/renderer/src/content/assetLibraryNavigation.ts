import type { AssetItem } from '../services/editorHostTypes';
import type { AssetLibraryMount } from './assetLibraryMounts';
import {
  assetLibraryScopes,
  assetScopeId,
  type AssetLibraryScope,
  type AssetLibraryScopeId,
} from './assetLibraryScopes';
import {
  assetIdsForVirtualView,
  assetLibraryVirtualViews,
  type AssetLibraryVirtualMembership,
  type AssetLibraryVirtualView,
} from './assetLibraryVirtualViews';

export type AssetLibraryScopeNavigationItem = AssetLibraryScope & {
  kind: 'scope';
  assetCount: number;
  available: boolean;
  mounted: boolean;
};

export type AssetLibraryVirtualNavigationItem = AssetLibraryVirtualView & {
  kind: 'virtual-view';
  assetCount: number;
};

export type AssetLibraryNavigationItem = AssetLibraryScopeNavigationItem | AssetLibraryVirtualNavigationItem;

/**
 * Builds the logical Content Browser scope roots without coupling navigation to
 * physical paths or providers. Project and Built-in remain visible even when
 * empty so a new project has stable navigation; optional shared mounts appear
 * when the host configures a mount or exposes assets from them.
 */
export function buildAssetLibraryScopeNavigation(
  assets: readonly Pick<AssetItem, 'scope'>[],
  mounts: readonly Pick<AssetLibraryMount, 'scope'>[] = [],
): readonly AssetLibraryScopeNavigationItem[] {
  const counts = new Map<AssetLibraryScopeId, number>();
  for (const asset of assets) {
    const scope = assetScopeId(asset);
    counts.set(scope, (counts.get(scope) ?? 0) + 1);
  }

  const mountedScopes = new Set(mounts.map((mount) => mount.scope));

  return assetLibraryScopes.map((scope) => ({
    ...scope,
    kind: 'scope' as const,
    assetCount: counts.get(scope.id) ?? 0,
    mounted: mountedScopes.has(scope.id),
    available: scope.id === 'project' || scope.id === 'builtin' || mountedScopes.has(scope.id) || counts.has(scope.id),
  }));
}

export function visibleAssetLibraryScopeNavigation(
  assets: readonly Pick<AssetItem, 'scope'>[],
  mounts: readonly Pick<AssetLibraryMount, 'scope'>[] = [],
): readonly AssetLibraryScopeNavigationItem[] {
  return buildAssetLibraryScopeNavigation(assets, mounts).filter((scope) => scope.available);
}

/**
 * Builds non-storage navigation roots from stable asset IDs. Virtual views are
 * explicitly tagged so callers cannot accidentally treat them as filesystem
 * scopes or derive storage paths from them. Search Results remain transient and
 * only appear while a query has results.
 */
export function buildAssetLibraryVirtualNavigation(
  membership: AssetLibraryVirtualMembership,
  searchResults: readonly string[] = [],
): readonly AssetLibraryVirtualNavigationItem[] {
  return assetLibraryVirtualViews()
    .map((view) => ({
      ...view,
      kind: 'virtual-view' as const,
      assetCount: assetIdsForVirtualView(view.id, membership, searchResults).length,
    }))
    .filter((view) => view.id !== 'search-results' || view.assetCount > 0);
}

/**
 * Returns one navigation model while preserving the scope/view distinction.
 * Assets may therefore appear in multiple virtual views without changing their
 * storage scope or stable identity.
 */
export function buildAssetLibraryNavigation(
  assets: readonly Pick<AssetItem, 'scope'>[],
  membership: AssetLibraryVirtualMembership,
  searchResults: readonly string[] = [],
  mounts: readonly Pick<AssetLibraryMount, 'scope'>[] = [],
): readonly AssetLibraryNavigationItem[] {
  return [
    ...buildAssetLibraryVirtualNavigation(membership, searchResults),
    ...visibleAssetLibraryScopeNavigation(assets, mounts),
  ];
}

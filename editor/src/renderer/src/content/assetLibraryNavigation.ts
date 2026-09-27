import type { AssetItem } from '../services/editorHostTypes';
import {
  assetLibraryScopes,
  assetScopeId,
  type AssetLibraryScope,
  type AssetLibraryScopeId,
} from './assetLibraryScopes';

export type AssetLibraryScopeNavigationItem = AssetLibraryScope & {
  assetCount: number;
  available: boolean;
};

/**
 * Builds the logical Content Browser scope roots without coupling navigation to
 * physical paths or providers. Project and Built-in remain visible even when
 * empty so a new project has stable navigation; optional shared mounts appear
 * once the host exposes assets from them.
 */
export function buildAssetLibraryScopeNavigation(
  assets: readonly Pick<AssetItem, 'scope'>[],
): readonly AssetLibraryScopeNavigationItem[] {
  const counts = new Map<AssetLibraryScopeId, number>();
  for (const asset of assets) {
    const scope = assetScopeId(asset);
    counts.set(scope, (counts.get(scope) ?? 0) + 1);
  }

  return assetLibraryScopes.map((scope) => ({
    ...scope,
    assetCount: counts.get(scope.id) ?? 0,
    available: scope.id === 'project' || scope.id === 'builtin' || counts.has(scope.id),
  }));
}

export function visibleAssetLibraryScopeNavigation(
  assets: readonly Pick<AssetItem, 'scope'>[],
): readonly AssetLibraryScopeNavigationItem[] {
  return buildAssetLibraryScopeNavigation(assets).filter((scope) => scope.available);
}

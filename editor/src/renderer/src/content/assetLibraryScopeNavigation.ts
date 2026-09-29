import type { AssetItem } from '../services/editorHostTypes';
import { assetLibraryScopes, assetScopeId, type AssetLibraryScopeId } from './assetLibraryScopes';

export type AssetLibraryScopeNavigationItem = {
  id: AssetLibraryScopeId;
  label: string;
  description: string;
  writable: boolean;
  assetCount: number;
};

/**
 * Builds the storage-neutral root navigation model for the Content Browser.
 *
 * Scope ordering comes from the shared scope contract rather than storage
 * discovery so mounts appearing/disappearing cannot reorder the browser UI.
 * Legacy assets without an explicit scope continue to count as Project assets
 * through assetScopeId().
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
    id: scope.id,
    label: scope.label,
    description: scope.description,
    writable: scope.writable,
    assetCount: counts.get(scope.id) ?? 0,
  }));
}

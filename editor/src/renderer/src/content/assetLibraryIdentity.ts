import type { AssetItem } from '../services/editorHostTypes';
import type { AssetLibraryScopeId } from './assetLibraryScopes';

export type AssetLibraryIdentity = Pick<AssetItem, 'id' | 'guid'>;

export type AssetLibraryLogicalReference = {
  assetId: string;
  scope: AssetLibraryScopeId;
  viewId?: string;
};

/**
 * Returns the ARC-owned identity used by every Content Browser view.
 *
 * Scope, folder, provider, and virtual-view membership are deliberately not
 * part of this key. Moving an asset between logical views must therefore not
 * manufacture a new identity.
 */
export function assetLibraryIdentity(asset: AssetLibraryIdentity): string {
  return asset.id;
}

/**
 * Creates a navigation reference without deriving identity from the selected
 * scope or virtual view. Callers may change either presentation dimension while
 * retaining the same assetId.
 */
export function assetLibraryLogicalReference(
  asset: AssetLibraryIdentity,
  scope: AssetLibraryScopeId,
  viewId?: string,
): AssetLibraryLogicalReference {
  return {
    assetId: assetLibraryIdentity(asset),
    scope,
    ...(viewId ? { viewId } : {}),
  };
}

/**
 * Verifies that presentation references still point at the authoritative ARC
 * asset identity. This is intended for integration boundaries where a scope or
 * virtual view is rebuilt independently from the asset snapshot.
 */
export function referencesPreserveAssetIdentity(
  assets: readonly AssetLibraryIdentity[],
  references: readonly Pick<AssetLibraryLogicalReference, 'assetId'>[],
): boolean {
  const identities = new Set(assets.map(assetLibraryIdentity));
  return references.every((reference) => identities.has(reference.assetId));
}

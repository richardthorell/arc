import type { AssetLibraryScopeNavigationItem } from './assetLibraryNavigation';

export type AssetLibraryScopeAccess = 'writable' | 'read-only';

export type AssetLibraryScopePresentation = {
  access: AssetLibraryScopeAccess;
  accessLabel: 'Writable' | 'Read only';
  ariaLabel: string;
  description: string;
};

/**
 * Converts the logical scope contract into explicit Content Browser copy.
 *
 * Keep this presentation derived from the shared scope model so individual
 * browser surfaces do not infer mutability from provider names, paths, or
 * whether a scope currently contains assets.
 */
export function assetLibraryScopePresentation(
  scope: Pick<AssetLibraryScopeNavigationItem, 'label' | 'writable' | 'description' | 'mounted'>,
): AssetLibraryScopePresentation {
  const access: AssetLibraryScopeAccess = scope.writable ? 'writable' : 'read-only';
  const accessLabel = scope.writable ? 'Writable' : 'Read only';
  const mountDescription = scope.mounted ? 'Configured library mount.' : 'Default library scope.';

  return {
    access,
    accessLabel,
    ariaLabel: `${scope.label}, ${accessLabel.toLocaleLowerCase()}`,
    description: `${scope.description} ${accessLabel}. ${mountDescription}`,
  };
}

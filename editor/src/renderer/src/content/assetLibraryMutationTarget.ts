import { assetMutationScope, type AssetLibraryScopeId } from './assetLibraryScopes';

export type AssetLibraryMutationTarget = {
  scope: AssetLibraryScopeId;
  relativeFolder: string;
};

const normalizeRelativeFolder = (folder: string) => {
  const normalized = folder
    .trim()
    .replaceAll('\\', '/')
    .replace(/\/+/g, '/')
    .replace(/^\/|\/$/g, '');
  if (!normalized) return '';

  const segments = normalized.split('/');
  if (segments.some((segment) => segment === '..' || segment === '.')) {
    throw new Error('Asset library mutation folder must be a logical relative path.');
  }
  return segments.join('/');
};

/**
 * Resolves a Content Browser create/import destination without coupling the
 * logical library scope to asset identity or a physical storage path.
 *
 * Read-only/unknown scopes inherit the shared Project fallback policy while
 * writable scopes remain explicit. Folder paths stay logical and relative so
 * the host/storage layer can map them to the configured mount.
 */
export function resolveAssetLibraryMutationTarget(
  selectedScope: AssetLibraryScopeId | string | null | undefined,
  relativeFolder = '',
): AssetLibraryMutationTarget {
  return {
    scope: assetMutationScope(selectedScope),
    relativeFolder: normalizeRelativeFolder(relativeFolder),
  };
}

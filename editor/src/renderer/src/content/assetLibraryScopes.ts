import type { AssetItem } from '../services/editorHostTypes';

export type AssetLibraryScopeId = 'builtin' | 'project' | 'user' | 'organization';

export type AssetLibraryScope = {
  id: AssetLibraryScopeId;
  label: string;
  writable: boolean;
  description: string;
};

export const assetLibraryScopes: readonly AssetLibraryScope[] = [
  {
    id: 'builtin',
    label: 'Built-in',
    writable: false,
    description: 'Engine-provided assets available to every project.',
  },
  { id: 'project', label: 'Project', writable: true, description: 'Assets owned by the current project.' },
  { id: 'user', label: 'User', writable: true, description: 'Assets available to the current user across projects.' },
  {
    id: 'organization',
    label: 'Organization',
    writable: false,
    description: 'Shared organization assets provided by configured library mounts.',
  },
] as const;

const scopeById = new Map(assetLibraryScopes.map((scope) => [scope.id, scope]));

export function assetLibraryScope(id: string | null | undefined): AssetLibraryScope {
  return scopeById.get((id ?? 'project') as AssetLibraryScopeId) ?? scopeById.get('project')!;
}

export function assetScopeId(asset: Pick<AssetItem, 'scope'>): AssetLibraryScopeId {
  return assetLibraryScope(asset.scope).id;
}

export function assetsInScope<T extends Pick<AssetItem, 'scope'>>(
  assets: readonly T[],
  scope: AssetLibraryScopeId,
): T[] {
  return assets.filter((asset) => assetScopeId(asset) === scope);
}

export function isAssetScopeWritable(scope: AssetLibraryScopeId): boolean {
  return assetLibraryScope(scope).writable;
}

/**
 * Resolves the logical scope that should own a Content Browser mutation.
 *
 * Browsing a read-only mount must never make that mount a write target. In
 * those cases ARC keeps mutations project-owned, matching legacy behavior.
 * Writable logical mounts, such as User, remain explicit mutation targets so
 * the UI can route create/import operations without coupling scope to asset
 * identity or physical storage paths.
 */
export function assetMutationScope(scope: AssetLibraryScopeId | string | null | undefined): AssetLibraryScopeId {
  const selected = assetLibraryScope(scope);
  return selected.writable ? selected.id : 'project';
}

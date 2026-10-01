import { assetLibraryScope, type AssetLibraryScopeId } from './assetLibraryScopes';

export type AssetLibraryMount = {
  scope: AssetLibraryScopeId;
  root: string;
  writable: boolean;
};

export type AssetLibraryMountMap = Partial<Record<AssetLibraryScopeId, string>>;

const normalizeMountRoot = (root: string) => root.trim().replaceAll('\\', '/').replace(/\/+/g, '/').replace(/\/$/, '');

/**
 * Builds the renderer-side logical mount table without making a mount path
 * part of asset identity. Missing optional mounts simply stay unavailable;
 * Project remains the compatibility/default library root.
 */
export function buildAssetLibraryMounts(roots: AssetLibraryMountMap): AssetLibraryMount[] {
  const mounts: AssetLibraryMount[] = [];

  for (const scopeId of ['builtin', 'project', 'user', 'organization'] as const) {
    const root = roots[scopeId];
    if (!root) continue;

    const normalizedRoot = normalizeMountRoot(root);
    if (!normalizedRoot) continue;

    const scope = assetLibraryScope(scopeId);
    mounts.push({ scope: scope.id, root: normalizedRoot, writable: scope.writable });
  }

  return mounts;
}

/** Resolve a logical scope to its configured mount without silently aliasing it. */
export function assetLibraryMountForScope(
  mounts: readonly AssetLibraryMount[],
  scope: AssetLibraryScopeId,
): AssetLibraryMount | null {
  return mounts.find((mount) => mount.scope === scope) ?? null;
}

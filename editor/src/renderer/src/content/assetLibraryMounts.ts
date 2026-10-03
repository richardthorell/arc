import { assetLibraryScope, assetLibraryScopes, type AssetLibraryScopeId } from './assetLibraryScopes';

export type AssetLibraryMount = {
  scope: AssetLibraryScopeId;
  root: string;
  writable: boolean;
};

export type AssetLibraryMountMap = Partial<Record<AssetLibraryScopeId, string>>;

export type AssetLibraryMountNavigationItem = {
  scope: AssetLibraryScopeId;
  label: string;
  description: string;
  available: boolean;
  writable: boolean;
};

const normalizeMountRoot = (root: string) => root.trim().replaceAll('\\', '/').replace(/\/+/g, '/').replace(/\/$/, '');

const normalizeRelativeAssetPath = (path: string): string | null => {
  const normalized = path.trim().replaceAll('\\', '/').replace(/\/+/g, '/').replace(/^\.\//, '');
  if (!normalized || normalized.startsWith('/') || /^[A-Za-z]:\//.test(normalized)) return null;

  const segments = normalized.split('/');
  if (segments.some((segment) => segment === '' || segment === '.' || segment === '..')) return null;
  return segments.join('/');
};

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

/**
 * Builds Content Browser navigation state from the authoritative logical mount
 * table. Physical roots stay out of renderer navigation so storage location can
 * change without changing asset identity or leaking host paths into UI state.
 */
export function buildAssetLibraryMountNavigation(
  mounts: readonly AssetLibraryMount[],
): AssetLibraryMountNavigationItem[] {
  const availableScopes = new Set(mounts.map((mount) => mount.scope));
  return assetLibraryScopes.map((scope) => ({
    scope: scope.id,
    label: scope.label,
    description: scope.description,
    available: availableScopes.has(scope.id),
    writable: scope.writable,
  }));
}

/**
 * Resolves a scope-relative asset path through the configured logical mount.
 * The returned physical path is deliberately derived at the storage boundary;
 * callers should keep stable asset identity and logical scope separate from it.
 * Absolute paths and traversal segments are rejected instead of escaping a mount.
 */
export function resolveAssetLibraryMountPath(
  mounts: readonly AssetLibraryMount[],
  scope: AssetLibraryScopeId,
  relativePath: string,
): string | null {
  const mount = assetLibraryMountForScope(mounts, scope);
  const normalizedRelativePath = normalizeRelativeAssetPath(relativePath);
  if (!mount || !normalizedRelativePath) return null;
  return `${mount.root}/${normalizedRelativePath}`;
}

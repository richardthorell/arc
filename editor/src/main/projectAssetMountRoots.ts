import path from 'node:path';

export type ProjectAssetMountRootOptions = {
  projectRoot: string;
  projectAssetRoots: string[];
  builtinAssetsRoot?: string;
  userAssetsRoot?: string;
  organizationAssetsRoot?: string;
};

export type ProjectAssetMountRoots = {
  builtinRoot: string;
  projectRoot: string;
  userRoot: string;
  organizationRoot: string;
};

export type ProjectAssetLogicalMounts = Partial<Record<'builtin' | 'project' | 'user' | 'organization', string>>;

const optionalRoot = (value: string | undefined): string => (value?.trim() ? path.resolve(value) : '');

export const resolveProjectAssetMountRoots = (options: ProjectAssetMountRootOptions): ProjectAssetMountRoots => {
  const projectRoot = path.resolve(options.projectRoot);
  const configuredProjectRoot = options.projectAssetRoots[0]?.trim() || 'Content';
  const resolvedProjectRoot = path.resolve(projectRoot, configuredProjectRoot);
  const relativeProjectRoot = path.relative(projectRoot, resolvedProjectRoot);
  if (
    relativeProjectRoot === '..' ||
    relativeProjectRoot.startsWith(`..${path.sep}`) ||
    path.isAbsolute(relativeProjectRoot)
  )
    throw new Error('The primary project asset root must remain inside the active project');

  return {
    builtinRoot: optionalRoot(options.builtinAssetsRoot),
    projectRoot: resolvedProjectRoot,
    userRoot: optionalRoot(options.userAssetsRoot),
    organizationRoot: optionalRoot(options.organizationAssetsRoot),
  };
};

/**
 * Converts host-owned physical mount roots into the production project-snapshot
 * contract. Unavailable optional mounts stay absent rather than masquerading as
 * empty storage providers, and physical paths remain storage configuration only.
 */
export const projectAssetLogicalMounts = (roots: ProjectAssetMountRoots): ProjectAssetLogicalMounts => ({
  ...(roots.builtinRoot ? { builtin: roots.builtinRoot } : {}),
  project: roots.projectRoot,
  ...(roots.userRoot ? { user: roots.userRoot } : {}),
  ...(roots.organizationRoot ? { organization: roots.organizationRoot } : {}),
});

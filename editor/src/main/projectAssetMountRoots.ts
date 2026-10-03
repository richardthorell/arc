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

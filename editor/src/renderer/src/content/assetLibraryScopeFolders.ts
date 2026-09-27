import type { AssetItem } from '../services/editorHostTypes';
import { assetScopeId, type AssetLibraryScopeId } from './assetLibraryScopes';

export type AssetLibraryFolderNode = {
  name: string;
  path: string;
  children: AssetLibraryFolderNode[];
};

const cleanPath = (path: string) =>
  path
    .replaceAll('\\', '/')
    .replace(/\/+/g, '/')
    .replace(/^\/|\/$/g, '');

const normalizedPath = (path: string) => cleanPath(path).toLocaleLowerCase();

const scopeAliases = (scope: AssetLibraryScopeId, projectRootName: string): readonly string[] => {
  switch (scope) {
    case 'builtin':
      return ['Engine', 'Builtin'];
    case 'project':
      return [projectRootName, 'Content'];
    case 'user':
      return ['User'];
    case 'organization':
      return ['Organization', 'Org'];
  }
};

export function assetLibraryRelativeFolderPath(
  assetPath: string,
  scope: AssetLibraryScopeId,
  projectRootName = 'Content',
): string {
  const segments = cleanPath(assetPath).split('/').filter(Boolean).slice(0, -1);
  if (segments.length === 0) return '';

  const aliases = new Set(
    scopeAliases(scope, projectRootName)
      .filter(Boolean)
      .map((value) => value.toLocaleLowerCase()),
  );
  const rootIndex = segments.findIndex((segment) => aliases.has(segment.replace(/:$/, '').toLocaleLowerCase()));
  return (rootIndex >= 0 ? segments.slice(rootIndex + 1) : segments).join('/');
}

export function buildAssetLibraryFolderTree(
  assets: readonly AssetItem[],
  scope: AssetLibraryScopeId,
  projectRootName = 'Content',
): AssetLibraryFolderNode[] {
  const roots: AssetLibraryFolderNode[] = [];
  const nodes = new Map<string, AssetLibraryFolderNode>();

  for (const asset of assets) {
    if (assetScopeId(asset) !== scope) continue;
    const folder = assetLibraryRelativeFolderPath(asset.path, scope, projectRootName);
    if (!folder) continue;

    let currentPath = '';
    let parent: AssetLibraryFolderNode | null = null;
    for (const segment of folder.split('/').filter(Boolean)) {
      currentPath = currentPath ? `${currentPath}/${segment}` : segment;
      const key = normalizedPath(currentPath);
      let node = nodes.get(key);
      if (!node) {
        node = { name: segment, path: currentPath, children: [] };
        nodes.set(key, node);
        if (parent) parent.children.push(node);
        else roots.push(node);
      }
      parent = node;
    }
  }

  const sortNodes = (items: AssetLibraryFolderNode[]) => {
    items.sort((left, right) => left.name.localeCompare(right.name));
    items.forEach((item) => sortNodes(item.children));
  };
  sortNodes(roots);
  return roots;
}

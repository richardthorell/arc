import { describe, expect, it } from 'vitest';

import type { AssetItem } from '../services/editorHostTypes';
import { assetLibraryRelativeFolderPath, buildAssetLibraryFolderTree } from './assetLibraryScopeFolders';

const asset = (id: string, path: string, scope: AssetItem['scope']): AssetItem =>
  ({ id, guid: id, name: id, path, scope, kind: 'texture', status: 'ready' }) as AssetItem;

describe('asset library scope folders', () => {
  it('strips logical mount aliases without making the mount part of folder identity', () => {
    expect(assetLibraryRelativeFolderPath('Content/Characters/Hero.arcasset', 'project')).toBe('Characters');
    expect(assetLibraryRelativeFolderPath('Engine/Materials/Grid.arcasset', 'builtin')).toBe('Materials');
    expect(assetLibraryRelativeFolderPath('User/Presets/Favorite.arcasset', 'user')).toBe('Presets');
    expect(assetLibraryRelativeFolderPath('Organization/Studio/Shared.arcasset', 'organization')).toBe('Studio');
  });

  it('supports project roots with a custom content directory name', () => {
    expect(assetLibraryRelativeFolderPath('GameAssets/World/Rock.arcasset', 'project', 'GameAssets')).toBe('World');
  });

  it('builds deterministic trees for every logical scope and ignores assets from other scopes', () => {
    const assets = [
      asset('project', 'Content/Zoo/B.arcasset', 'project'),
      asset('user-b', 'User/Brushes/B.arcasset', 'user'),
      asset('user-a', 'User/Brushes/A.arcasset', 'user'),
      asset('user-nested', 'User/Brushes/Natural/C.arcasset', 'user'),
      asset('org', 'Organization/Shared/D.arcasset', 'organization'),
    ];

    expect(buildAssetLibraryFolderTree(assets, 'user')).toEqual([
      {
        name: 'Brushes',
        path: 'Brushes',
        children: [{ name: 'Natural', path: 'Brushes/Natural', children: [] }],
      },
    ]);
    expect(buildAssetLibraryFolderTree(assets, 'organization')).toEqual([
      { name: 'Shared', path: 'Shared', children: [] },
    ]);
  });

  it('treats legacy unscoped assets as project assets through the shared scope contract', () => {
    const legacy = asset('legacy', 'Content/Legacy/A.arcasset', undefined);
    expect(buildAssetLibraryFolderTree([legacy], 'project')).toEqual([
      { name: 'Legacy', path: 'Legacy', children: [] },
    ]);
    expect(buildAssetLibraryFolderTree([legacy], 'user')).toEqual([]);
  });
});

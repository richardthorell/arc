import { describe, expect, it } from 'vitest';

import type { AssetItem } from '../services/editorHostTypes';
import { buildAssetLibraryMountNavigation, buildAssetLibraryMounts } from './assetLibraryMounts';
import { assetsForLibraryScope, buildAssetLibraryScopeViews } from './assetLibraryScopeView';

const asset = (id: string, scope: AssetItem['scope']): AssetItem =>
  ({ id, guid: id, name: id, path: `${scope}/${id}.arcasset`, kind: 'material', status: 'ready', scope }) as AssetItem;

describe('asset library scope views', () => {
  it('projects stable asset IDs into available logical scopes', () => {
    const assets = [asset('builtin-id', 'builtin'), asset('project-id', 'project'), asset('user-id', 'user')];
    const navigation = buildAssetLibraryMountNavigation(
      buildAssetLibraryMounts({ builtin: '/engine', project: '/project', user: '/user' }),
    );

    const views = buildAssetLibraryScopeViews(navigation, assets);

    expect(views.find((view) => view.scope === 'builtin')).toMatchObject({
      available: true,
      writable: false,
      assetIds: ['builtin-id'],
    });
    expect(views.find((view) => view.scope === 'project')).toMatchObject({
      available: true,
      writable: true,
      assetIds: ['project-id'],
    });
    expect(views.find((view) => view.scope === 'user')).toMatchObject({
      available: true,
      writable: true,
      assetIds: ['user-id'],
    });
  });

  it('does not expose assets through an unavailable scope', () => {
    const assets = [asset('organization-id', 'organization')];
    const navigation = buildAssetLibraryMountNavigation(buildAssetLibraryMounts({ project: '/project' }));
    const views = buildAssetLibraryScopeViews(navigation, assets);
    const organization = views.find((view) => view.scope === 'organization')!;

    expect(organization.available).toBe(false);
    expect(organization.assetIds).toEqual([]);
    expect(assetsForLibraryScope(organization, assets)).toEqual([]);
  });

  it('keeps asset identity stable when the same snapshot is projected through scope navigation', () => {
    const projectAsset = asset('stable-guid', 'project');
    const navigation = buildAssetLibraryMountNavigation(buildAssetLibraryMounts({ project: '/first/root' }));
    const first = buildAssetLibraryScopeViews(navigation, [projectAsset]);
    const remounted = buildAssetLibraryMountNavigation(buildAssetLibraryMounts({ project: '/different/root' }));
    const second = buildAssetLibraryScopeViews(remounted, [projectAsset]);

    expect(first.find((view) => view.scope === 'project')?.assetIds).toEqual(['stable-guid']);
    expect(second.find((view) => view.scope === 'project')?.assetIds).toEqual(['stable-guid']);
  });
});
